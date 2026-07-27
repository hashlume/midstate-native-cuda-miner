mod cuda;
mod protocol;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use crate::cuda::{BatchEvent, SharedJob, WorkerConfig, WorkerStats};
use crate::protocol::{
    authorize, below_target, parse_job, response_accepted, response_error, response_id, submit,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(name = "midstate-native-cuda-miner", version = VERSION)]
struct Args {
    #[arg(short = 'o', long, default_value = "stratum+tcp://127.0.0.1:3333")]
    pool: String,
    #[arg(short = 'a', long)]
    address: String,
    #[arg(short = 'w', long, default_value = "native-cuda")]
    worker: String,
    /// CUDA blocks. Zero auto-sizes the grid from SM occupancy.
    #[arg(long, default_value_t = 0)]
    blocks: i32,
    #[arg(long, default_value_t = 128)]
    threads: i32,
    /// Independent nonce chains per CUDA thread. Zero selects automatically.
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(i32).range(0..=2))]
    chains_per_thread: i32,
    /// Nonces per GPU batch. Zero auto-selects from detected GPU models.
    #[arg(long, default_value_t = 0)]
    batch: u64,
    #[arg(long, default_value_t = 1_000_000, hide = true)]
    iterations: u32,
    #[arg(long, default_value_t = 2)]
    max_submit_per_batch: usize,
    #[arg(long, default_value_t = 32)]
    max_outstanding_shares: usize,
    #[arg(long, default_value_t = 120)]
    response_timeout_secs: u64,
    #[arg(long)]
    protocol_debug: bool,
}

#[derive(Debug)]
struct PendingShare {
    sent_at: Instant,
    gpu: usize,
    block: bool,
}

#[derive(Default)]
struct SessionStats {
    accepted: AtomicU64,
    rejected: AtomicU64,
    submitted: AtomicU64,
    skipped: AtomicU64,
    blocks_submitted: AtomicU64,
}

fn parse_pool(pool: &str) -> Result<String> {
    let endpoint = pool.strip_prefix("stratum+tcp://").unwrap_or(pool).trim();
    if endpoint.is_empty() || !endpoint.contains(':') {
        bail!("pool must be stratum+tcp://host:port");
    }
    Ok(endpoint.to_string())
}

async fn write_json(writer: &mut tokio::net::tcp::OwnedWriteHalf, message: &Value) -> Result<()> {
    let mut encoded = serde_json::to_vec(message)?;
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    Ok(())
}

fn format_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2} MH/s", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2} kH/s", rate / 1_000.0)
    } else {
        format!("{rate:.2} H/s")
    }
}

fn print_status(
    started: Instant,
    workers: &[Arc<WorkerStats>],
    stats: &SessionStats,
    pending: usize,
    job_id: u64,
) {
    let total_hps: f64 = workers.iter().map(|worker| worker.hps()).sum();
    let total_hashes: u64 = workers
        .iter()
        .map(|worker| worker.hashes.load(Ordering::Relaxed))
        .sum();
    let total_candidates: u64 = workers
        .iter()
        .map(|worker| worker.candidates.load(Ordering::Relaxed))
        .sum();
    let elapsed = started.elapsed().as_secs_f64();
    let average = if elapsed > 0.0 {
        total_hashes as f64 / elapsed
    } else {
        0.0
    };
    eprintln!(
        "status uptime={}s current={} average={} job={} accepted={} rejected={} pending={} submitted={} skipped={} candidates={} block_submits={}",
        elapsed as u64,
        format_rate(total_hps),
        format_rate(average),
        job_id,
        stats.accepted.load(Ordering::Relaxed),
        stats.rejected.load(Ordering::Relaxed),
        pending,
        stats.submitted.load(Ordering::Relaxed),
        stats.skipped.load(Ordering::Relaxed),
        total_candidates,
        stats.blocks_submitted.load(Ordering::Relaxed),
    );
    for (gpu, worker) in workers.iter().enumerate() {
        eprintln!(
            "  gpu{} {} rate={} hashes={} job={}",
            gpu,
            worker.name,
            format_rate(worker.hps()),
            worker.hashes.load(Ordering::Relaxed),
            worker.job_id.load(Ordering::Relaxed),
        );
    }
}

async fn run_connection(
    args: &Args,
    endpoint: &str,
    jobs: &Arc<SharedJob>,
    events: &mut mpsc::Receiver<BatchEvent>,
    workers: &[Arc<WorkerStats>],
    stats: &Arc<SessionStats>,
    stop: &Arc<AtomicBool>,
    generation_seed: &mut u64,
    started: Instant,
) -> Result<()> {
    let stream = TcpStream::connect(endpoint)
        .await
        .with_context(|| format!("connect to {endpoint}"))?;
    stream.set_nodelay(true)?;
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    let authorization = authorize(1, &args.address, &args.worker);
    write_json(&mut writer, &authorization).await?;
    eprintln!("connected to {endpoint}; authorization sent");

    let mut pending = HashMap::<u64, PendingShare>::new();
    let mut next_submit_id = 1000u64;
    let mut current_job = None;
    let mut status_tick = tokio::time::interval(Duration::from_secs(5));
    status_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut reauthorize_tick = tokio::time::interval(Duration::from_secs(10));
    reauthorize_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(());
        }
        tokio::select! {
            biased;

            line = lines.next_line() => {
                let Some(line) = line.context("read Stratum line")? else {
                    bail!("pool closed connection");
                };
                if args.protocol_debug {
                    eprintln!("protocol rx bytes={} prefix={}", line.len(), &line[..line.len().min(160)]);
                }
                let message: Value = serde_json::from_str(&line).context("invalid Stratum JSON")?;
                *generation_seed = generation_seed.wrapping_add(1);
                if let Some(job) = parse_job(&message, *generation_seed)? {
                    let job = jobs.publish(job);
                    eprintln!(
                        "new job {} generation={} midstate={} share_target={} network_target={}",
                        job.id,
                        job.generation,
                        hex::encode(job.midstate),
                        hex::encode(job.share_target),
                        hex::encode(job.network_target),
                    );
                    current_job = Some(job);
                    continue;
                }

                let Some(id) = response_id(&message) else { continue; };
                if id == 1 {
                    if response_accepted(&message) == Some(true) {
                        eprintln!("pool authorization accepted");
                    } else {
                        bail!("pool authorization rejected: {}", response_error(&message).unwrap_or_default());
                    }
                    continue;
                }
                if let Some(share) = pending.remove(&id) {
                    let latency = share.sent_at.elapsed().as_millis();
                    if response_accepted(&message) == Some(true) {
                        stats.accepted.fetch_add(1, Ordering::Relaxed);
                        if args.protocol_debug || share.block {
                            eprintln!("share accepted id={id} gpu={} block={} latency={}ms", share.gpu, share.block, latency);
                        }
                    } else {
                        stats.rejected.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "share rejected id={id} gpu={} block={} latency={}ms error={}",
                            share.gpu,
                            share.block,
                            latency,
                            response_error(&message).unwrap_or_else(|| "unknown".to_string()),
                        );
                    }
                }
            }
            Some(batch) = events.recv() => {
                let Some(job) = current_job.as_ref() else { continue; };
                if batch.generation != job.generation || batch.job_id != job.id {
                    continue;
                }

                let mut blocks = Vec::new();
                let mut shares = Vec::new();
                for candidate in batch.candidates {
                    if below_target(&candidate.hash, &job.network_target) {
                        blocks.push(candidate);
                    } else {
                        shares.push(candidate);
                    }
                }

                let ordinary_room = args.max_outstanding_shares.saturating_sub(pending.len());
                let ordinary_count = shares.len().min(args.max_submit_per_batch).min(ordinary_room);
                stats.skipped.fetch_add((shares.len() - ordinary_count) as u64, Ordering::Relaxed);

                for (candidate, block) in blocks
                    .into_iter()
                    .map(|candidate| (candidate, true))
                    .chain(shares.into_iter().take(ordinary_count).map(|candidate| (candidate, false)))
                {
                    let id = next_submit_id;
                    next_submit_id = next_submit_id.wrapping_add(1).max(1000);
                    let request = submit(id, &args.address, job.id, candidate.nonce, candidate.hash);
                    write_json(&mut writer, &request).await?;
                    pending.insert(id, PendingShare {
                        sent_at: Instant::now(),
                        gpu: batch.gpu,
                        block,
                    });
                    stats.submitted.fetch_add(1, Ordering::Relaxed);
                    if block {
                        stats.blocks_submitted.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "BLOCK CANDIDATE submitted id={} gpu={} job={} nonce={} hash={}",
                            id,
                            batch.gpu,
                            job.id,
                            candidate.nonce,
                            hex::encode(candidate.hash),
                        );
                    } else if args.protocol_debug {
                        eprintln!(
                            "protocol tx submit id={} gpu={} job={} nonce={} pending={} batch_ms={:.1} checked={}",
                            id,
                            batch.gpu,
                            job.id,
                            candidate.nonce,
                            pending.len(),
                            batch.elapsed_ms,
                            batch.checked,
                        );
                    }
                }
            }
            _ = status_tick.tick() => {
                let job_id = current_job.as_ref().map(|job| job.id).unwrap_or(0);
                print_status(started, workers, stats, pending.len(), job_id);
                if args.response_timeout_secs > 0 {
                    if let Some(oldest) = pending.values().map(|share| share.sent_at).min() {
                        if oldest.elapsed() >= Duration::from_secs(args.response_timeout_secs) {
                            bail!("oldest share response timed out with {} pending", pending.len());
                        }
                    }
                }
            }
            _ = reauthorize_tick.tick(), if current_job.is_none() => {
                write_json(&mut writer, &authorization).await?;
                eprintln!("waiting for job; authorization repeated");
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let endpoint = parse_pool(&args.pool)?;
    let stop = Arc::new(AtomicBool::new(false));
    let jobs = Arc::new(SharedJob::default());
    let stats = Arc::new(SessionStats::default());
    let (event_tx, mut event_rx) = mpsc::channel::<BatchEvent>(128);

    let (workers, handles) = cuda::start_workers(
        WorkerConfig {
            blocks: args.blocks,
            threads: args.threads,
            chains_per_thread: args.chains_per_thread,
            batch: args.batch,
            iterations: args.iterations,
        },
        jobs.clone(),
        stop.clone(),
        event_tx,
    )?;
    eprintln!(
        "midstate-native-cuda-miner v{}: {} CUDA GPU(s), one Stratum connection",
        VERSION,
        workers.len()
    );
    for (index, worker) in workers.iter().enumerate() {
        eprintln!("  gpu{} {}", index, worker.name);
    }

    let signal_stop = stop.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        signal_stop.store(true, Ordering::SeqCst);
    });

    let started = Instant::now();
    let mut generation_seed = 0u64;
    while !stop.load(Ordering::Relaxed) {
        jobs.clear();
        match run_connection(
            &args,
            &endpoint,
            &jobs,
            &mut event_rx,
            &workers,
            &stats,
            &stop,
            &mut generation_seed,
            started,
        )
        .await
        {
            Ok(()) if stop.load(Ordering::Relaxed) => break,
            Ok(()) => eprintln!("connection ended; reconnecting in 5 seconds"),
            Err(error) => eprintln!("connection error: {error:#}; reconnecting in 5 seconds"),
        }
        jobs.clear();
        tokio::time::sleep(Duration::from_secs(5)).await;
    }

    stop.store(true, Ordering::SeqCst);
    drop(event_rx);
    for handle in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => eprintln!("GPU worker error: {error:#}"),
            Err(_) => eprintln!("GPU worker panicked"),
        }
    }
    eprintln!("miner stopped");
    Ok(())
}
