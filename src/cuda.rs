use std::collections::hash_map::RandomState;
use std::ffi::{c_char, c_void, CStr};
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use tokio::sync::mpsc;

use crate::protocol::Job;

pub const MAX_CANDIDATES: usize = 512;

#[repr(C)]
struct NativeResult {
    count: u32,
    checked: u64,
    elapsed_ms: f64,
    nonce: [u64; MAX_CANDIDATES],
    hash: [[u8; 32]; MAX_CANDIDATES],
}

unsafe extern "C" {
    fn midstate_cuda_device_count() -> i32;
    fn midstate_cuda_device_name(device: i32, output: *mut c_char, output_len: u32) -> i32;
    fn midstate_cuda_worker_create(
        device: i32,
        blocks: i32,
        threads: i32,
        chains_per_thread: i32,
        batch: u64,
        iterations: u32,
    ) -> *mut c_void;
    fn midstate_cuda_worker_config(
        worker: *mut c_void,
        blocks: *mut i32,
        threads: *mut i32,
        chains: *mut i32,
    ) -> i32;
    fn midstate_cuda_worker_set_job(
        worker: *mut c_void,
        midstate: *const u8,
        target: *const u8,
    ) -> i32;
    fn midstate_cuda_worker_mine(worker: *mut c_void, base: u64, result: *mut NativeResult) -> i32;
    fn midstate_cuda_hash_one(
        device: i32,
        midstate: *const u8,
        nonce: u64,
        iterations: u32,
        output: *mut u8,
    ) -> i32;
    fn midstate_cuda_hash_pair(
        device: i32,
        midstate: *const u8,
        nonce: u64,
        iterations: u32,
        output: *mut u8,
    ) -> i32;
    fn midstate_cuda_worker_destroy(worker: *mut c_void);
    fn midstate_cuda_last_error() -> *const c_char;
    fn midstate_cuda_reference_hash_pair(
        midstate: *const u8,
        nonce: u64,
        iterations: u32,
        output: *mut u8,
    );
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub nonce: u64,
    pub hash: [u8; 32],
}

#[derive(Debug)]
pub struct BatchEvent {
    pub gpu: usize,
    pub generation: u64,
    pub job_id: u64,
    pub checked: u64,
    pub elapsed_ms: f64,
    pub candidates: Vec<Candidate>,
}

#[derive(Debug)]
pub struct WorkerStats {
    pub name: String,
    pub hashes: AtomicU64,
    pub batches: AtomicU64,
    pub candidates: AtomicU64,
    pub hps_bits: AtomicU64,
    pub job_id: AtomicU64,
}

impl WorkerStats {
    pub fn hps(&self) -> f64 {
        f64::from_bits(self.hps_bits.load(Ordering::Relaxed))
    }
}

pub struct SharedJob {
    generation: AtomicU64,
    nonce_cursor: AtomicU64,
    process_entropy: u64,
    job: RwLock<Option<Job>>,
}

impl Default for SharedJob {
    fn default() -> Self {
        let state = RandomState::new();
        let mut hasher = state.build_hasher();
        std::process::id().hash(&mut hasher);
        std::time::SystemTime::now().hash(&mut hasher);
        (&state as *const RandomState as usize).hash(&mut hasher);
        Self {
            generation: AtomicU64::new(0),
            nonce_cursor: AtomicU64::new(0),
            process_entropy: hasher.finish(),
            job: RwLock::new(None),
        }
    }
}

impl SharedJob {
    pub fn publish(&self, mut job: Job) -> Job {
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        job.generation = generation;
        self.nonce_cursor.store(
            job_nonce_seed(&job, generation, self.process_entropy),
            Ordering::SeqCst,
        );
        *self.job.write().unwrap() = Some(job.clone());
        job
    }

    pub fn clear(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        *self.job.write().unwrap() = None;
    }

    fn current(&self) -> Option<Job> {
        self.job.read().unwrap().clone()
    }

    fn claim_nonce_range(&self, batch: u64) -> u64 {
        self.nonce_cursor.fetch_add(batch, Ordering::SeqCst)
    }
}

fn job_nonce_seed(job: &Job, generation: u64, process_entropy: u64) -> u64 {
    let mut seed_bytes = [0u8; 8];
    seed_bytes.copy_from_slice(&job.midstate[..8]);
    u64::from_le_bytes(seed_bytes)
        ^ job.id.rotate_left(17)
        ^ generation.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ process_entropy
}

fn native_error() -> String {
    unsafe {
        let pointer = midstate_cuda_last_error();
        if pointer.is_null() {
            "unknown CUDA error".to_string()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    }
}

pub fn device_count() -> Result<usize> {
    let count = unsafe { midstate_cuda_device_count() };
    if count < 0 {
        bail!("CUDA device discovery failed: {}", native_error());
    }
    Ok(count as usize)
}

fn device_name(device: usize) -> Result<String> {
    let mut buffer = [0 as c_char; 256];
    let result = unsafe {
        midstate_cuda_device_name(device as i32, buffer.as_mut_ptr(), buffer.len() as u32)
    };
    if result != 0 {
        bail!("GPU {device} name query failed: {}", native_error());
    }
    Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}

fn extension_hash(midstate: &[u8; 32], nonce: u64, iterations: u32) -> [u8; 32] {
    let mut preimage = [0u8; 40];
    preimage[..32].copy_from_slice(midstate);
    preimage[32..].copy_from_slice(&nonce.to_le_bytes());
    let mut hash = *blake3::hash(&preimage).as_bytes();
    for _ in 0..iterations {
        hash = *blake3::hash(&hash).as_bytes();
    }
    hash
}

fn self_test(device: usize, iterations: u32) -> Result<()> {
    let midstate = [0xa5; 32];
    let nonce = 0x0123_4567_89ab_cdef;
    let expected = extension_hash(&midstate, nonce, iterations);
    let mut actual = [0u8; 32];
    let result = unsafe {
        midstate_cuda_hash_one(
            device as i32,
            midstate.as_ptr(),
            nonce,
            iterations,
            actual.as_mut_ptr(),
        )
    };
    if result != 0 {
        bail!("GPU {device} self-test failed to run: {}", native_error());
    }
    if actual != expected {
        bail!(
            "GPU {device} self-test mismatch: CUDA={} CPU={}",
            hex::encode(actual),
            hex::encode(expected)
        );
    }
    let expected_pair = [
        extension_hash(&midstate, nonce, iterations),
        extension_hash(&midstate, nonce.wrapping_add(1), iterations),
    ];
    let mut actual_pair = [0u8; 64];
    let pair_result = unsafe {
        midstate_cuda_hash_pair(
            device as i32,
            midstate.as_ptr(),
            nonce,
            iterations,
            actual_pair.as_mut_ptr(),
        )
    };
    if pair_result != 0 {
        bail!(
            "GPU {device} dual-chain self-test failed to run: {}",
            native_error()
        );
    }
    if actual_pair[..32] != expected_pair[0] || actual_pair[32..] != expected_pair[1] {
        bail!(
            "GPU {device} dual-chain self-test mismatch: CUDA0={} CPU0={} CUDA1={} CPU1={}",
            hex::encode(&actual_pair[..32]),
            hex::encode(expected_pair[0]),
            hex::encode(&actual_pair[32..]),
            hex::encode(expected_pair[1]),
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct WorkerConfig {
    pub blocks: i32,
    pub threads: i32,
    pub chains_per_thread: i32,
    pub batch: u64,
    pub iterations: u32,
}

fn auto_batch_for_devices(names: &[String]) -> u64 {
    if names.iter().any(|name| {
        let upper = name.to_ascii_uppercase();
        upper.contains("CMP 30") || upper.contains("CMP 40") || upper.contains("CMP 50")
    }) {
        32_768
    } else {
        65_536
    }
}

pub fn start_workers(
    config: WorkerConfig,
    jobs: Arc<SharedJob>,
    stop: Arc<AtomicBool>,
    events: mpsc::Sender<BatchEvent>,
) -> Result<(Vec<Arc<WorkerStats>>, Vec<thread::JoinHandle<Result<()>>>)> {
    let count = device_count()?;
    if count == 0 {
        bail!("no NVIDIA CUDA devices found");
    }

    let names = (0..count)
        .map(|gpu| device_name(gpu).with_context(|| format!("GPU {gpu}")))
        .collect::<Result<Vec<_>>>()?;
    let config = if config.batch == 0 {
        let batch = auto_batch_for_devices(&names);
        eprintln!("auto CUDA batch={} for {} GPU(s)", batch, count);
        WorkerConfig { batch, ..config }
    } else {
        config
    };

    let mut stats = Vec::with_capacity(count);
    let mut handles = Vec::with_capacity(count);
    for (gpu, name) in names.into_iter().enumerate() {
        let gpu_stats = Arc::new(WorkerStats {
            name,
            hashes: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            candidates: AtomicU64::new(0),
            hps_bits: AtomicU64::new(0.0f64.to_bits()),
            job_id: AtomicU64::new(0),
        });
        stats.push(gpu_stats.clone());

        let jobs = jobs.clone();
        let stop = stop.clone();
        let events = events.clone();
        let config = WorkerConfig { ..config };
        handles.push(thread::spawn(move || {
            self_test(gpu, config.iterations)?;
            let worker = unsafe {
                midstate_cuda_worker_create(
                    gpu as i32,
                    config.blocks,
                    config.threads,
                    config.chains_per_thread,
                    config.batch,
                    config.iterations,
                )
            };
            if worker.is_null() {
                bail!("GPU {gpu} initialization failed: {}", native_error());
            }
            let (mut launch_blocks, mut launch_threads, mut chains) = (0, 0, 0);
            if unsafe {
                midstate_cuda_worker_config(
                    worker,
                    &mut launch_blocks,
                    &mut launch_threads,
                    &mut chains,
                )
            } != 0
            {
                unsafe { midstate_cuda_worker_destroy(worker) };
                bail!("GPU {gpu} launch configuration query failed");
            }
            eprintln!(
                "gpu{gpu} CUDA launch blocks={launch_blocks} threads={launch_threads} chains/thread={chains}"
            );

            let run = || -> Result<()> {
                let mut active_generation = 0;
                let mut active_job = 0;
                let mut share_target = [0xff; 32];
                share_target[0] = 0x00;
                share_target[1] = 0x0f;
                while !stop.load(Ordering::Relaxed) {
                    let Some(job) = jobs.current() else {
                        thread::sleep(Duration::from_millis(100));
                        continue;
                    };
                    if job.generation != active_generation {
                        let result = unsafe {
                            midstate_cuda_worker_set_job(
                                worker,
                                job.midstate.as_ptr(),
                                share_target.as_ptr(),
                            )
                        };
                        if result != 0 {
                            bail!("GPU {gpu} job setup failed: {}", native_error());
                        }
                        active_generation = job.generation;
                        active_job = job.id;
                        gpu_stats.job_id.store(job.id, Ordering::Relaxed);
                    }

                    let base = jobs.claim_nonce_range(config.batch);
                    let mut native: NativeResult = unsafe { std::mem::zeroed() };
                    let result = unsafe { midstate_cuda_worker_mine(worker, base, &mut native) };
                    if result != 0 {
                        bail!("GPU {gpu} batch failed: {}", native_error());
                    }
                    let count = usize::min(native.count as usize, MAX_CANDIDATES);
                    let candidates = (0..count)
                        .map(|index| Candidate {
                            nonce: native.nonce[index],
                            hash: native.hash[index],
                        })
                        .collect::<Vec<_>>();
                    let hps = if native.elapsed_ms > 0.0 {
                        native.checked as f64 / (native.elapsed_ms / 1000.0)
                    } else {
                        0.0
                    };
                    gpu_stats
                        .hashes
                        .fetch_add(native.checked, Ordering::Relaxed);
                    gpu_stats.batches.fetch_add(1, Ordering::Relaxed);
                    gpu_stats
                        .candidates
                        .fetch_add(count as u64, Ordering::Relaxed);
                    gpu_stats.hps_bits.store(hps.to_bits(), Ordering::Relaxed);

                    if active_generation == job.generation && active_job == job.id {
                        events
                            .blocking_send(BatchEvent {
                                gpu,
                                generation: job.generation,
                                job_id: job.id,
                                checked: native.checked,
                                elapsed_ms: native.elapsed_ms,
                                candidates,
                            })
                            .map_err(|_| anyhow!("network event channel closed"))?;
                    }
                }
                Ok(())
            }();

            unsafe { midstate_cuda_worker_destroy(worker) };
            run
        }));
    }

    Ok((stats, handles))
}

#[cfg(test)]
mod tests {
    use super::{auto_batch_for_devices, extension_hash, midstate_cuda_reference_hash_pair};

    const IV: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const SCHEDULE: [[usize; 16]; 7] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
        [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
        [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
        [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
        [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
        [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
    ];

    fn mix(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    }

    fn compress(input: [u32; 8], word8: u32, word9: u32, length: u32) -> [u32; 8] {
        let mut message = [0u32; 16];
        message[..8].copy_from_slice(&input);
        message[8] = word8;
        message[9] = word9;
        let mut v = [0u32; 16];
        v[..8].copy_from_slice(&IV);
        v[8..].copy_from_slice(&IV);
        v[12] = 0;
        v[13] = 0;
        v[14] = length;
        v[15] = 11;
        for schedule in SCHEDULE {
            mix(
                &mut v,
                0,
                4,
                8,
                12,
                message[schedule[0]],
                message[schedule[1]],
            );
            mix(
                &mut v,
                1,
                5,
                9,
                13,
                message[schedule[2]],
                message[schedule[3]],
            );
            mix(
                &mut v,
                2,
                6,
                10,
                14,
                message[schedule[4]],
                message[schedule[5]],
            );
            mix(
                &mut v,
                3,
                7,
                11,
                15,
                message[schedule[6]],
                message[schedule[7]],
            );
            mix(
                &mut v,
                0,
                5,
                10,
                15,
                message[schedule[8]],
                message[schedule[9]],
            );
            mix(
                &mut v,
                1,
                6,
                11,
                12,
                message[schedule[10]],
                message[schedule[11]],
            );
            mix(
                &mut v,
                2,
                7,
                8,
                13,
                message[schedule[12]],
                message[schedule[13]],
            );
            mix(
                &mut v,
                3,
                4,
                9,
                14,
                message[schedule[14]],
                message[schedule[15]],
            );
        }
        std::array::from_fn(|index| v[index] ^ v[index + 8])
    }

    fn word_hash(midstate: [u8; 32], nonce: u64, iterations: u32) -> [u8; 32] {
        let input = std::array::from_fn(|index| {
            u32::from_le_bytes(midstate[index * 4..index * 4 + 4].try_into().unwrap())
        });
        let mut state = compress(input, nonce as u32, (nonce >> 32) as u32, 40);
        for _ in 0..iterations {
            state = compress(state, 0, 0, 32);
        }
        let mut output = [0u8; 32];
        for (index, word) in state.into_iter().enumerate() {
            output[index * 4..index * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        output
    }

    #[test]
    fn word_state_matches_blake3_reference() {
        for iterations in [0, 1, 7] {
            for nonce in [0, 1, 0x0123_4567_89ab_cdef] {
                let midstate = [0xa5; 32];
                assert_eq!(
                    word_hash(midstate, nonce, iterations),
                    extension_hash(&midstate, nonce, iterations)
                );
            }
        }
    }

    #[test]
    fn auto_batch_prefers_smaller_batches_for_slow_cmp_cards() {
        assert_eq!(
            auto_batch_for_devices(&["NVIDIA CMP 50HX".to_string()]),
            32_768
        );
        assert_eq!(
            auto_batch_for_devices(&["NVIDIA CMP 90HX".to_string()]),
            65_536
        );
        assert_eq!(
            auto_batch_for_devices(&["NVIDIA GeForce RTX 5090".to_string()]),
            65_536
        );
    }

    #[test]
    fn unrolled_cuda_schedule_matches_blake3() {
        let midstate = [0x5au8; 32];
        for iterations in [0, 1, 7, 31] {
            for nonce in [0, 1, 0x0123_4567_89ab_cdef] {
                let mut actual = [0u8; 64];
                unsafe {
                    midstate_cuda_reference_hash_pair(
                        midstate.as_ptr(),
                        nonce,
                        iterations,
                        actual.as_mut_ptr(),
                    );
                }
                assert_eq!(actual[..32], extension_hash(&midstate, nonce, iterations));
                assert_eq!(
                    actual[32..],
                    extension_hash(&midstate, nonce.wrapping_add(1), iterations)
                );
            }
        }
    }
}
