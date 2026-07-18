# Midstate Native CUDA Miner

Multi-GPU NVIDIA miner for the open Midstate Stratum protocol. It follows the
official client's single-session structure while replacing the process-wide
`wgpu` backend with one native CUDA worker per detected GPU.

## Runtime

The release binary requires only a compatible NVIDIA driver. It does not use
Vulkan, `wgpu`, or the CUDA toolkit at runtime. The CUDA runtime is linked into
the release binary.

Release architecture targets:

```text
GTX 1070 Ti: sm_61
RTX 3090:    sm_86
RTX 4090:    sm_89
RTX 5090:    sm_120
```

## Design

- One structured Stratum TCP session shared by all GPUs.
- One native CUDA context and worker thread per detected NVIDIA GPU.
- Unique nonce ranges and stale-job rejection across workers.
- CPU-vs-CUDA million-iteration self-test on every GPU before mining.
- All network-valid block candidates bypass share and pending limits.
- Ordinary share traffic is bounded independently of block discovery.
- Structured JSON parsing for jobs, targets, responses, and errors.
- Clean Ctrl+C shutdown after active CUDA batches finish.
- Register-resident BLAKE3 chaining state across all one million iterations.
- Launch grids are clamped to useful nonce work instead of scheduling empty blocks.

The pool sees one logical worker because all GPUs deliberately share one
connection. Per-GPU rates, candidates, jobs, and totals are printed locally.

## Run

```bash
./midstate-native-cuda-miner \
  -o stratum+tcp://n1.us.clorecloud.net:1820 \
  -a YOUR_PAYOUT_ADDRESS \
  -w rig \
  --batch 131072 \
  --max-submit-per-batch 2 \
  --max-outstanding-shares 32
```

Add `--protocol-debug` when diagnosing job or share responses.

## Build

Building requires Rust and CUDA 12.8 or newer for RTX 5090 support:

```bash
cargo build --release
```

The GitHub release workflow builds in NVIDIA's CUDA 12.8 development image and
publishes a packaged Linux x86-64 binary with a SHA-256 checksum.

## Pool Compatibility

The fourth `mining.submit` parameter contains the GPU-computed final hash. A
standard Midstate pool may ignore it and verify the nonce normally. The private
fast-pool extension can use it to avoid repeating the million-iteration chain
on CPU. Trusting this hash is suitable only for a private pool whose miners are
controlled by the operator.
