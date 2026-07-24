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
Pascal:      sm_61   GTX 10-series, GTX 1070/1080-class cards
Volta:       sm_70   V100-class cards
Turing:      sm_75   RTX 20-series, GTX 16-series, CMP 30/40/50HX
Ampere DC:   sm_80   A100-class cards
Ampere:      sm_86   RTX 30-series, RTX A5000/A6000, CMP 70/90HX
Ada:         sm_89   RTX 40-series, RTX 5000/6000 Ada
Hopper:      sm_90   H100-class cards
Blackwell:   sm_120  RTX 50-series, RTX PRO Blackwell cards
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
- Blackwell GPUs interleave two independent nonce chains per CUDA thread to hide
  the sequential integer dependency chain.
- BLAKE3's seven message schedules are expanded at compile time; the hot loop
  does not load permutation tables or index a generic message array.
- The default launch grid is sized from CUDA occupancy and the GPU's SM count.

The pool sees one logical worker because all GPUs deliberately share one
connection. Per-GPU rates, candidates, jobs, and totals are printed locally.

## Run

```bash
./midstate-native-cuda-miner \
  -o stratum+tcp://n1.us.clorecloud.net:1820 \
  -a YOUR_PAYOUT_ADDRESS \
  -w rig \
  --max-submit-per-batch 2 \
  --max-outstanding-shares 32
```

Add `--protocol-debug` when diagnosing job or share responses.

## Build

Building requires Rust and CUDA 12.8 or newer for RTX 5090 support:

```bash
cargo build --release
```

`--blocks 0` (the default) enables occupancy-based launch sizing. Supplying a
positive value retains manual grid control for benchmarking. On Blackwell,
`--chains-per-thread 0` selects the dual-chain kernel; pass `1` or `2` to force
either path for an A/B benchmark.

`--batch 0` (the default) auto-selects a per-GPU batch size from detected GPU
models. It uses a smaller batch for CMP 30/40/50HX-class cards and a larger
batch for faster Ampere/Ada/Blackwell rigs. Pass `--batch N` to override.

The tagged GitHub release workflow builds CUDA 12.8 packages for Linux x86-64
and Windows x86-64, then publishes both archives with a combined SHA-256
checksum manifest.

## Pool Compatibility

The fourth `mining.submit` parameter contains the GPU-computed final hash. A
standard Midstate pool may ignore it and verify the nonce normally. The private
fast-pool extension can use it to avoid repeating the million-iteration chain
on CPU. Trusting this hash is suitable only for a private pool whose miners are
controlled by the operator.
