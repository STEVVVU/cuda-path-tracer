---
name: verify-cuda-path-tracer
description: Build and drive the bounded CUDA path tracer CLI on a real NVIDIA GPU.
---

## Launch
From the root run `cmd /c tools/build.cmd`. The short-lived executable is the product;
no persistent process is launched. Baseline upstream build uses the same nvcc includes,
substituting `main.cu` for `wavefront.cu`; do not run its large default render.

## Doctor
Run `nvidia-smi` and `build/path-tracer.exe --doctor`. Both must succeed.

## Drive
Run `powershell -NoProfile -ExecutionPolicy Bypass -File tools/verify.ps1`.
This drives real GPU intersection checks and all render backends with identical seeds.

For the frozen larger workload, follow the maintained [Google Benchmark fixture](../../../tools/benchmark/README.md): `tools/benchmark.ps1 Build`, `Smoke`, then `Measure`. Build and GPU phases require an exclusive resource slot. Smoke is a one-iteration framework dry run, not accepted performance evidence. The framework owns warmup, calibration, repetitions, interleaving, and JSON reporting.

The reviewed raycast-wait A/B uses the same three commands with `-CompareRaycastSync`, separate raw receipts and `python tools/summarize-benchmark.py --compare-raycast-sync`. Preserve original nine-group evidence. Its 637 measured rows pass their oracles, but all 28 wall/event A/B ranges overlap; no production default switch or speedup is established. Curator controls run with `python tools/test_summarize_benchmark.py` without launching CUDA.

## Evidence
The driver preserves transcripts and PPM renders under `evidence/`. Compare linear
framebuffers before quantization, finite colors, path IDs and bounce counts.

## Cleanup
CLI processes exit themselves; there are no ports or scratch services. Evidence survives.
After any failure confirm no owned path-tracer process remains. Never kill by name.
