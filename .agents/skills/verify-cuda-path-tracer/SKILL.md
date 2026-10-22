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

## Evidence
The driver preserves transcripts and PPM renders under `evidence/`. Compare linear
framebuffers before quantization, finite colors, path IDs and bounce counts.

## Cleanup
CLI processes exit themselves; there are no ports or scratch services. Evidence survives.
After any failure confirm no owned path-tracer process remains. Never kill by name.
