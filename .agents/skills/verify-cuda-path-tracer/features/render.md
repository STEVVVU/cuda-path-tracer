# Render verification

## Sub-features
GPU BVH versus naive traversal; compacted and material-grouped wavefront rendering;
stochastic antialiasing; deterministic path sampling and CUDA event timings.

## How to get to it (user POV)
Build the CLI, then run `build/path-tracer.exe evidence`.

## Driving it with PowerShell
`tools/verify.ps1` captures the CLI transcript and requires success and image files.

## Gotchas
Requires real CUDA-capable NVIDIA GPU and matching toolchain. Defaults are intentionally
bounded. Timing is educational diagnostic evidence, not a production speed claim.
