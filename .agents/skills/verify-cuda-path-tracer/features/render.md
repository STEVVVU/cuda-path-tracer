# Render verification

## Sub-features
GPU BVH versus naive traversal; compacted and material-grouped wavefront rendering;
stochastic antialiasing; deterministic path sampling and CUDA event timings.

## How to get to it (user POV)
Build the CLI, then run `build/path-tracer.exe evidence`.

## Driving it with PowerShell
`tools/verify.ps1` captures the CLI transcript and requires success and image files.

## Gotchas

The larger frozen campaign uses `tools/benchmark.ps1 Build`, then `Smoke`, then `Measure` in an exclusive CPU/GPU slot. [The framework fixture](../../../../tools/benchmark/README.md) pins Google Benchmark v1.9.5. Seven builds cover the same 40-sphere scene, a 404-sphere lattice, and a 1604-sphere dense scene at 640x360 with four and sixteen samples, plus 1280x720 at four samples. All use eight bounces and the same counter-keyed samples across backends. Before framework timing, each executable independently checks 4,102 rays and full ID/bounce/linear-image parity across all four backends. Smoke uses the framework's one-iteration dry run and is correctness evidence only. Measure excludes ID validation kernels and megakernel bounce-count atomics, verifies the final full linear output of every framework sample, and uses framework warmup, seven repetitions, and random interleaving. Wall time uses `UseRealTime`; render CUDA event intervals include required host scheduling and managed-memory gaps. The separate raycast event interval brackets one actual GPU launch. Final Build, all seven Smoke and all seven Measure cases passed on 22 October 2026, with all 441 repetitions retained in `docs/evidence/metrics-summary.json`. Each owned process has a three-minute timeout; retained evidence survives exit. The measured wavefront paths lose to the BVH megakernel; actual-limiter profiling remains pending.
Requires real CUDA-capable NVIDIA GPU and matching toolchain. Defaults are intentionally
bounded. Timing is educational diagnostic evidence, not a production speed claim.

The separately retained `-CompareRaycastSync` campaign passed all seven Smoke/Measure cases with 13 groups × 7 repetitions = 637 oracle rows (588 final full-image checks, 49 raycast hit checks). Use `python tools/summarize-benchmark.py --compare-raycast-sync` and `python tools/test_summarize_benchmark.py` to validate/curate those existing receipts. Candidate compact/grouped paths omit only the post-raycast host wait; initialization and waits before CPU managed-memory reads remain. Every one of 28 wall/event A/B ranges overlaps, so the synchronized production default is unchanged. See the fixture guide for all event medians/ranges and README for same-run wall/control results. This does not complete hardware-counter limiter profiling.
