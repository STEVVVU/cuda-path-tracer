# Frozen CUDA renderer benchmark

Run from the repository root in an exclusive CPU/GPU slot:

```powershell
./tools/benchmark.ps1 Build
./tools/benchmark.ps1 Smoke
./tools/benchmark.ps1 Measure
```

The build reuses the isolated CUDA 13 toolkit and VS 2022 environment. It compiles the actual `wavefront.cu` renderer once per frozen workload, with Release host code, `sm_120`, `-O2`, and the existing `--fmad=false` arithmetic policy. No alternate rendering implementation is timed.

| Scene | Resolution | Samples/pixel | Spheres |
| --- | --- | --- | --- |
| Original room | 640×360 | 4, 16 | 40 |
| Lattice | 640×360 | 4, 16 | 404 |
| Dense lattice | 640×360 | 4, 16 | 1,604 |
| Original room | 1280×720 | 4 | 40 |

All cases use eight bounces, fixed counter-keyed samples, diffuse/mirror materials and the same camera/integrator. They are owned procedural scenes, not captured production assets. Every executable first compares 4,102 GPU ray intersections with naive traversal and an independent double-precision CPU oracle. All four backends then run with queue-ID and bounce checks, full linear-image parity, finite colors and nonzero lighting. Measured loops omit diagnostic queue validation; each framework sample still checks its full final linear output against the independently checked warm output.

Google Benchmark owns calibration, warmup, seven repetitions, random interleaving and raw JSON reporting. Each of the four backends has a real-time wall fixture and a CUDA-event manual-time fixture; one raycast-only fixture brackets a single actual kernel with CUDA events. Render event intervals include required host scheduling and managed-memory gaps. Wall timings include the real consumer call and output synchronization. Allocation, scene construction, output poisoning and oracle checks are outside the timed region. On this Windows device managed prefetch is unsupported, which is recorded in JSON; migration remains in the operation. Logical managed allocation bytes and launches per render are counters, not physical VRAM or achieved occupancy measurements.

Smoke uses `--benchmark_dry_run`: its single iterations only prove fixture execution. Final Build, all seven dry runs and all seven repeated measurements passed on the RTX 5080 Laptop GPU on 22 October 2026. Each of nine fixtures per workload has seven repetitions; all 441 output checks passed. The original tiny CLI smoke also passed after sharing the renderer implementation. JSON, stderr, stdout, source hashes and binary hashes remain under `evidence/google-benchmark/`. `python tools/summarize-benchmark.py` validates that retained data and curates `docs/evidence/metrics-summary.json`; it never measures work. p95 is nearest-rank across seven repetition means and therefore their maximum, not a per-frame tail estimate. An owned executable has a 180-second timeout; cleanup never stops unrelated processes.

## Dependency provenance

A separate raycast-wait A/B campaign passed Build, all seven Smoke cases and all seven Measure cases on 22 October 2026. Original nine-group receipts are preserved. Reproduce in an exclusive slot:

```powershell
./tools/benchmark.ps1 Build -CompareRaycastSync
./tools/benchmark.ps1 Smoke -CompareRaycastSync
./tools/benchmark.ps1 Measure -CompareRaycastSync
python tools/summarize-benchmark.py --compare-raycast-sync
python tools/test_summarize_benchmark.py
```

This mode retains separate `evidence/google-benchmark-raycast-sync/` receipts and `docs/evidence/metrics-raycast-sync-summary.json`. Its exact thirteen groups are the original four backend wall/event pairs, four compact/grouped candidate wall/event groups, and one raycast kernel group. All 637 measured repetition rows pass: 588 full final-image checks and 49 nonzero-hit raycast checks. Constructor checks exercise four synchronized backends and two candidate paths with full queue-ID/bounce/image validation before timing. Each Smoke/Measure executable also independently checks 4,102 rays. The curator rejects missing/duplicated candidate groups/repetitions and mislabeled wait policy; its retained-input controls exercise those failures. It consumes framework medians and raw repetition ranges, never starts a benchmark or schedules samples.

Only candidate compact/grouped paths omit the post-raycast host wait. Subsequent kernels consume hits in the same default stream; immediate launch checks remain. The initialization synchronization is essential because the CPU next writes a managed count. Grouping synchronization precedes CPU material-size reads; shading synchronization precedes CPU survivor-count reads. Those waits and all validation remain. [NVIDIA's Unified Memory guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html) forbids concurrent CPU access even to a different managed allocation on this Windows model. Final output checks also read managed colors on the CPU between framework samples; no claim of fully resident allocation is made.

The README shows wall comparisons. CUDA render-event median [range] milliseconds from the same campaign are below. They are separately calibrated/interleaved fixtures, not paired measurements of the exact wall iterations, and include stream idle gaps while the host schedules required work.

| Workload | Compact wait | Compact candidate | Grouped wait | Grouped candidate |
| --- | --- | --- | --- | --- |
| Room 360p, 4 spp | 10.021 [9.487,15.249] | 8.603 [8.105,13.393] | 18.080 [12.285,22.519] | 17.120 [12.047,21.311] |
| Room 360p, 16 spp | 30.378 [26.239,65.635] | 24.909 [18.737,27.610] | 63.538 [50.653,102.863] | 55.894 [45.980,100.190] |
| Lattice 360p, 4 spp | 10.601 [9.098,18.995] | 9.431 [8.702,20.981] | 18.947 [18.535,21.504] | 18.629 [17.668,33.728] |
| Lattice 360p, 16 spp | 25.574 [19.298,35.989] | 24.602 [20.646,27.817] | 57.909 [47.871,84.201] | 54.892 [42.665,86.396] |
| Dense 360p, 4 spp | 10.401 [9.732,11.548] | 9.841 [9.598,10.921] | 19.013 [14.835,20.220] | 18.429 [17.873,20.133] |
| Dense 360p, 16 spp | 25.632 [16.687,35.534] | 24.346 [21.418,26.581] | 55.215 [29.392,62.878] | 42.828 [32.354,54.784] |
| Room 720p, 4 spp | 26.170 [21.234,30.018] | 23.765 [19.105,53.293] | 48.700 [40.695,53.073] | 46.025 [37.585,53.193] |

All 28 wall/event A/B ranges overlap, including the large outliers; none is discarded. No gain is established and production/default behavior remains synchronized. All controls, including the 11.518 ms lattice/16-spp raycast repetition outlier, remain in the curated receipt. Source SHA256 is `eabda75f7afff6e550caffb51f626b9d1f537d433c2fc6831a221a61c6361ea7`, fixture SHA256 `b95692a112ce8ee5dcb8ced2e772d35fc035262c86ed9a54b13cb884956943d5`; recorded base commit is `4c30a92f21f452f56e48f2ee498544afc4cb1afd` plus retained source patch. All seven actual binary hashes match the recorded build manifest. GPU/driver are RTX 5080 Laptop/592.01; retained compiler configuration is nvcc13.0.88, MSVC19.44.35228.0, Release host `/O2 /Ob2`, device `-O2 --fmad=false`, sm120. Upstream nvcc warnings in `GetTimeUnitString`/`GetTimeUnitMultiplier` were retained rather than suppressed. Hardware CSV is a pre-run snapshot, not continuous telemetry. Logical managed payload remains at most 457,280,428 bytes; unsupported prefetch, required host reads and stream gaps prevent attributing these ranges to an isolated device kernel or a measured migration fraction.

The CMake fixture downloads the official [Google Benchmark v1.9.5 release](https://github.com/google/benchmark/releases/tag/v1.9.5), commit `192ef10025eb2c4cdd392bc502f0c852196baa48`, with archive SHA256 `f82705a2726d8f6cdcda274b841f6314dbfc6f731cdda06c946f310ec1cc3ad9`. It links the maintained framework only into benchmark executables. The upstream [Apache 2.0 license](https://github.com/google/benchmark/blob/v1.9.5/LICENSE) remains in the fetched source tree. Tests, install targets and GoogleTest fetching are disabled; no global package or runtime configuration is changed. The [official manual-timing guide](https://github.com/google/benchmark/blob/v1.9.5/docs/user_guide.md#manual-timing) documents the CUDA-event adapter, and the same guide documents random interleaving. Pinning and archive verification bound supply changes; they do not constitute a comprehensive dependency security audit.
