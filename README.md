# CUDA path tracer

Bounded real-GPU path tracing with naive/BVH megakernels, stream-compacted wavefront
queues, and per-material GPU index queues feeding specialized diffuse/metal kernels.
The four modes run the same scene, integrator and counter-keyed samples.

![Actual CUDA render](evidence/grouped-wavefront.png)

## Run

Windows, VS 2022 C++ Build Tools, CUDA 13.0 and a Blackwell NVIDIA GPU are required;
the local build targets `sm_120`. From the repository root:

```powershell
cmd /c tools\build.cmd upstream
cmd /c tools\build.cmd
powershell -NoProfile -ExecutionPolicy Bypass -File tools/verify.ps1
python tools/check-evidence.py
```

`build/path-tracer.exe --doctor` reports the GPU. `build/path-tracer.exe evidence`
renders four 96x64 PPM files, four samples/pixel, eight bounces. Invalid arguments
return nonzero. The reusable lifecycle driver lives in
`.agents/skills/verify-cuda-path-tracer/SKILL.md`. There are no persistent processes;
cleanup preserves the evidence.

The isolated toolchain lives at `%USERPROFILE%/.codex/tools/coursework-cuda/toolkit`:
overlay `cuda_nvcc`, `cuda_cudart`, `cuda_crt`, `cuda_cccl`, `libnvvm` and `libcurand`
Windows x86-64 archives from NVIDIA's official
[13.0.2 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json).
Every archive was byte-count/SHA256 verified before extraction. Tools and NVIDIA
licenses stay outside Git. There is no global PATH mutation, driver installation,
unsupported-compiler bypass or watchdog change.

## Implementation and checks

`wavefront.cu` reuses upstream `vec3.h`. A host-built balanced median BVH is traversed
on the GPU with a checked fixed stack. Materials are cosine-weighted Lambertian and
perfect mirror. Camera jitter provides stochastic antialiasing. Stable path IDs and
keyed `(path,bounce,dimension)` sampling make queue reordering reproducible. Atomic
compaction reserves one full-capacity output slot per survivor. Grouping fills separate
diffuse and metal index queues before specialized shader launches. The owned procedural
scene contains 40 spheres; no external assets are required.

The first default-contraction experiment failed cross-kernel integrator parity;
`--fmad=false` gives the same explicit arithmetic policy across fused and split kernels.
The resulting maximum linear color difference is zero. This reproducibility choice
can affect performance.

[GPU evidence](evidence/verification.txt): RTX 5080 Laptop GPU, compute 12.0, driver
592.01, CUDA compiler 13.0.88, MSVC 14.44.35207. All 4,102 GPU BVH intersections agree
with naive traversal and an independent double-precision CPU oracle, including inside,
tangent, parallel and miss cases. All four renders have finite nonzero lighting,
identical linear colors and identical PPM pixels. Per-bounce counts are
`24576,14265,5718,2736,1634,1079,752,564`; kernels launched are 1/1/25/41 respectively.
Queue IDs are checked for range and duplicates at every bounce; counts match baseline.

One warmup precedes five runs per mode. CUDA event intervals surround the backend,
including synchronization, managed-memory transfers, host gaps and validation; these
are **not isolated kernel timings**. The tiny wavefront workload is slower, and no
speedup or production-scale conclusion is claimed. CI only checks retained historical
GPU artifacts and upstream integrity on CPU; it does not execute CUDA.
Three additional `raycast_kernel` timings bracket only the BVH raycast launch with
CUDA events recorded immediately around it; nonzero resulting hits prove the work.

## Sources and limits

Complete upstream history is retained from Roger Allen's
[Ray Tracing in One Weekend in CUDA](https://github.com/rogerallen/raytracinginoneweekendincuda),
branch `ch12_where_next_cuda`, SHA `ab140b12d4923b75270831baabab5e4d4209f305`.
Original source is unchanged; [original README](README.upstream.md) retains attribution.
The author explicitly dedicates the work to the public domain in the repository
description, captured in [metadata](evidence/upstream-repository.json). GitHub reports
no formal license file, so none is invented for upstream. The underlying Shirley book
code uses [CC0](https://github.com/RayTracing/raytracing.github.io/blob/release/COPYING.txt),
retained in [book-license.txt](evidence/book-license.txt). This is an educational 2018
base, not a maintained production renderer. Existing camera, dielectric, metal and
depth-of-field code is inherited; the new backend implements diffuse and mirror only.

The experiment is informed by the public
[UPenn instructions](https://github.com/CIS5650-Fall-2025/Project3-CUDA-Path-Tracer/blob/main/INSTRUCTION.md)
and [Laine/Karras/Aila 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf).
No university/student skeleton is copied. This is not a submission, grader result or
claim of full assignment completion. No triangle/OBJ loader, emissive geometry, MIS,
large-scene benchmark or production scheduler is included. Original `main.cu` builds
unmodified, but its large default render was not run.
