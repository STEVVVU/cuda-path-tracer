$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force evidence | Out-Null
& ./build/path-tracer.exe --doctor 2>&1 | Tee-Object evidence/doctor.txt
if ($LASTEXITCODE -ne 0) { throw 'CUDA doctor failed' }
& ./build/path-tracer.exe evidence 2>&1 | Tee-Object evidence/verification.txt
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed' }
foreach ($name in @('naive-megakernel','bvh-megakernel','compact-wavefront','grouped-wavefront')) {
    if (!(Test-Path "evidence/$name.ppm")) { throw "Missing $name image" }
}
$ErrorActionPreference = 'Continue'
& ./build/path-tracer.exe a b *> evidence/invalid-arguments.txt
if ($LASTEXITCODE -eq 0) { throw 'Invalid arguments accepted' }
$savedVisibleDevices = $env:CUDA_VISIBLE_DEVICES
try {
    $env:CUDA_VISIBLE_DEVICES = '-1'
    & ./build/path-tracer.exe --doctor *> evidence/no-visible-gpu.txt
    if ($LASTEXITCODE -eq 0) { throw 'Missing GPU reported success' }
} finally {
    $env:CUDA_VISIBLE_DEVICES = $savedVisibleDevices
}
$ErrorActionPreference = 'Stop'
'PASS failure path returned nonzero; CLI processes exited; evidence retained' | Set-Content evidence/cleanup.txt
