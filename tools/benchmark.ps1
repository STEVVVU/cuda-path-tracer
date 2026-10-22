param(
    [ValidateSet('Build', 'Smoke', 'Measure')]
    [string]$Phase = 'Smoke',
    [switch]$CompareRaycastSync
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo
if ((Get-PSDrive C).Free -lt 10GB) { throw 'Benchmark requires 10 GiB free disk.' }
$build = Join-Path $repo 'build/google-benchmark'
$output = Join-Path $repo $(if ($CompareRaycastSync) { 'evidence/google-benchmark-raycast-sync' } else { 'evidence/google-benchmark' })
$cases = @('room_360p_4spp','room_360p_16spp','lattice_360p_4spp','lattice_360p_16spp','dense_360p_4spp','dense_360p_16spp','room_720p_4spp')
New-Item -ItemType Directory -Force $output | Out-Null
if ($Phase -eq 'Build') {
    $vs = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/Common7/Tools/VsDevCmd.bat'
    $lines = & $env:ComSpec /d /c "call `"$vs`" -arch=amd64 -host_arch=amd64 >nul && set"
    if ($LASTEXITCODE -ne 0) { throw 'VS environment failed.' }
    foreach ($line in $lines) { if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
    $cuda = Join-Path $env:USERPROFILE '.codex/tools/coursework-cuda/toolkit'
    & cmake -S tools/benchmark -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc.exe"
    if ($LASTEXITCODE -ne 0) { throw 'Framework CMake configuration failed.' }
    & cmake --build $build --parallel 2
    if ($LASTEXITCODE -ne 0) { throw 'Framework fixture build failed.' }
    git rev-parse HEAD | Set-Content "$output/source-sha.txt"
    git diff HEAD | Set-Content "$output/source.patch"
    Get-FileHash wavefront.cu, tools/benchmark_main.cu, tools/benchmark/CMakeLists.txt, "$build/*.exe" | Format-List | Out-File "$output/hashes.txt"
} else {
    & nvidia-smi --query-gpu=name,driver_version,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv | Set-Content "$output/hardware-$Phase.csv"
    if ($LASTEXITCODE -ne 0) { throw 'GPU inventory failed.' }
    foreach ($case in $cases) {
        $jsonPath = Join-Path $output "$case.$Phase.json"
        $arguments = @('--benchmark_enable_random_interleaving=true', '--benchmark_repetitions=7', '--benchmark_out_format=json', "--benchmark_out=$jsonPath")
        if (-not $CompareRaycastSync) { $arguments += '--benchmark_filter=^(naive|bvh|compact|grouped)/(wall|event_span)|^raycast/kernel_event' }
        if ($Phase -eq 'Smoke') { $arguments += '--benchmark_dry_run' }
        $process = Start-Process -FilePath "$build/$case.exe" -ArgumentList $arguments -PassThru -WindowStyle Hidden -RedirectStandardOutput "$output/$case.$Phase.log" -RedirectStandardError "$output/$case.$Phase.stderr"
        if (-not $process.WaitForExit(180000)) { Stop-Process -Id $process.Id; throw "Owned Google Benchmark process timed out $case." }
        if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $jsonPath)) { throw "Benchmark failed $case. Retain logs." }
        $result = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
        if ($result.context.source_sha256 -ne (Get-FileHash wavefront.cu).Hash.ToLowerInvariant() -or $result.context.adapter_sha256 -ne (Get-FileHash tools/benchmark_main.cu).Hash.ToLowerInvariant()) { throw "Stale fixture binary $case; rebuild before accepting results." }
        if (@($result.benchmarks | Where-Object error_occurred).Count) { throw "Benchmark reported an oracle/CUDA failure $case." }
        $groups = @($result.benchmarks | Where-Object run_type -EQ 'iteration' | Group-Object run_name)
        $expectedGroups = if ($CompareRaycastSync) { 13 } else { 9 }
        if ($groups.Count -ne $expectedGroups) { throw "Missing benchmark fixtures $case." }
        if ($Phase -eq 'Measure' -and @($groups | Where-Object Count -NE 7).Count) { throw "Missing framework repetitions $case." }
        Write-Output "PASS $case $Phase; framework JSON and oracle transcript retained"
    }
}
