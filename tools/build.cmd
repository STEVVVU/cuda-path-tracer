@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "PKG=%USERPROFILE%\.codex\tools\coursework-cuda\packages"
set "NVCC=%USERPROFILE%\.codex\tools\coursework-cuda\toolkit\bin\nvcc.exe"
set "RT=%PKG%\cuda_cudart-windows-x86_64-13.0.96-archive"
set "CCCL=%PKG%\cuda_cccl-windows-x86_64-13.0.85-archive"
set "CURAND=%PKG%\libcurand-windows-x86_64-10.4.0.35-archive"
if not exist build mkdir build
if "%1"=="upstream" goto upstream
"%NVCC%" -std=c++17 -O2 --fmad=false -arch=sm_120 -o build\path-tracer.exe wavefront.cu
exit /b %errorlevel%
:upstream
"%NVCC%" -std=c++17 -O2 -arch=sm_120 -o build\upstream.exe main.cu
exit /b %errorlevel%
