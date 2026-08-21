@echo off
setlocal EnableDelayedExpansion

rem ===========================================================================
rem  build-cuda.bat — build luckfind with the CUDA backend enabled (Windows).
rem
rem  What it does:
rem    1. Locate the Visual Studio C++ toolchain (vcvars64.bat) via vswhere,
rem       falling back to common install paths.
rem    2. Verify the CUDA toolkit (nvcc) is reachable; warn if not.
rem    3. cargo build --release --features cuda
rem
rem  cl.exe alone is not enough — nvcc needs the full MSVC environment
rem  (PATH + INCLUDE + LIB) that vcvars64.bat sets up, so this script runs
rem  the build from that environment.
rem
rem  Usage:
rem    build-cuda.bat            incremental release build
rem    build-cuda.bat clean      full clean rebuild.  Use this if the binary
rem                              still reports "CUDA kernel was not compiled":
rem                              cargo's build-script cache can keep a stale
rem                              stub PTX alive across `cargo build` and even
rem                              `cargo clean -p <pkg>`.
rem
rem  Optional: set CUDA_ARCH to target a specific GPU (default sm_75, Turing+).
rem    set CUDA_ARCH=sm_89   for RTX 40-series (Ada)
rem    set CUDA_ARCH=sm_86   for RTX 30-series (Ampere)
rem    set CUDA_ARCH=sm_61   for GTX 10-series (Pascal) and older
rem
rem  NOTE for distribution: a binary built with a NEW CUDA toolkit embeds a NEW
rem  PTX ISA version, which requires a RECENT NVIDIA driver on the target
rem  machine (see installer\README.md FAQ "Failed to load CUDA PTX module").
rem ===========================================================================

set "ROOT=%~dp0"
cd /d "%ROOT%" || goto :fail

rem ---- 1. Locate Visual Studio (vcvars64.bat) ---------------------------------
set "VCDIR="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VCDIR=%%i"
)
rem Fallbacks if vswhere is unavailable
if not defined VCDIR (
    for %%v in (2026\Enterprise 2026\Professional 2026\Community 2022\Enterprise 2022\Professional 2022\Community) do (
        if not defined VCDIR if exist "%ProgramFiles%\Microsoft Visual Studio\%%v" set "VCDIR=%ProgramFiles%\Microsoft Visual Studio\%%v"
        if not defined VCDIR if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\%%v" set "VCDIR=%ProgramFiles(x86)%\Microsoft Visual Studio\%%v"
    )
)
if not defined VCDIR (
    echo [ERROR] Visual Studio with the C++ workload not found.
    echo         Install "Desktop development with C++" via the Visual Studio
    echo         Installer, then re-run build-cuda.bat.
    goto :fail
)

set "VCVARS=%VCDIR%\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [ERROR] %VCVARS%
    echo         not found — is the C++ workload installed?
    goto :fail
)

echo [VS] %VCDIR%
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed to initialize the MSVC environment.
    goto :fail
)

rem ---- 2. Check the CUDA toolkit (nvcc) --------------------------------------
set "NVCC_FOUND="
where nvcc >nul 2>&1 && set "NVCC_FOUND=1"
if not defined NVCC_FOUND if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "NVCC_FOUND=1"
if not defined NVCC_FOUND if defined CUDA_HOME if exist "%CUDA_HOME%\bin\nvcc.exe" set "NVCC_FOUND=1"
if not defined NVCC_FOUND (
    for %%v in (v13.3 v12.8 v12.6 v12.4 v12.2 v12.0 v11.8 v11.0) do (
        if not defined NVCC_FOUND if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%v\bin\nvcc.exe" set "NVCC_FOUND=1"
    )
)
if defined NVCC_FOUND (
    for /f "delims=" %%o in ('nvcc --version ^| findstr /c:"release"') do echo [CUDA] %%o
) else (
    echo [WARN] nvcc not found — the build will still succeed, but the CUDA
    echo        backend will be disabled at runtime and luckfind will fall back
    echo        to the WebGPU backend.  Install the CUDA toolkit from
    echo        https://developer.nvidia.com/cuda-downloads
)

rem ---- 3. Build ---------------------------------------------------------------
if /i "%~1"=="clean" (
    echo [BUILD] Full clean build...
    cargo clean
) else (
    echo [BUILD] Release build ^(--features cuda^)...
)
cargo build --release --features cuda
if errorlevel 1 goto :fail

echo.
echo [OK] Build complete: %ROOT%target\release\luckfind.exe
echo [OK] Run:           %ROOT%target\release\luckfind.exe --gpu_framework cuda
echo [OK] Expect:        "N CUDA device(s) detected" plus one "[CUDA] lottery worker #i up on ..." line per GPU in the startup log
exit /b 0

:fail
echo.
echo [ERROR] Build failed.  Fix the reported error and re-run build-cuda.bat
echo         ^(use "build-cuda.bat clean" if the binary keeps reporting a
echo          disabled CUDA backend^).
exit /b 1
