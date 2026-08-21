@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0" || goto :fail

rem ===========================================================================
rem  build-installer.bat -- one-click Windows installer build.
rem
rem  Pipeline:
rem     1. Build luckfind.exe with the CUDA backend (reuses ..\build-cuda.bat).
rem     2. Ensure vc_redist.x64.exe is present (download from Microsoft if not).
rem     3. Compile luckfind.iss with Inno Setup (ISCC.exe).
rem
rem  Requirements on this (BUILD) machine:
rem     - Visual Studio "Desktop development with C++"   (used by build-cuda.bat)
rem     - NVIDIA CUDA toolkit                            (ditto)
rem     - Inno Setup 6+                                  https://jrsoftware.org/isdl.php
rem
rem  The TARGET machine only needs an NVIDIA driver -- no VS, no CUDA toolkit.
rem ===========================================================================

echo.
echo [Step 1/3] Building luckfind.exe with CUDA ...
call "..\build-cuda.bat"
if errorlevel 1 goto :fail
if not exist "..\target\release\luckfind.exe" (
    echo [ERROR] ..\target\release\luckfind.exe not found after build.
    goto :fail
)

echo.
echo [Step 2/3] Ensuring vc_redist.x64.exe is present ...
set "VCREDIST=%~dp0vc_redist.x64.exe"
if not exist "%VCREDIST%" (
    echo    Downloading from Microsoft ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%VCREDIST%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
    if errorlevel 1 (
        echo    [ERROR] Download failed.  Put vc_redist.x64.exe here manually:
        echo            %VCREDIST%
        echo            URL:  https://aka.ms/vs/17/release/vc_redist.x64.exe
        goto :fail
    )
)
if not exist "%VCREDIST%" (
    echo    [ERROR] vc_redist.x64.exe is missing.
    echo            Download from:  https://aka.ms/vs/17/release/vc_redist.x64.exe
    goto :fail
)
echo    OK: %VCREDIST%

echo.
echo [Step 3/3] Compiling the installer with Inno Setup (ISCC.exe) ...
set "ISCC="
where ISCC >nul 2>&1 && set "ISCC=ISCC"
if not defined ISCC (
    for /d %%P in ("%ProgramFiles(x86)%\Inno Setup*" "%ProgramFiles%\Inno Setup*") do (
        if not defined ISCC if exist "%%P\ISCC.exe" set "ISCC=%%P\ISCC.exe"
    )
)
if not defined ISCC (
    echo    [ERROR] Inno Setup not found.  Install it from:
    echo            https://jrsoftware.org/isdl.php
    goto :fail
)

echo    Using: %ISCC%
"%ISCC%" "%~dp0luckfind.iss"
if errorlevel 1 goto :fail

echo.
echo [OK] Installer written to:  %~dp0output\Luckfind-Setup.exe
echo     Copy this ONE file to the target machine and double-click it.
echo     It will install the VC++ 2015-2022 x64 runtime if missing, then
echo     place luckfind.exe under {app} and create a Start-menu entry.
exit /b 0

:fail
echo.
echo [ERROR] Installer build failed.  See messages above.
exit /b 1
