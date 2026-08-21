@echo off
setlocal EnableDelayedExpansion

rem ===========================================================================
rem  install-vcredist.bat -- install the Microsoft VC++ 2015-2022 x64 runtime.
rem
rem  For the PORTABLE workflow: copy luckfind.exe + vc_redist.x64.exe +
rem  this file to the target machine, then run this file as Administrator.
rem  (Or skip this file entirely and just double-click vc_redist.x64.exe.)
rem
rem  Nothing is downloaded if vc_redist.x64.exe sits next to this script;
rem  otherwise it tries to fetch it from Microsoft.
rem ===========================================================================

echo.
echo [1/3] Checking for Administrator rights ...
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Please run this file "as Administrator".
    exit /b 1
)
echo    OK.

echo.
echo [2/3] Checking whether the VC++ x64 runtime is already installed ...
set "VCINST="
for /f "tokens=3" %%v in ('reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2^>nul ^| findstr /i "0x"') do set "VCINST=%%v"
if /i "!VCINST!"=="0x1" (
    echo    Already installed.  Nothing to do.
    echo.
    echo [OK] luckfind.exe is ready to run.
    exit /b 0
)

set "VCREDIST=%~dp0vc_redist.x64.exe"
if not exist "%VCREDIST%" (
    echo    vc_redist.x64.exe not found next to this script -- downloading ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "try { Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%VCREDIST%' -UseBasicParsing; exit 0 } catch { Write-Host $_.Exception.Message; exit 1 }"
    if errorlevel 1 goto :downloadfail
)
if not exist "%VCREDIST%" goto :downloadfail

echo.
echo [3/3] Installing VC++ 2015-2022 x64 runtime ...
"%VCREDIST%" /install /quiet /norestart
set "RC=%errorlevel%"
if not "%RC%"=="0" (
    echo [ERROR] vc_redist.x64.exe exited with code %RC%.
    exit /b 1
)

echo.
echo [OK] VC++ x64 runtime installed.  luckfind.exe is ready to run.
exit /b 0

:downloadfail
echo [ERROR] Could not obtain vc_redist.x64.exe.
echo         Download it manually from:
echo         https://aka.ms/vs/17/release/vc_redist.x64.exe
echo         then re-run this script.
exit /b 1
