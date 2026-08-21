;=============================================================================
; luckfind.iss -- Inno Setup script for the Luckfind CUDA scanner.
;
; Produces a single-file installer that:
;   1. Installs the CUDA-enabled luckfind.exe (PTX embedded at build time).
;   2. Installs the Microsoft VC++ 2015-2022 Redistributable (x64) silently
;      IF the registry shows it is missing (the only runtime the exe needs).
;
; The target machine does NOT need Visual Studio or the CUDA toolkit:
; the embedded PTX is JIT-compiled by the NVIDIA driver at first run, and
; cust loads nvcuda.dll (which ships with the driver) via libloading.
;
; Build machine requirements:
;   - Windows x64, CUDA build done first (..\build-cuda.bat)
;   - Inno Setup 6+        https://jrsoftware.org/isdl.php
;   - vc_redist.x64.exe in this directory (build-installer.bat fetches it)
;
; Compile:  ISCC.exe luckfind.iss     (or just run build-installer.bat)
;=============================================================================

#define MyAppName "Luckfind"
#define MyAppVersion "0.1.0"
#define MyAppExeName "luckfind.exe"

[Setup]
AppId={{4A3C6E2B-8F1D-4A9C-B2E7-5D8F9A0B1C2D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Luckfind
AppComments=Bitcoin dormant address lottery scanner (CUDA / WebGPU)
DefaultDirName={autopf}\Luckfind
DisableProgramGroupPage=yes
OutputDir=.\output
OutputBaseFilename=Luckfind-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
MinVersion=10.0
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Run]
Filename: "{tmp}\vc_redist.x64.exe"; \
  Parameters: "/install /quiet /norestart"; \
  StatusMsg: "Installing Microsoft Visual C++ 2015-2022 Redistributable (x64) ..."; \
  Check: NotVCInstalled; \
  Flags: waituntilterminated

[Icons]
Name: "{autoprograms}\Luckfind"; Filename: "{app}\{#MyAppExeName}"

[Code]
// VC++ 2015/2017/2019/2022 all share registry key ...\14.0\VC\Runtimes\x64.
// Check both machine-wide (HKLM) and per-user (HKCU) installs.
function VCInstalled: Boolean;
var
  v: Cardinal;
begin
  Result := False;
  if RegQueryDWordValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', v) then
    Result := (v = 1);
  if not Result then
    if RegQueryDWordValue(HKCU, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', v) then
      Result := (v = 1);
end;

function NotVCInstalled: Boolean;
begin
  Result := not VCInstalled;
end;
