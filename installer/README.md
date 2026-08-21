# Luckfind Windows + CUDA 安装包

把 `luckfind.exe`（内嵌 CUDA PTX）做成一个 **Windows 安装包**，目标机器只需 NVIDIA 显卡驱动，
**不需要** Visual Studio、不需要 CUDA Toolkit。安装包会自动补装微软 VC++ 2015-2022 x64 运行库
（即 `vcruntime140.dll` / `msvcp140.dll` —— 这就是所谓"Visual Studio C++"运行库）。

## 为什么目标机器不需要装 VS 和 CUDA Toolkit？

- `build.rs` 在**构建时**用 nvcc 把 `kernel.cu` 编译成 PTX，通过 `include_str!` **内嵌进 exe**。
- `cust` 通过 `libloading` 在**运行时**动态加载显卡驱动自带的 `nvcuda.dll`（CUDA driver）。
- 因此目标机器上只要 NVIDIA 驱动够新，就能对内嵌 PTX 做 JIT 编译，直接跑起来。

**唯一需要额外安装的**是 MSVC 运行库（MSVC 编译产物都需要它），由安装包自动处理。

## 两种交付方式（任选）

| 方式 | 产物 | 目标机器需要做什么 |
|------|------|--------------------|
| **A. 安装包（推荐）** | `Luckfind-Setup.exe`（单个文件） | 双击 → 自动装 VC++ 运行库 + 拷入 exe + 开始菜单项 |
| **B. 便携版** | `luckfind.exe` + `install-vcredist.bat` | 管理员运行 `install-vcredist.bat`（或直接双击 `vc_redist.x64.exe`），再复制 exe 到任意目录 |

## 方式 A：一键构建安装包（在 Windows 构建机上）

### 构建机前置条件
1. **Visual Studio**：勾选"使用 C++ 的桌面开发"（`build-cuda.bat` 依赖它的 MSVC 环境）。
2. **NVIDIA CUDA Toolkit**（nvcc，`build-cuda.bat` 会自动探测）。
3. **Rust MSVC toolchain**。
4. **Inno Setup 6+**：<https://jrsoftware.org/isdl.php>（仅构建安装包时需要）。

### 步骤

```bat
cd crates\luckfind\installer
build-installer.bat
```

脚本自动完成三步：

1. 调 `..\build-cuda.bat` 构建 `target\release\luckfind.exe`（CUDA 特性、PTX 内嵌）。
2. 若当前目录没有 `vc_redist.x64.exe`，从微软官方下载
   （`https://aka.ms/vs/17/release/vc_redist.x64.exe`）。
3. 用 `ISCC.exe` 编译 `luckfind.iss`，产出 **`output\Luckfind-Setup.exe`**。

> 若已装好 Inno Setup，也可手动执行：
> `ISCC.exe luckfind.iss`（前提：`target\release\luckfind.exe` 和 `vc_redist.x64.exe` 就位）。

### 安装包行为
- 检测注册表 `HKLM/HKCU\...\VC\Runtimes\x64\Installed`，缺失才装运行库（静默、不重启）。
- `luckfind.exe` 默认装到 `C:\Program Files\Luckfind\`，并生成开始菜单项。
- 已装过运行库的机器会直接跳过该步，秒装。

## 方式 B：便携版（直接复制 exe 的场景）

1. 构建 `luckfind.exe`：`..\build-cuda.bat`。
2. 拷贝这三个文件到目标机器同一目录：
   `luckfind.exe`、`vc_redist.x64.exe`、`install-vcredist.bat`。
3. 在目标机器上**右键 → 以管理员身份运行** `install-vcredist.bat`（已装则自动跳过）。
4. 之后 `luckfind.exe` 可复制到任何目录直接运行。

也可以只带 `luckfind.exe`，让目标机器用户自行双击微软官方的 `vc_redist.x64.exe`。

## 安装后使用

```bat
"C:\Program Files\Luckfind\luckfind.exe" --gpu_framework cuda
```

启动日志应出现 `[CUDA] lottery worker up on NVIDIA ...`。
若之前跑过 WebGPU 版，首次启动建议 `build-cuda.bat clean` 后重装，避免 build-script 缓存留下旧 stub PTX。

## 目标机器清单

- Windows 10/11 x64（ARM64 请改用 x64 模拟或自行另编）。
- **NVIDIA 显卡 + 较新的驱动**（CUDA 11.8+ 驱动，内含 CUDA driver `nvcuda.dll`）。
- 其余（VC++ 运行库）由安装包自动补装。

## 常见问题

| 现象 | 处理 |
|------|------|
| 运行报 `VCRUNTIME140.dll 缺失` / `0xc000007b` | 运行库没装上。管理员运行 `install-vcredist.bat` 或手动双击 `vc_redist.x64.exe` |
| 启动报 `CUDA initialization failed` | NVIDIA 驱动太旧或驱动没装。去 <https://www.nvidia.com/drivers> 更新驱动 |
| 启动报 `Failed to load CUDA PTX module` | **目标机驱动太旧，带不动构建机 nvcc 产出的 PTX ISA 版本**（如 CUDA 12.8/13.x 编的 PTX 需较新驱动）。升级目标机 NVIDIA 驱动即可；或构建机换老版本 CUDA Toolkit 重编（PTX 版本更老，兼容更广的驱动）。新报错信息会打印显卡算力 + PTX 版本/目标架构，据此对照 [CUDA-驱动版本表](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| 启动报 `CUDA kernel was not compiled` | 构建机 nvcc/MSVC 环境不对，PTX 是 stub。用 `build-cuda.bat clean` 全量重编，确认构建日志里有 nvcc 编译成功提示 |

## 目录结构

```
installer/
  luckfind.iss            Inno Setup 脚本（安装包定义）
  build-installer.bat     一键构建安装包（build exe → 下载运行库 → 编译）
  install-vcredist.bat    便携版：目标机器上补装 VC++ 运行库
  vc_redist.x64.exe       微软 VC++ 2015-2022 x64 运行库（脚本自动下载）
  output/Luckfind-Setup.exe   最终安装包
```
