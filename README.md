# luckfind — Bitcoin Dormant Address Lottery

Fast secp256k1 **key-collision scanner** for the unsolved BTC puzzles.  Scans
private keys inside the covered puzzle key space `[2^70, 2^160)`, derives
P2PKH hash160s, and checks them against the 77 embedded unsolved-puzzle targets.
A collision (a scanned key whose derived address matches a puzzle target) is
printed, persisted, and stops the whole run.

- Pure Rust, single binary.  ~1.4M keys/s CPU (5 workers, Apple M4, release).
- Optional GPU accelerator (WebGPU/Metal or CUDA); auto-detected with CPU-only
  fallback.
- Three scan modes: **lottery** (default), **puzzle mode** (`--puzzle <worklist>`),
  and **remote mode** (`--remote <hub-url>`), selectable via CLI flags or a
  TOML config file.
- The 77 puzzle targets are compiled in as a static literal — no network access.

## Collision logic

The scan hot path, in all modes, is the same shape:

```
key (32B) ──► pubkey ──► SHA256 ──► RIPEMD160 ──► hash160 (20B) ──► compare
```

`btc::hash160` (`src/btc.rs`) is SHA256 followed by RIPEMD160, giving the
20-byte P2PKH hash160.  Each key is checked under **both** serialisations —
compressed (33B) and uncompressed (65B) — because either can hash to the target.

### Default lottery mode (no `--puzzle`)

Entry: `src/main.rs` → `workers::run` → `worker_loop` (`src/workers.rs`).

**CPU worker** (`src/workers.rs:202`):

1. **Seed.** `pick_random_puzzle` selects a puzzle weighted by range size
   (`P ∝ 2^bit`, `src/puzzles.rs:60`); `generate_key_in_range` produces a
   random key uniformly inside that puzzle's `[2^n, 2^(n+1))` range.  One full
   scalar multiply `pk = sk·G` is performed up front.
2. **Check.** Serialise compressed + uncompressed, `hash160` both, test
   membership in the puzzle hash160 set (`ps.contains`, an `FxHashMap` → O(1)).
3. **+1 walk.** Advance `sk += 1` and `pk += G` (a point add, ~10-20× cheaper
   than a scalar multiply) so the expensive multiply is paid once per seeded
   key, not per key.
4. **Bounds.** Every 2048 keys the worker checks the deadline, the shared stop
   flag, and whether the key is still inside its range.  When the walk exits
   the range (reaches `end`, hits a solved-puzzle gap, or exceeds `2^160`) it
   re-seeds with a fresh random key and continues.

**GPU worker** (`src/gpu/lottery.rs`, or `src/cuda/lottery.rs`):

- 100k parallel random walkers (`NUM_GPU_THREADS`, stride 1), re-seeded every
  `RESEED_INTERVAL_KEYS = 2^26` keys.  The shader compares each key against all
  77 candidate hash160s at once.
- Every GPU candidate is **re-verified on the CPU**: re-derive the pubkey,
  recompute hash160, look up `puzzle_number_for_hash160`.  The GPU flag is never
  trusted alone (defense in depth against spurious matches).
- Auto-detected at startup; absent a device the run is CPU-only.

### Puzzle mode (`--puzzle <worklist>`)

Deterministic scan of a single puzzle from a worklist file (`.json` one-time
import → `.db` SQLite runtime format).  Entry: `src/main.rs` → `puzzle::run`
(`src/puzzle.rs`).

**CPU worker** (`puzzle_worker`, `src/puzzle.rs:1027`):

- Claims a random *pending* sub-range chunk (splitting wide ones; the worklist
  caps at 2^24 = 4096×4096 chunks), then scans `current → end` sequentially with
  `key += 1` (each claim also flips a coin for forward/reverse direction).
- **Single-target comparison:** `h160_eq(&pk_c, target_h160) || h160_eq(&pk_u, target_h160)`
  (`src/puzzle.rs:1139`) — both pubkey serialisations against the worklist's
  one `target_h160`.
- Wide chunks are parked every `ROTATION_BUDGET` keys (CPU `2^27`, GPU `2^31`)
  and a fresh chunk is claimed, so progress is always resumable.  SQLite holds
  the per-chunk scan position; Ctrl+C writes it back and exits, and the next
  run resumes exactly there.

**GPU worker** (`puzzle_gpu_scan_loop`, `src/puzzle.rs:1552`):

- Dense tiling: `stride = NUM_GPU_THREADS`, a single target candidate, one
  dispatch covers `N·steps_per_call` keys.  Matches are CPU-verified
  (`btc::hash160 == target_h160`, `src/puzzle.rs:1670`) before being recorded.

### Hit handling (both modes)

The first worker to match a target:

1. Records a `MatchEvent` (private key, both pubkeys, worker id, puzzle number)
   — never dropped, even under a poisoned lock.
2. Prints `[HIT] 🎯 puzzle=#N worker=#w … sk_hex=…`.
3. Sets the shared `hit_flag`/`stop_flag` — every worker and the heartbeat
   ticker stop, and the run ends.
4. On exit, `report::flush_match_files` writes `aman_<UTC>.txt` (per-hit,
   UTC-timestamped, never overwrites) into `--output-dir`, and the final report
   prints keys checked / rate / matches.

## Key space & puzzle set

The 77 unsolved puzzles ship compiled in as a JSON literal in
`src/puzzles.rs` (synced with `docs/puzzles.json`).  Each record carries its
target hash160 and key range `[2^n, 2^(n+1))`; puzzles run from #71 (9 bytes)
to #160 (20 bytes), covering `[2^70, 2^160)`.  Key generation is
range-constrained to this space (`PuzzleRange.top_byte_idx` /
`start_top` / `end_top`, `src/puzzles.rs:22`), a ~2^96× reduction vs the full
256-bit space.

When a puzzle is solved it is dropped from the embedded set (both the literal
and `docs/puzzles.json`).

## Build

```sh
cargo build --release                          # CPU + WebGPU (Metal/Vulkan/DX12)
cargo build --release --features cuda          # + CUDA backend (NVIDIA only)
```

The binary is `target/release/luckfind`; `bin/luckfind` in the skill root is
the deployed copy.

### Windows: building the CUDA backend

Requirements: the NVIDIA CUDA toolkit and a Visual Studio installation with
the **"Desktop development with C++"** workload.  nvcc on Windows shells out
to `cl.exe`, which needs the full MSVC environment (`PATH` + `INCLUDE` +
`LIB`) that `vcvars64.bat` sets up — adding `cl.exe` to your PATH alone is
*not* enough.

Simplest: run [`build-cuda.bat`](build-cuda.bat) from the project root.  It
locates the Visual Studio toolchain via `vswhere`, initializes the MSVC
environment, and builds with `--features cuda`:

```bat
build-cuda.bat            :: incremental release build
build-cuda.bat clean      :: full clean rebuild
```

Or do it manually in a terminal:

```bat
call "F:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cargo build --release --features cuda
```

Notes:

- `--features cuda` is **required** — the CUDA runtime is off by default
  (`default = []` in `Cargo.toml`); without it the binary silently falls back
  to the WebGPU backend.
- If the binary still logs `CUDA kernel was not compiled at build time` after
  enabling the toolchain, use `build-cuda.bat clean` (or `cargo clean`):
  cargo's build-script cache can keep a stale stub PTX embedded across
  rebuilds, and `cargo clean -p <pkg>` does not always invalidate it.
- Verify the backend: `target\release\luckfind.exe --gpu_framework cuda`
  should log `[CUDA] lottery worker up on NVIDIA ...` at startup.

**Multi-GPU (mainboard + discrete cards):** all CUDA-capable devices are
enumerated at startup and **one worker thread is spawned per device**, in every
mode (lottery, `--puzzle`, `--remote`).  Startup logs the full inventory, e.g.

```
  [GPU] 2 CUDA device(s) detected -- enabling one GPU worker per device.
  [GPU]   device #0: NVIDIA GeForce RTX 3070
  [GPU]   device #1: NVIDIA GeForce RTX 3090
  [CUDA] lottery worker #0 up on NVIDIA GeForce RTX 3070
  [CUDA] lottery worker #1 up on NVIDIA GeForce RTX 3090
```

Each card runs its own CUDA context (one OS thread per device); a card that
fails to load the kernel (e.g. too old to JIT the embedded PTX) logs a diagnostic
and skips itself — the other cards and the CPU workers continue.  `[HIT]` /
`[claim]` lines carry the device ordinal (`worker=CUDA[0]`, `w=CUDA[1]`) so you
can tell which physical GPU found a key.  Use `nvidia-smi` to confirm the load
spreads across cards.

### Windows: building an installer (distribute to other machines)

To hand `luckfind.exe` to machines that don't have VS / CUDA toolkit installed,
use the [`installer/`](installer/README.md) package.  The target machine only
needs an NVIDIA driver; the installer auto-installs the Microsoft VC++ 2015-2022
x64 runtime (the only MSVC dependency of the exe) and copies the binary:

```bat
cd installer
build-installer.bat        :: build exe → fetch vc_redist → produce Luckfind-Setup.exe
```

A portable option (`luckfind.exe` + `install-vcredist.bat`) is also provided in
[`installer/README.md`](installer/README.md).

## Usage

```sh
bin/luckfind --duration 10                     # lottery, 10 minutes
bin/luckfind --workers 8 --heartbeat 5         # 8 threads, 5s heartbeat
bin/luckfind --gpu_framework auto|webgpu|cuda  # GPU backend (default auto)
bin/luckfind --puzzle bin/71.db                # puzzle mode: resume worklist 71
bin/luckfind --puzzle bin/71.json              # puzzle mode: first run (imports → .db)
bin/luckfind --remote http://192.168.1.10:42069  # remote mode: claim chunks from a LAN hub
bin/luckfind --config luckfind.toml            # run from a TOML config file
bin/luckfind --bench                           # 5s speed benchmark, then exit
```

Remote-mode wire protocol (claim / heartbeat / release / done, plus the hub's
reclaim timing) is documented in [`docs/remote-protocol.md`](docs/remote-protocol.md);
the hub's HTTP API (endpoints, payloads, error codes) lives in the lan-hub repo
(`docs/remote-api.md`).

| Flag | Default | Description |
|---|---|---|
| `--duration`, `-d` | none (forever) | Runtime limit in minutes |
| `--workers`, `-w` | CPUs | OS thread count (default all logical CPUs) |
| `--load`, `-l` | 1.0 | Accepted for back-compat; unused |
| `--output-dir`, `-o` | `.` | Directory for `aman_<UTC>.txt` |
| `--heartbeat`, `-H` | 10.0 | Seconds between progress lines |
| `--gpu_framework` | auto | `auto` / `webgpu` / `cuda`; `cuda` uses every detected GPU, one worker each |
| `--config` | auto | TOML config file; else `luckfind.toml`/`config.toml` in cwd |
| `--puzzle` | — | Puzzle mode worklist (`.json` → `.db`) |
| `--remote` | — | Remote mode hub URL (`http://host:port`) |
| `--cpu-rotate-keys` | 2^27 | Puzzle per-claim CPU rotation budget; `0` = off |
| `--gpu-rotate-keys` | 2^31 | Puzzle per-claim GPU rotation budget; `0` = off |
| `--bench` | off | 5s burn-in, then exit |
| `--profile` | off | Per-stage pipeline profile, then exit |

### Configuration file

Drive the run from a TOML file — no flags needed.  The file is auto-discovered
in the current working directory (`luckfind.toml` first, then `config.toml`),
or given explicitly with `--config <path>`.  Explicit CLI flags always win; the
file fills in whatever the flags omit.  See
[`config.example.toml`](config.example.toml) for a fully-commented copy.

```sh
cd bin && ./luckfind          # auto-loads ./config.toml if present
./luckfind --config conf.toml # or point at a specific file
```

```toml
# Select the scan mode.  "random" is the default.
mode = "puzzle"

# Puzzle-mode rotation (reclaim) budgets — how many keys a worker scans before
# parking its chunk and claiming a fresh random one.  0 disables rotation.
cpu_rotate_keys = 134217728      # 2^27, CPU per-claim budget
gpu_rotate_keys = 2147483648     # 2^31, GPU per-claim budget

# CPU workers: enabled toggles CPU scanning (default true); load is the
# proportion (0.1..=1.0) of the resolved worker thread count (--workers,
# default all logical CPUs) that actually runs.  load = 0.5 leaves half the
# CPU cores idle for other work.
[cpu]
enabled = true
load = 1.0

# GPU collision scanning: engaged only when enabled (default true) AND a GPU
# device is actually available.  false forces a CPU-only run even on a GPU box.
[gpu]
enabled = true

# Which pubkey serialisations the scan compares against the target hash160.
# Both default to true — a puzzle address may have been derived from either the
# compressed (33-byte) or uncompressed (65-byte) form.  Setting one to false
# skips that form's serialize + hash160 entirely on the CPU and GPU hot paths.
# Both false is a config error (exit 2).
[btc]
check_compressed_pk = true
check_uncompressed_pk = true

[puzzle]                         # required when mode = "puzzle"
database = "bin/71.db"           #   → run: bin/luckfind --config luckfind.toml

# [remote]                       # required when mode = "remote"
# uri = "http://192.168.1.10:42069"
```

## Performance

| Path | Rate |
|---|---|
| CPU lottery (release) | ~1.4M keys/s |
| Debug build | ~50-80 kkeys/s |
| GPU lottery (100k walkers) | ~80-150 Mkeys/s target |
| `--bench` / `--profile` | hot-path / per-stage measurement |

Bottleneck is the secp256k1 point-add (~2.4µs/key, ~99% of per-key cost), not
the hash160 (~25ns/key).  Expected time per match exceeds the age of the
universe — this is a lottery, not an income source.

**CAUTION:** Do not fetch or scrape puzzle data from the internet.  Use only
the built-in set or a locally-provided worklist.
