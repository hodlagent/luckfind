//! GPU-accelerated secp256k1 key search (OpenCL backend).
//!
//! Mirrors `workers::run` but dispatches the hot loop to a GPU. Falls back to
//! the CPU pool automatically when no OpenCL device is available.
//!
//! ────────────────────────────────────────────────────────────────────────
//!  RESPONSIBLE USE: only run this against addresses you own, authorized
//!  "puzzle" ranges, or for legitimate key recovery. Brute-forcing third-party
//!  wallets is theft — this code is dual-use cryptographic engineering only.
//! ────────────────────────────────────────────────────────────────────────

use std::sync::Arc;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

use crate::addrs::{CandidateSet, CandidateSetExt};
use crate::args::RuntimeLimits;
use crate::progress::Progress;
use crate::workers::MatchEvent;

const KERNEL_SRC: &str = include_str!("kernels/secp256k1_search.cl");

#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub platform_index: usize,
    pub device_index: usize,
    pub workgroup_size: usize,   // must divide batch nicely (e.g. 64/128/256)
    pub batch_keys: usize,       // candidate keys per GPU dispatch
    pub match_capacity: usize,   // max matches kept per dispatch
    pub start_key: [u8; 32],     // base private key (big-endian, secret_bytes order)
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            platform_index: 0,
            device_index: 0,
            workgroup_size: 128,
            batch_keys: 1 << 22, // ~4.2M keys / dispatch
            match_capacity: 1024,
            start_key: [0u8; 32],
        }
    }
}

/// Entry point — drop-in sibling of `workers::run`.
pub fn run(
    cfg: &GpuConfig,
    candidates: CandidateSet,
    limits: RuntimeLimits,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    match try_run_gpu(cfg, &candidates, &limits) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("[GPU] OpenCL unavailable ({}); falling back to CPU pool.", e);
            let n = available_parallelism().map(|n| n.get()).unwrap_or(4);
            crate::workers::run(n, candidates, limits)
        }
    }
}

fn try_run_gpu(
    cfg: &GpuConfig,
    candidates: &CandidateSet,
    limits: &RuntimeLimits,
) -> Result<(Arc<Progress>, Vec<MatchEvent>), String> {
    use ocl::core::DeviceType;
    use ocl::{Buffer, Context, Device, Kernel, MemFlags, Platform, Program, Queue, SpatialDims};

    // ── OpenCL setup ────────────────────────────────────────────────────
    let platform = Platform::list()
        .into_iter()
        .nth(cfg.platform_index)
        .ok_or_else(|| "platform index out of range".to_string())?;
    let mut gpu_devices = Device::list(platform, Some(DeviceType::GPU))
        .map_err(|e| e.to_string())?;
    let device = gpu_devices
        .drain(..)
        .nth(cfg.device_index)
        .ok_or_else(|| "no GPU device at given index".to_string())?;
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .map_err(|e| e.to_string())?;
    let queue = Queue::new(&context, device, None).map_err(|e| e.to_string())?;
    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&context)
        .map_err(|e| e.to_string())?;

    // ── upload candidate table (sorted 20-byte hashes) ───────────────────
    let mut targets: Vec<[u8; 20]> = candidates.hashes();
    targets.sort();
    let ntargets = targets.len() as u32;
    let mut target_bytes = Vec::with_capacity(targets.len() * 20);
    for h in &targets {
        target_bytes.extend_from_slice(h);
    }

    // ── precompute the 4-bit window table for G (1..15, Jacobian w/ Z=1) ─
    let table = build_g_table();

    let cap = cfg.match_capacity;
    let table_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_ONLY)
        .len(table.len())
        .copy_host_slice(&table)
        .build()
        .map_err(|e| e.to_string())?;
    let targets_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_ONLY)
        .len(target_bytes.len())
        .copy_host_slice(&target_bytes)
        .build()
        .map_err(|e| e.to_string())?;
    let base_buf: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_ONLY)
        .len(32)
        .build()
        .map_err(|e| e.to_string())?;
    let counter_buf: Buffer<u32> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(1)
        .build()
        .map_err(|e| e.to_string())?;
    let out_priv: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::WRITE_ONLY)
        .len(cap * 32)
        .build()
        .map_err(|e| e.to_string())?;
    let out_comp: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::WRITE_ONLY)
        .len(cap * 33)
        .build()
        .map_err(|e| e.to_string())?;
    let out_unc: Buffer<u8> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::WRITE_ONLY)
        .len(cap * 65)
        .build()
        .map_err(|e| e.to_string())?;
    let out_wid: Buffer<u32> = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::WRITE_ONLY)
        .len(cap)
        .build()
        .map_err(|e| e.to_string())?;

    let kernel = Kernel::builder()
        .program(&program)
        .name("search")
        .queue(queue.clone())
        .arg(&table_buf)
        .arg(&targets_buf)
        .arg(ntargets)
        .arg(&base_buf)
        .arg(0u32) // batch (set per dispatch)
        .arg(cap as u32)
        .arg(0u32) // wid (GPU ordinal)
        .arg(&counter_buf)
        .arg(&out_priv)
        .arg(&out_comp)
        .arg(&out_unc)
        .arg(&out_wid)
        .build()
        .map_err(|e| e.to_string())?;

    // ── dispatch loop ────────────────────────────────────────────────────
    let progress = Arc::new(Progress::new(1));
    let matches = Arc::new(std::sync::Mutex::new(Vec::<MatchEvent>::new()));
    let deadline = limits.duration_secs.map(|s| Instant::now() + Duration::from_secs_f64(s));
    let start = Instant::now();
    let wg = cfg.workgroup_size.max(1);
    let mut base = cfg.start_key;
    let mut total_keys: u64 = 0;

    loop {
        if deadline.map_or(false, |dl| Instant::now() >= dl) {
            break;
        }
        let batch = cfg.batch_keys;
        let gsize = ((batch + wg - 1) / wg) * wg;

        base_buf.write(base.as_slice()).enq().map_err(|e| e.to_string())?;
        counter_buf.write(&[0u32][..]).enq().map_err(|e| e.to_string())?;

        kernel.set_arg(4, batch as u32).map_err(|e| e.to_string())?; // batch
        unsafe {
            kernel
                .cmd()
                .queue(&queue)
                .global_work_size(SpatialDims::One(gsize))
                .enq()
                .map_err(|e| e.to_string())?;
        }

        let mut cnt = [0u32; 1];
        counter_buf.read(&mut cnt[..]).enq().map_err(|e| e.to_string())?;
        let found = (cnt[0] as usize).min(cap);

        if found > 0 {
            let mut pv = vec![0u8; cap * 32];
            let mut cv = vec![0u8; cap * 33];
            let mut uv = vec![0u8; cap * 65];
            let mut wv = vec![0u32; cap];
            out_priv.read(&mut pv[..]).enq().map_err(|e| e.to_string())?;
            out_comp.read(&mut cv[..]).enq().map_err(|e| e.to_string())?;
            out_unc.read(&mut uv[..]).enq().map_err(|e| e.to_string())?;
            out_wid.read(&mut wv[..]).enq().map_err(|e| e.to_string())?;

            if let Ok(mut g) = matches.lock() {
                for i in 0..found {
                    let mut pk = [0u8; 32];
                    pk.copy_from_slice(&pv[i * 32..i * 32 + 32]);
                    g.push(MatchEvent {
                        address: String::new(),
                        private_key: pk,
                        compressed: cv[i * 33..i * 33 + 33].to_vec(),
                        uncompressed: uv[i * 65..i * 65 + 65].to_vec(),
                        worker_id: wv[i],
                        key_index: total_keys + (i as u64),
                        elapsed: start.elapsed().as_secs_f64(),
                    });
                }
            }
        }

        total_keys += batch as u64;
        progress.increment(batch as u64);
        add_to_base(&mut base, batch as u64);
    }

    let matches = match Arc::try_unwrap(matches) {
        Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
        Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
    };
    Ok((progress, matches))
}

/// 256-bit scalar add (big-endian `base` += `add`, wraps mod 2^256).
fn add_to_base(base: &mut [u8; 32], add: u64) {
    let mut acc = add;
    for i in (0..32).rev() {
        let v = base[i] as u64 + (acc & 0xFF);
        base[i] = v as u8;
        acc = (acc >> 8) + (v >> 8);
        if acc == 0 {
            break;
        }
    }
}

/// Build the 4-bit window table for G: entries 1..15 as Jacobian (Z=1).
fn build_g_table() -> Vec<u8> {
    use secp256k1::{PublicKey, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let mut out = Vec::with_capacity(15 * 96);
    for k in 1u32..=15 {
        let sk = SecretKey::from_byte_array({
            let mut b = [0u8; 32];
            b[32 - 4..].copy_from_slice(&k.to_be_bytes());
            b
        })
        .expect("k in range");
        let pk = PublicKey::from_secret_key(&secp, &sk);
        let aff = pk.serialize_uncompressed(); // 0x04 | X(32) | Y(32)
        out.extend_from_slice(&aff[1..33]); // X
        out.extend_from_slice(&aff[33..65]); // Y
        out.extend_from_slice(&[0u8; 31]); // Z = 1 (big-endian)
        out.push(1);
    }
    out
}
