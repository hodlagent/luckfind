//! Worker pool — spawns `n` OS threads, each running a tight check loop.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::addrs::CandidateSet;
use crate::args::RuntimeLimits;
use crate::progress::Progress;

/// Match surfaced by the pool.
#[derive(Debug, Clone)]
pub struct MatchEvent {
    #[allow(dead_code)]
    pub address: String,          // reserved for future: Base58 of the matching p2pkh addr
    pub private_key: [u8; 32],
    pub compressed: Vec<u8>,
    pub uncompressed: Vec<u8>,
    pub worker_id: u32,
    pub chunk_id: Option<u32>,    // puzzle-mode: which worklist chunk this came from (None = lottery)
    pub key_index: u64,
    pub elapsed: f64,
}

pub fn run(
    n_workers: usize,
    candidates: CandidateSet,
    limits: RuntimeLimits,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    let progress = Arc::new(Progress::new(n_workers as u64));
    let matches = Arc::new(std::sync::Mutex::new(Vec::<MatchEvent>::new()));
    let deadline = limits
        .duration_secs
        .map(|s| Instant::now() + Duration::from_secs_f64(s));
    let start = Instant::now();

    let handles: Vec<_> = (0..n_workers)
        .map(|wid| {
            let progress = progress.clone();
            let matches = matches.clone();
            let cand = candidates.clone();

            thread::spawn(move || {
                worker_loop(wid as u32, &cand, &progress, &matches, deadline, start);
            })
        })
        .collect();

    // ── heartbeat ticker (just print line, no channel) ───────────────
    let hb_progress = progress.clone();
    let hb_deadline = deadline;
    let hb_interval = limits.heartbeat_secs;
    let hb_handle = thread::spawn(move || {
        let mut prev_total = 0u64;
        let mut prev_instant = Instant::now();
        loop {
            thread::sleep(Duration::from_secs_f64(hb_interval));
            if hb_deadline.is_some_and(|dl| Instant::now() >= dl) {
                break;
            }
            let total = hb_progress.checked.load(std::sync::atomic::Ordering::Relaxed);
            let alive = hb_progress.workers_alive.load(std::sync::atomic::Ordering::Relaxed);
            let now = Instant::now();
            let dt = now.duration_since(prev_instant).as_secs_f64();
            let rate = if dt > 0.1 { (total - prev_total) as f64 / dt } else { 0.0 };
            prev_total = total;
            prev_instant = now;
            println!(
                "  [HEARTBEAT] Keys: {:>14} | Speed: {:>10} H/s | Workers: {}",
                fmt_comma(total),
                fmt_comma(rate as u64),
                alive,
            );
        }
    });

    for h in handles {
        let _ = h.join();
    }
    drop(hb_handle);  // ticker will see deadline on next wake and exit

    let matches = match Arc::try_unwrap(matches) {
        Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
        Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
    };
    (progress, matches)
}

fn worker_loop(
    wid: u32,
    candidates: &CandidateSet,
    progress: &Progress,
    matches: &std::sync::Mutex<Vec<MatchEvent>>,
    deadline: Option<Instant>,
    start: Instant,
) {
    let secp = secp256k1::Secp256k1::new();

    // 1. 初始化：只在线程启动时调用一次操作系统的随机数，作为这台"割草机"的起点
    let mut sk = new_key();

    // 2. 创建标量常量 "1"，用于每次循环给私钥自增 1
    let tweak = secp256k1::Scalar::from_be_bytes({
        let mut one = [0u8; 32];
        one[31] = 1; // 大端序的 1
        one
    })
    .expect("Scalar 1 is always valid");

    // 3. 初始一次完整标量乘 `pk = sk * G`。之后的循环不再做标量乘，只用
    //    点加 `pk = pk + G` 推进 —— P_{n+1} = (sk+n+1)*G = (sk+n)*G + G。
    //
    //    点加 (~1 次 Jacobian add) 比标量乘 (~256 次 double + ~128 次 add)
    //    便宜 10-20×，这是整个扫描器最大的单项优化来源。
    let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
    let point_g = crate::btc::generator_public_key();

    let mut local_count = 0u64;

    loop {
        // 4. 性能拦截：将 Instant::now() 的系统调用频率降低 ~2000 倍
        if local_count.is_multiple_of(2048) && deadline.is_some_and(|dl| Instant::now() >= dl) {
            break;
        }

        let pk_c = pk.serialize();
        let pk_u = pk.serialize_uncompressed();

        let h_c = crate::btc::hash160(&pk_c);
        let h_u = crate::btc::hash160(&pk_u);

        if candidates.contains(&h_c) || candidates.contains(&h_u) {
            let ev = MatchEvent {
                address: String::new(),
                private_key: sk.secret_bytes(),
                compressed: pk_c.to_vec(),
                uncompressed: pk_u.to_vec(),
                worker_id: wid,
                chunk_id: None,
                key_index: local_count,
                elapsed: start.elapsed().as_secs_f64(),
            };
            if let Ok(mut g) = matches.lock() {
                g.push(ev);
            }
        }

        // 5. 推进到下一个 key：标量 +1（私钥，用于上报），点加 +G（公钥，用于匹配）。
        //
        //    两者必须配对：sk 推进后仍满足 pk == sk * G。sk.add_tweak 是标量
        //    加法（mod n），极便宜；pk.combine(&G) 是单次点加，底层走
        //    secp256k1_ec_pubkey_combine → gej_add_ge，零次 doubling。
        //
        //    add_tweak 失败意味着 sk 溢出曲线订单（扫完整个 2^256 空间）；
        //    combine 失败意味着 pk 是无穷远点（pk == -G，即 sk == n-1，等价）。
        sk = sk
            .add_tweak(&tweak)
            .expect("Tweak overflow impossible at key_index < 2^256");
        pk = pk
            .combine(&point_g)
            .expect("Adding G cannot land on infinity for sk < n-1");
        local_count += 1;

        if local_count.is_multiple_of(1_000) {
            progress.increment(1_000);
        }
    }
}

fn new_key() -> secp256k1::SecretKey {
    use rand::TryRng;
    let mut buf = [0u8; 32];
    loop {
        rand::rngs::SysRng
            .try_fill_bytes(&mut buf)
            .expect("Os entropy source always available");
        if buf.iter().any(|b| *b != 0) && buf.iter().any(|b| *b != 0xff) {
            if let Ok(sk) = secp256k1::SecretKey::from_byte_array(buf) {
                return sk;
            }
        }
    }
}

/// Format an integer with comma separators.
pub fn fmt_comma(n: u64) -> String {
    n.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap_or_default()
        .join(",")
}
