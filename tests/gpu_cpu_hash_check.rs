//! GPU-vs-CPU hash160 cross-check for one specific private key.
//!
//! The scan pipeline hashes the pubkey and compares the RIPEMD-160 result
//! against puzzle targets.  A puzzle address may have been derived from EITHER
//! the compressed (33B `02|03 || X`) or uncompressed (65B `04 || X || Y`)
//! serialisation, so the CPU workers check both — and since the shader now
//! computes both too, the GPU must match either.  This test seeds walker 0 at
//! the target key with each CPU-computed hash as the candidate and confirms the
//! shader independently recomputes that same hash (the match record stores the
//! shader's own RIPEMD-160 words, not a copy of the candidate).  That exercises
//! the full GPU hash path end to end: affine → serialize → SHA256 → RIPEMD160 →
//! byte-swap, for both serialisations.
//!
//! Run with output visible:
//!   cargo test --test gpu_cpu_hash_check -- --nocapture

use luckfind::gpu::{self, GpuScanner};

/// The private key to test (big-endian hex).
const KEY_HEX: &str = "a87b97d2079817651fce36a71c0b761af1676c9a466c2dff6df8c4d1003eceaa";

#[test]
fn gpu_cpu_hash160_compare() {
    let sk_be = scalar_from_hex(KEY_HEX);

    // ── CPU reference: RIPEMD160(SHA256(pubkey)), both serialisations ───────
    let secp = secp256k1::Secp256k1::new();
    let pk = secp256k1::PublicKey::from_secret_key(
        &secp,
        &secp256k1::SecretKey::from_byte_array(sk_be).expect("valid scalar < n"),
    );
    let compressed = pk.serialize();
    let uncompressed = pk.serialize_uncompressed();
    let h_comp = luckfind::btc::hash160(&compressed);
    let h_uncomp = luckfind::btc::hash160(&uncompressed);

    println!("── CPU reference ─────────────────────────────────────────────");
    println!("  private key : {}", KEY_HEX);
    println!("  pubkey c    : {}", hex::encode(compressed));
    println!("  pubkey u    : {}", hex::encode(uncompressed));
    println!("  hash160 (c) : {}", hex::encode(h_comp));
    println!("  hash160 (u) : {}", hex::encode(h_uncomp));

    // ── GPU: probe device once, then scan each serialisation ────────────────
    let probe = gpu::GpuContext::new_blocking(0);
    let device_name = match &probe {
        Ok(c) => c.device_name(),
        Err(e) => {
            println!("── GPU ────────────────────────────────────────────────────");
            println!("  no Metal/WebGPU device, skipping GPU comparison: {e}");
            return;
        }
    };
    println!("── GPU (Metal, shader recomputes both serialisations) ────────");
    println!("  device      : {device_name}");
    drop(probe);

    for (label, cand, want) in [
        ("compressed  ", h_comp, h_comp),
        ("uncompressed", h_uncomp, h_uncomp),
    ] {
        match gpu_recomputed_hash(sk_be, cand) {
            Some(got) => {
                println!("  hash160 {label} : {}", hex::encode(got));
                if got == want {
                    println!("                 ✅ identical to CPU");
                } else {
                    println!("                 ❌ DIFFERS from CPU ({})", hex::encode(want));
                }
                assert_eq!(got, want, "GPU {label} hash160 != CPU reference");
            }
            None => {
                println!(
                    "  hash160 {label} : ❌ no match reported — the shader did not recompute \
                     the {} serialisation's hash against the candidate",
                    if label.starts_with("compressed") { "compressed" } else { "uncompressed" }
                );
                panic!("GPU {label} scan produced no match");
            }
        }
    }
}

/// Seed walker 0 at `sk_be`·G with `candidate` as the single target slot, run
/// one dispatch, and return the shader's own recomputed hash160 for walker 0.
/// `None` = no match reported (shader's hash differed from the candidate).
fn gpu_recomputed_hash(sk_be: [u8; 32], candidate: [u8; 20]) -> Option<[u8; 20]> {
    let ctx = gpu::GpuContext::new_blocking(0).expect("device probed above");
    let candidates = gpu::convert::hash160_to_candidates(&candidate);
    let mut scanner = GpuScanner::new(ctx, &candidates).expect("GpuScanner::new");
    scanner.stride = 1; // step = G, walker 0 checks exactly the seed key
    scanner.num_candidates = 1;
    scanner.steps_per_call = 1;
    // init_random populates initial_scalars so set_initial_state can patch walker 0.
    scanner
        .init_random(luckfind::puzzles::puzzle_set())
        .expect("init_random");
    seed_at(&mut scanner, sk_be);

    let matches = scanner.step().expect("step");
    matches.iter().find(|m| m.thread_id == 0).map(reassemble_hash160)
}

/// Seed walker 0 at `sk_be`·G (Jacobian z=1), step point = G (stride 1).
fn seed_at(scanner: &mut GpuScanner, sk_be: [u8; 32]) {
    let secp = secp256k1::Secp256k1::new();
    let pk = secp256k1::PublicKey::from_secret_key(
        &secp,
        &secp256k1::SecretKey::from_byte_array(sk_be).unwrap(),
    );
    let u = pk.serialize_uncompressed();
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&u[1..33]);
    y.copy_from_slice(&u[33..65]);
    let (step_px, step_py) = gpu::convert::stride_step_point(scanner.stride);
    scanner
        .set_initial_state(
            0,
            gpu::convert::scalar_be_to_limbs(&sk_be),
            gpu::GpuState {
                x: gpu::convert::be_bytes_to_limbs(&x),
                y: gpu::convert::be_bytes_to_limbs(&y),
                z: [1, 0, 0, 0, 0, 0, 0, 0],
                scalar: gpu::convert::scalar_be_to_limbs(&sk_be),
                step_px,
                step_py,
            },
        )
        .expect("set_initial_state");
}

/// Reassemble the shader's big-endian-per-word RIPEMD160 state into the
/// canonical 20-byte digest.
fn reassemble_hash160(m: &gpu::GpuMatchOutput) -> [u8; 20] {
    let mut h = [0u8; 20];
    for i in 0..5 {
        h[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_be_bytes());
    }
    h
}

fn scalar_from_hex(h: &str) -> [u8; 32] {
    let s = h.strip_prefix("0x").unwrap_or(h);
    let raw = hex::decode(s).unwrap();
    let mut b = [0u8; 32];
    b[32 - raw.len()..].copy_from_slice(&raw);
    b
}
