//! Cross-backend private-key → pubkey consistency check.
//!
//! Derives pubkey + hash160 for a private key on the CPU (libsecp256k1), then
//! seeds the WebGPU scanner (and, with `--features cuda`, the CUDA scanner) at
//! that key with candidates {hash160(sk), hash160(sk+1)}. Two step() dispatches
//! exercise:
//!   1. the hash pipeline — the seed key sk is hashed and matched (z=1), and
//!   2. the EC point-add stepping — after P += G the next key sk+1 is matched,
//!      with the reported scalar / affine pubkey re-checked against libsecp256k1.
//!
//! Usage: cargo run --example verify_key [hex_private_key]
//!        cargo run --example verify_key --features cuda [hex_private_key]
//! Default key is the Puzzle 66 solution.

use luckfind::btc;
use luckfind::gpu::{self, convert, GpuMatchOutput, GpuScanner};

const PUZZLE_66: &str = "0000000000000000000000000000000000060f4d11574f5deee49961d9609ac6";

fn main() -> anyhow::Result<()> {
    let key = std::env::args().nth(1).unwrap_or_else(|| PUZZLE_66.to_string());
    let bytes = hex::decode(&key).expect("private key must be hex");
    assert_eq!(bytes.len(), 32, "private key must be 32 bytes");
    let sk_be: [u8; 32] = bytes.try_into().unwrap();
    let sk1_be = convert::scalar_add_be(&sk_be, 1);

    let (h0, px0, py0) = cpu_reference(&sk_be);
    let (h1, px1, py1) = cpu_reference(&sk1_be);

    println!("sk    = {}", hex::encode(&sk_be));
    println!("sk+1  = {}", hex::encode(&sk1_be));
    println!("addr0 = {}", btc::p2pkh_compressed(&cpu_compressed(&sk_be)));
    println!("addr1 = {}", btc::p2pkh_compressed(&cpu_compressed(&sk1_be)));

    let mut exit = 0;
    match verify_webgpu(&sk_be, &h0, &h1, &px0, &py0, &px1, &py1) {
        Ok(Some(true)) => println!("[PASS] WebGPU"),
        Ok(Some(false)) => {
            println!("[FAIL] WebGPU");
            exit = 1;
        }
        Ok(None) => println!("[SKIP] WebGPU (no device)"),
        Err(e) => {
            println!("[ERROR] WebGPU: {e:#}");
            exit = 1;
        }
    }
    match verify_cuda(&sk_be, &h0, &h1, &px0, &py0, &px1, &py1) {
        Ok(Some(true)) => println!("[PASS] CUDA"),
        Ok(Some(false)) => {
            println!("[FAIL] CUDA");
            exit = 1;
        }
        Ok(None) => println!("[SKIP] CUDA (not built / no device)"),
        Err(e) => {
            println!("[ERROR] CUDA: {e:#}");
            exit = 1;
        }
    }
    if exit != 0 {
        std::process::exit(exit);
    }
    Ok(())
}

fn cpu_compressed(sk_be: &[u8; 32]) -> Vec<u8> {
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_byte_array(*sk_be).expect("sk < n");
    secp256k1::PublicKey::from_secret_key(&secp, &sk).serialize().to_vec()
}

fn cpu_reference(sk_be: &[u8; 32]) -> ([u8; 20], [u32; 8], [u32; 8]) {
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_byte_array(*sk_be).expect("sk < n");
    let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
    let h = btc::hash160(&pk.serialize());
    let enc = pk.serialize_uncompressed();
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&enc[1..33]);
    y.copy_from_slice(&enc[33..65]);
    (h, convert::be_bytes_to_limbs(&x), convert::be_bytes_to_limbs(&y))
}

fn verify_match(
    m: &GpuMatchOutput,
    label: &str,
    expected_sk: &[u8; 32],
    expected_px: &[u32; 8],
    expected_py: &[u32; 8],
    expected_h160: &[u8; 20],
) -> bool {
    let scalar = convert::limbs_to_be_bytes(&m.scalar);
    let mut hash = [0u8; 20];
    for i in 0..5 {
        hash[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_be_bytes());
    }
    let ok = scalar == *expected_sk
        && m.pubkey_x == *expected_px
        && m.pubkey_y == *expected_py
        && hash == *expected_h160;
    println!(
        "  {label}: scalar={} pubkey_x={} pubkey_y={} hash160={} -> {}",
        hex::encode(scalar),
        hex::encode(convert::limbs_to_be_bytes(&m.pubkey_x)),
        hex::encode(convert::limbs_to_be_bytes(&m.pubkey_y)),
        hex::encode(hash),
        if ok { "OK" } else { "MISMATCH" },
    );
    ok
}

fn candidates_two(h0: &[u8; 20], h1: &[u8; 20]) -> Vec<[u32; 5]> {
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(h0[4 * i..4 * i + 4].try_into().unwrap());
        cand[1][i] = u32::from_le_bytes(h1[4 * i..4 * i + 4].try_into().unwrap());
    }
    cand
}

fn step_checks(
    scanner: &mut GpuScanner,
    sk_be: &[u8; 32],
    h0: &[u8; 20],
    h1: &[u8; 20],
    px0: &[u32; 8],
    py0: &[u32; 8],
    px1: &[u32; 8],
    py1: &[u32; 8],
) -> anyhow::Result<bool> {
    let m1 = scanner.step()?;
    let m2 = scanner.step()?;
    let mut pass = true;
    match m1.iter().find(|m| m.thread_id == 0) {
        Some(m) => pass &= verify_match(m, "step1/thread0", sk_be, px0, py0, h0),
        None => {
            println!("  step1/thread0: NO MATCH");
            pass = false;
        }
    }
    let sk1 = convert::scalar_add_be(sk_be, 1);
    match m2.iter().find(|m| m.thread_id == 0) {
        Some(m) => pass &= verify_match(m, "step2/thread0", &sk1, px1, py1, h1),
        None => {
            println!("  step2/thread0: NO MATCH");
            pass = false;
        }
    }
    Ok(pass)
}

fn verify_webgpu(
    sk_be: &[u8; 32],
    h0: &[u8; 20],
    h1: &[u8; 20],
    px0: &[u32; 8],
    py0: &[u32; 8],
    px1: &[u32; 8],
    py1: &[u32; 8],
) -> anyhow::Result<Option<bool>> {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            println!("  WebGPU unavailable: {e}");
            return Ok(None);
        }
    };
    println!("  WebGPU device: {}", ctx.device_name());
    let cand = candidates_two(h0, h1);
    let mut scanner = match GpuScanner::new(ctx, &cand) {
        Ok(s) => s,
        Err(e) => {
            println!("  GpuScanner::new failed: {e}");
            return Ok(None);
        }
    };
    scanner.stride = 1;
    scanner.num_candidates = 2;
    scanner.steps_per_call = 1;
    scanner.seed_range(*sk_be)?;
    Ok(Some(step_checks(
        &mut scanner, sk_be, h0, h1, px0, py0, px1, py1,
    )?))
}

#[cfg(feature = "cuda")]
fn verify_cuda(
    sk_be: &[u8; 32],
    h0: &[u8; 20],
    h1: &[u8; 20],
    px0: &[u32; 8],
    py0: &[u32; 8],
    px1: &[u32; 8],
    py1: &[u32; 8],
) -> anyhow::Result<Option<bool>> {
    use luckfind::cuda::CudaScanner;
    if !CudaScanner::probe() {
        println!("  no CUDA device");
        return Ok(None);
    }
    let cand = candidates_two(h0, h1);
    let mut scanner = match CudaScanner::new(&cand) {
        Ok(s) => s,
        Err(e) => {
            println!("  CudaScanner::new failed: {e:#}");
            return Ok(None);
        }
    };
    println!("  CUDA device: {}", scanner.device_name());
    scanner.stride = 1;
    scanner.num_candidates = 2;
    scanner.steps_per_call = 1;
    scanner.seed_range(*sk_be)?;
    let m1 = scanner.step()?;
    let m2 = scanner.step()?;
    let sk1 = convert::scalar_add_be(sk_be, 1);
    let sk2 = convert::scalar_add_be(&sk1, 1);
    let states = scanner.readback_states()?;
    let s0 = &states[0];
    println!(
        "  CUDA diag: thread0 post-step scalar = {} (expected {})",
        hex::encode(convert::limbs_to_be_bytes(&s0.scalar)),
        hex::encode(sk2),
    );
    let mut pass = true;
    match m1.iter().find(|m| m.thread_id == 0) {
        Some(m) => pass &= verify_match(m, "step1/thread0", sk_be, px0, py0, h0),
        None => {
            println!("  step1/thread0: NO MATCH");
            pass = false;
        }
    }
    let sk1 = convert::scalar_add_be(sk_be, 1);
    match m2.iter().find(|m| m.thread_id == 0) {
        Some(m) => pass &= verify_match(m, "step2/thread0", &sk1, px1, py1, h1),
        None => {
            println!("  step2/thread0: NO MATCH");
            pass = false;
        }
    }
    Ok(Some(pass))
}

#[cfg(not(feature = "cuda"))]
fn verify_cuda(
    _sk_be: &[u8; 32],
    _h0: &[u8; 20],
    _h1: &[u8; 20],
    _px0: &[u32; 8],
    _py0: &[u32; 8],
    _px1: &[u32; 8],
    _py1: &[u32; 8],
) -> anyhow::Result<Option<bool>> {
    Ok(None)
}
