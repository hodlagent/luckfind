use luckfind::gpu::{self, GpuScanner};

fn main() -> anyhow::Result<()> {
    let ctx = gpu::GpuContext::new_blocking(0)?;
    let secp = secp256k1::Secp256k1::new();
    let two = secp256k1::SecretKey::from_byte_array({ let mut b = [0u8; 32]; b[31] = 2; b }).unwrap();
    let pk_2g = secp256k1::PublicKey::from_secret_key(&secp, &two).serialize();
    let expected = luckfind::btc::hash160(&pk_2g);
    println!("CPU hash160(2G) = {}", hex::encode(&expected));
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 { cand[0][i] = u32::from_le_bytes(expected[4 * i..4 * i + 4].try_into().unwrap()); }
    let mut scanner = GpuScanner::new(ctx, &cand)?;
    scanner.init_random(luckfind::puzzles::puzzle_set())?;
    let g = secp256k1::PublicKey::from_slice(&luckfind::btc::GENERATOR_COMPRESSED).unwrap();
    let ge = g.serialize_uncompressed();
    let mut gx = [0u8; 32]; let mut gy = [0u8; 32];
    gx.copy_from_slice(&ge[1..33]); gy.copy_from_slice(&ge[33..65]);
    let (step_px, step_py) = gpu::convert::stride_step_point(1);
    scanner.set_initial_state(0, [1u32, 0, 0, 0, 0, 0, 0, 0], gpu::GpuState {
        x: gpu::convert::be_bytes_to_limbs(&gx), y: gpu::convert::be_bytes_to_limbs(&gy),
        z: [1, 0, 0, 0, 0, 0, 0, 0], scalar: [1, 0, 0, 0, 0, 0, 0, 0],
        step_px, step_py,
    })?;
    scanner.steps_per_call = 1;
    let matches = scanner.step()?;
    println!("GPU matches: {}", matches.len());
    if let Some(m) = matches.first() {
        // Shader stores ripemd160 output as big-endian u32 words (kangaroo
        // convention).  Reassemble to canonical BE bytes via to_be_bytes.
        let mut h = [0u8; 20];
        for i in 0..5 { h[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_be_bytes()); }
        println!("GPU hash160   = {}", hex::encode(h));
        println!("{}", if h == expected { "✅ MATCH" } else { "❌ mismatch" });
    }
    Ok(())
}
