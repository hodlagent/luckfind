//! GPU regression test for Montgomery batch inversion.
//!
//! Strategy: set thread 0's starting point to G (scalar 1), configure the first
//! candidate to be `hash160(2G)` (the point after one step), and check that the
//! GPU scan actually finds it.  The only way `pubkey_x/pubkey_y` can hash to
//! `2G` is if the affine conversion is correct — and that requires the shared
//! `batch_inv_128()` Z-inversion to round-trip every lane's Jacobian Z against
//! the CPU's `fe_inv`.  A wrong batch-inverted Z silently corrupts every affine
//! hash, and we wouldn't find the candidate.

use luckfind::gpu::{self, GpuScanner};

fn scan_one_step_finds_2g() -> bool {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gpu_batch_inv: no Metal device available, skipping: {e}");
            return true; // skip, don't fail the suite on CI boxes without a GPU
        }
    };

    let secp = secp256k1::Secp256k1::new();

    // CPU reference: 2G compressed pubkey and its hash160.
    let two = secp256k1::SecretKey::from_byte_array({
        let mut b = [0u8; 32];
        b[31] = 2;
        b
    })
    .unwrap();
    let pk_2g = secp256k1::PublicKey::from_secret_key(&secp, &two).serialize();
    let expected = luckfind::btc::hash160(&pk_2g);

    // Candidate slot 0 = hash160(2G), packed as little-endian u32 words.
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(expected[4 * i..4 * i + 4].try_into().unwrap());
    }

    let mut scanner = GpuScanner::new(ctx, &cand).expect("GpuScanner::new");
    scanner.init_random(42).expect("init_random");

    // Overwrite thread 0 with G (Jacobian z=1, scalar 1) so one step walks to 2G.
    let g = secp256k1::PublicKey::from_slice(&luckfind::btc::GENERATOR_COMPRESSED).unwrap();
    let ge = g.serialize_uncompressed();
    let mut gx = [0u8; 32];
    let mut gy = [0u8; 32];
    gx.copy_from_slice(&ge[1..33]);
    gy.copy_from_slice(&ge[33..65]);
    scanner
        .set_initial_state(
            0,
            [1u32, 0, 0, 0, 0, 0, 0, 0],
            gpu::GpuState {
                x: gpu::convert::be_bytes_to_limbs(&gx),
                y: gpu::convert::be_bytes_to_limbs(&gy),
                z: [1, 0, 0, 0, 0, 0, 0, 0],
                scalar: [1, 0, 0, 0, 0, 0, 0, 0],
            },
        )
        .expect("set_initial_state");

    scanner.steps_per_call = 1;
    let matches = scanner.step().expect("step");
    assert!(
        !matches.is_empty(),
        "expected 2G match but batch inversion produced no candidate — \
         affine conversion is broken (Z inverted incorrectly)"
    );

    let m = &matches[0];
    let mut h = [0u8; 20];
    for i in 0..5 {
        h[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_le_bytes());
    }
    assert_eq!(
        h, expected,
        "hash160 mismatch: GPU={} CPU={}",
        hex::encode(h),
        hex::encode(expected)
    );

    // candidate_index 0 is the only slot we filled; anything else means the
    // scan is reporting noise (which a corrupted affine pipeline would do).
    assert_eq!(m.candidate_index, 0, "unexpected candidate_index");
    assert_eq!(m.thread_id, 0, "match must come from thread 0");

    true
}

#[test]
fn batch_inv_produces_correct_affine_point() {
    assert!(scan_one_step_finds_2g());
}
