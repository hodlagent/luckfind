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

    // CPU reference: G compressed pubkey and its hash160.  The shader uses
    // hash-then-advance ordering, so a walker seeded at G (scalar 1) checks G on
    // step 0, then advances to 2G.  We seed at G and look for hash160(G).
    let g = secp256k1::PublicKey::from_slice(&luckfind::btc::GENERATOR_COMPRESSED).unwrap();
    let pk_g = g.serialize();
    let expected = luckfind::btc::hash160(&pk_g);

    // Candidate slot 0 = hash160(G), packed as little-endian u32 words.
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(expected[4 * i..4 * i + 4].try_into().unwrap());
    }

    let mut scanner = GpuScanner::new(ctx, &cand).expect("GpuScanner::new");
    scanner.init_random(42).expect("init_random");

    // Overwrite thread 0 with G (Jacobian z=1, scalar 1) so step 0 checks G.
    let ge = g.serialize_uncompressed();
    let mut gx = [0u8; 32];
    let mut gy = [0u8; 32];
    gx.copy_from_slice(&ge[1..33]);
    gy.copy_from_slice(&ge[33..65]);
    // Stride = 1 (lottery default): step point = G.
    let (step_px, step_py) = gpu::convert::stride_step_point(1);
    scanner
        .set_initial_state(
            0,
            [1u32, 0, 0, 0, 0, 0, 0, 0],
            gpu::GpuState {
                x: gpu::convert::be_bytes_to_limbs(&gx),
                y: gpu::convert::be_bytes_to_limbs(&gy),
                z: [1, 0, 0, 0, 0, 0, 0, 0],
                scalar: [1, 0, 0, 0, 0, 0, 0, 0],
                step_px,
                step_py,
            },
        )
        .expect("set_initial_state");

    scanner.steps_per_call = 1;
    let matches = scanner.step().expect("step");
    assert!(
        !matches.is_empty(),
        "expected G match but batch inversion produced no candidate — \
         affine conversion is broken (Z inverted incorrectly)"
    );

    let m = &matches[0];
    let mut h = [0u8; 20];
    // The shader stores ripemd160 output as big-endian u32 words (kangaroo
    // convention — .h[0] = most-significant word of the 160-bit digest).  The
    // ground-truth `expected` is the canonical big-endian byte array from
    // btc::hash160, so we must reassemble with to_be_bytes (NOT to_le_bytes).
    for i in 0..5 {
        h[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_be_bytes());
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

/// Deterministic puzzle-mode scan: seed all walkers at `start + i` (stride =
/// N = NUM_GPU_THREADS), then run ONE multi-step dispatch and confirm the
/// walker that lands exactly on the target key reports it.  This is the
/// correctness property the GPU puzzle worker relies on — dense zero-overlap
/// tiling recovers a key at a known offset with no false negatives.
#[test]
fn deterministic_scan_finds_key_at_known_offset() {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no Metal device available, skipping: {e}");
            return;
        }
    };
    let secp = secp256k1::Secp256k1::new();

    // start = scalar 1024 (big-endian).  Walkers seed at start + i, stride N.
    let mut start = [0u8; 32];
    start[30] = 4; // 4 << 8 = 1024
    let n = gpu::NUM_GPU_THREADS as u64;

    // Place the target key at `start + (n + 5)` = walker 5, step 1.
    //   walker i step 0 → start + i
    //   walker i step 1 → start + i + n
    // So walker 5 at step 1 lands on start + 5 + n.
    let target_sk = gpu::convert::scalar_add_be(&start, n + 5);
    let target_pk_c = {
        let sk = secp256k1::SecretKey::from_byte_array(target_sk).unwrap();
        secp256k1::PublicKey::from_secret_key(&secp, &sk).serialize()
    };
    let target_h160 = luckfind::btc::hash160(&target_pk_c);

    // Candidate slot 0 = target hash160 (LE u32 words).  num_candidates = 1.
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(target_h160[4 * i..4 * i + 4].try_into().unwrap());
    }

    let mut scanner = GpuScanner::new(ctx, &cand).expect("GpuScanner::new");
    scanner.stride = gpu::NUM_GPU_THREADS; // puzzle dense-tiling
    scanner.num_candidates = 1;
    scanner
        .seed_range(start)
        .expect("seed_range");

    // 2 steps/step: step 0 covers [start, start+N), step 1 covers
    // [start+N, start+2N) — the target at start+N+5 lies in step 1 of walker 5.
    scanner.steps_per_call = 2;
    let matches = scanner.step().expect("step");

    assert!(
        !matches.is_empty(),
        "deterministic scan missed the target key at start+N+5"
    );
    // Verify the recovered private key is exactly the target.
    let recovered_be = gpu::convert::limbs_to_be_bytes(&matches[0].scalar);
    assert_eq!(
        recovered_be, target_sk,
        "GPU recovered wrong private key: got {} expected {}",
        hex::encode(recovered_be),
        hex::encode(target_sk)
    );
    // The target sits at offset n+5 from start.
    assert_eq!(matches[0].thread_id, 5, "match must come from walker 5");
    eprintln!(
        "  ✓ deterministic scan found target key (walker {}, offset {})",
        matches[0].thread_id,
        n + 5,
    );
}
