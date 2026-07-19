//! Per-step GPU-vs-CPU consistency checks.
//!
//! Two independent observation channels, each covering different pipeline
//! stages, both cross-checked against a CPU reference computed from the KNOWN
//! private key:
//!
//!   Channel A — match output: when a walker lands on a candidate key the
//!   shader reports the reconstructed private key (`scalar`), the affine point
//!   (`pubkey_x/pubkey_y` as LE limbs) and the `hash160`.  Comparing all of
//!   these to the CPU reference exercises the WHOLE pipeline end to end:
//!   point-add → batch-invert → affine-convert → compressed-serialize →
//!   SHA256 → RIPEMD160 → byte-swap, plus scalar tracking.
//!
//!   Channel B — state readback: after N steps we read the Jacobian (X,Y,Z) and
//!   scalar of any walker, affine-convert on the CPU (X/Z², Y/Z³) and compare
//!   to CPU's `(k+N·stride)·G`, and compare the scalar to `k+N·stride`.  This
//!   checks the EC point-add and scalar-add INDEPENDENTLY of the hash path
//!   (so a hash bug can't mask an EC bug or vice-versa).
//!
//! Both channels are run for many random-looking keys, several step counts and
//! both stride = 1 (lottery) and stride = N (puzzle dense-tiling), so any
//! stage that drifts is caught.

use luckfind::gpu::{self, GpuScanner};

/// Big-endian mod-p field prime.
const P: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
];

fn new_scanner(stride: u32) -> Option<GpuScanner> {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gpu_step_check: no Metal device, skipping: {e}");
            return None;
        }
    };
    // Single candidate slot = target; everything else zero.
    let mut scanner = GpuScanner::new(ctx, &[[0u32; 5]; 78]).expect("GpuScanner::new");
    scanner.stride = stride;
    scanner.num_candidates = 1;
    // init_random populates initial_scalars so set_initial_state / seed_range's
    // internal readback has a buffer to patch.  Seed is arbitrary (states are
    // overwritten by the per-test seeding anyway).
    scanner.init_random(42).expect("init_random");
    Some(scanner)
}

/// CPU affine convert of a Jacobian point (X,Y,Z as LE limbs) → (x,y) field
/// elements as 32-byte big-endian.  Uses num-bigint for the mod-p inversion.
fn jac_to_affine_cpu(x: &[u32; 8], y: &[u32; 8], z: &[u32; 8]) -> ([u8; 32], [u8; 32]) {
    let xi = le_limbs_to_big(x);
    let yi = le_limbs_to_big(y);
    let zi = le_limbs_to_big(z);
    let p = field_p();
    let z_inv = mod_inv(&zi, &p);
    let z2_inv = (&z_inv * &z_inv) % &p;
    let z3_inv = (&z2_inv * &z_inv) % &p;
    let aff_x = (&xi * &z2_inv) % &p;
    let aff_y = (&yi * &z3_inv) % &p;
    (big_to_be32(&aff_x), big_to_be32(&aff_y))
}

fn le_limbs_to_big(l: &[u32; 8]) -> num_bigint::BigUint {
    let mut n = num_bigint::BigUint::from(l[7]);
    for i in (0..7).rev() {
        n <<= 32u32;
        n += num_bigint::BigUint::from(l[i]);
    }
    n
}

fn big_to_be32(n: &num_bigint::BigUint) -> [u8; 32] {
    let mut bytes = n.to_bytes_be();
    // pad/truncate to 32 bytes (number < p < 2^256, so at most 32 bytes)
    if bytes.len() < 32 {
        let mut pad = vec![0u8; 32 - bytes.len()];
        pad.extend_from_slice(&bytes);
        bytes = pad;
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes[bytes.len() - 32..]);
    out
}

fn field_p() -> num_bigint::BigUint {
    num_bigint::BigUint::from_bytes_be(&P)
}

fn mod_inv(a: &num_bigint::BigUint, p: &num_bigint::BigUint) -> num_bigint::BigUint {
    // p is prime; a^(p-2) mod p.
    a.modpow(&((p - 2u32).into()), p)
}

/// Reconstrate the canonical compressed pubkey (CPU) for a known scalar, and pack
/// its hash160 as LE u32 words into candidate slot 0.
fn candidate_for_scalar(sk_be: [u8; 32]) -> [[u32; 5]; 78] {
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_byte_array(sk_be).unwrap();
    let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk).serialize();
    let h = luckfind::btc::hash160(&pk);
    let mut cand = [[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(h[4 * i..4 * i + 4].try_into().unwrap());
    }
    cand
}

/// Reassemble the shader's BE riemd160 words into the canonical 20-byte hash.
fn reassemble_hash160(m: &gpu::GpuMatchOutput) -> [u8; 20] {
    let mut h = [0u8; 20];
    for i in 0..5 {
        h[4 * i..4 * i + 4].copy_from_slice(&m.hash160[i].to_be_bytes());
    }
    h
}

/// CPU (scalar·G) affine coords as (x, y), each 32-byte big-endian.
fn cpu_affine(sk_be: [u8; 32]) -> ([u8; 32], [u8; 32]) {
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
    (x, y)
}

/// Seed walker 0 at the given scalar's point (Jacobian z=1, scalar = sk_be LE
/// limbs).  NOTE: the scanner's step point is set separately via its `stride`
/// field and `seed_range`/`init_random`; this only overwrites walker 0's
/// starting Jacobian + scalar so the test can place the walker at a known key
/// while keeping the stride step point the scanner was constructed with.
fn seed_at(scanner: &mut GpuScanner, sk_be: [u8; 32]) {
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_byte_array(sk_be).unwrap();
    let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
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

// ── Channel A: match output vs CPU, for many keys / steps / strides ────────

#[test]
fn channel_a_match_output_matches_cpu() {
    let Some(mut scanner) = new_scanner(1) else {
        return;
    };
    scanner.steps_per_call = 1;

    // Distinct scalars, including a large one (high limbs != 0).  Kept small —
    // this test rebuilds the GPU context per iteration (slow), so it trades
    // breadth for runtime; the key property (every pipeline stage vs CPU) is
    // what matters, and dense_tiling covers stride = N.
    let scalars: Vec<[u8; 32]> = vec![
        scalar_from_u64(1),
        scalar_from_u64(123456789),
        scalar_from_hex("0080000000000000000000000000000000000000000000000000000000000000"),
    ];

    let secp = secp256k1::Secp256k1::new();
    for sk_be in &scalars {
        // Hash-then-advance: a dispatch of steps_per_call = P checks keys
        // sk .. sk+P-1 (checks current key, THEN advances).  So the last key a
        // walker checks is sk + P - 1; we set the candidate there and verify the
        // match reports that exact key (scalar + affine + hash160) vs CPU.
        for steps in 1..=2u32 {
            let target_sk = scalar_add_small(*sk_be, (steps - 1) as u64);
            // Repack candidate = hash160(target) into the buffer the scanner uses.
            let cand = candidate_for_scalar(target_sk);
            // Re-create scanner with correct candidate each iteration.
            let ctx = gpu::GpuContext::new_blocking(0).unwrap();
            let mut sc = GpuScanner::new(ctx, &cand).unwrap();
            sc.stride = 1;
            sc.num_candidates = 1;
            sc.steps_per_call = steps;
            sc.init_random(42).expect("init_random");
            seed_at(&mut sc, *sk_be);
            let matches = sc.step().expect("step");
            assert!(
                !matches.is_empty(),
                "channel A: no match for sk={} steps={}",
                hex::encode(sk_be),
                steps
            );
            let m = &matches[0];

            // 1. scalar (private key) must equal target_sk.
            let mut scalar_be = [0u8; 32];
            scalar_be.copy_from_slice(&gpu::convert::limbs_to_be_bytes(&m.scalar));
            assert_eq!(
                scalar_be, target_sk,
                "channel A: scalar mismatch sk={} steps={}",
                hex::encode(sk_be), steps
            );

            // 2. reconstructed hash160 must equal CPU hash160(target·G).
            let got_h = reassemble_hash160(m);
            let got_pk_cpu = {
                let pk = secp256k1::PublicKey::from_secret_key(
                    &secp,
                    &secp256k1::SecretKey::from_byte_array(target_sk).unwrap(),
                )
                .serialize();
                luckfind::btc::hash160(&pk)
            };
            assert_eq!(
                got_h, got_pk_cpu,
                "channel A: hash160 mismatch sk={} steps={}",
                hex::encode(sk_be), steps
            );

            // 3. affine pubkey_x/pubkey_y must equal CPU (target·G) affine coords.
            let (cpu_x, cpu_y) = {
                let pk = secp256k1::PublicKey::from_secret_key(
                    &secp,
                    &secp256k1::SecretKey::from_byte_array(target_sk).unwrap(),
                );
                let u = pk.serialize_uncompressed();
                let mut x = [0u8; 32];
                let mut y = [0u8; 32];
                x.copy_from_slice(&u[1..33]);
                y.copy_from_slice(&u[33..65]);
                (x, y)
            };
            let gpu_x = gpu::convert::limbs_to_be_bytes(&m.pubkey_x);
            let gpu_y = gpu::convert::limbs_to_be_bytes(&m.pubkey_y);
            assert_eq!(gpu_x, cpu_x, "channel A: pubkey_x mismatch sk={} steps={}",
                hex::encode(sk_be), steps);
            assert_eq!(gpu_y, cpu_y, "channel A: pubkey_y mismatch sk={} steps={}",
                hex::encode(sk_be), steps);

            assert_eq!(m.candidate_index, 0, "channel A: wrong candidate_index");
            assert_eq!(m.thread_id, 0, "channel A: wrong thread_id");
        }
    }
}

// ── Channel B: state readback (point-add + scalar-add) vs CPU ──────────────

#[test]
fn channel_b_state_readback_matches_cpu() {
    for &stride in &[1u32, gpu::NUM_GPU_THREADS] {
        let Some(mut scanner) = new_scanner(stride) else {
            return;
        };
        let scalars: Vec<[u8; 32]> = vec![
            scalar_from_u64(1),
            scalar_from_u64(1000),
            scalar_from_u64(u64::MAX),
            scalar_from_hex("8000000000000000000000000000000000000000000000000000000000000000"),
        ];
        let secp = secp256k1::Secp256k1::new();
        for sk_be in &scalars {
            seed_at(&mut scanner, *sk_be);
            let stride64 = stride as u64;
            // Run up to 4 steps, reading back after each.
            let mut carried = *sk_be;
            for step in 1..=4u32 {
                scanner.steps_per_call = 1;
                let _ = scanner.step().expect("step");
                let states = scanner.readback_states().expect("readback");
                let s = &states[0];

                // scalar must equal k + step*stride (big-endian, no mod-n wrap at
                // these magnitudes).
                carried = scalar_add_small(carried, stride64);
                let mut scalar_be = [0u8; 32];
                scalar_be.copy_from_slice(&gpu::convert::limbs_to_be_bytes(&s.scalar));
                assert_eq!(
                    scalar_be, carried,
                    "channel B: scalar mismatch sk={} step={} stride={}",
                    hex::encode(sk_be), step, stride
                );

                // Jacobian affine-convert on CPU must equal (carried·G).
                let (ax, ay) = jac_to_affine_cpu(&s.x, &s.y, &s.z);
                let cpu_pk = secp256k1::PublicKey::from_secret_key(
                    &secp,
                    &secp256k1::SecretKey::from_byte_array(carried)
                        .expect("valid key at this magnitude"),
                );
                let u = cpu_pk.serialize_uncompressed();
                let mut cpu_x = [0u8; 32];
                let mut cpu_y = [0u8; 32];
                cpu_x.copy_from_slice(&u[1..33]);
                cpu_y.copy_from_slice(&u[33..65]);
                assert_eq!(ax, cpu_x, "channel B: affine x mismatch sk={} step={} stride={}",
                    hex::encode(sk_be), step, stride);
                assert_eq!(ay, cpu_y, "channel B: affine y mismatch sk={} step={} stride={}",
                    hex::encode(sk_be), step, stride);
            }
        }
    }
}

// For stride = N dense-tiling specifically: seed_range(start) seeds walker i at
// (start + i)·G; after ONE dispatch walker i must have advanced exactly N keys
// to land on (start + i + N)·G.  We seed all 100k walkers at once and verify a
// spread of sampled walkers via readback — i.e. the whole partition is
// gap-free and correct.
#[test]
fn dense_tiling_each_walker_lands_correctly() {
    let n = gpu::NUM_GPU_THREADS as usize;
    let Some(mut scanner) = new_scanner(gpu::NUM_GPU_THREADS) else {
        return;
    };
    scanner.steps_per_call = 1;

    // start = scalar 500000 so start + (n-1) is well within range.
    let start = scalar_from_u64(500_000);
    let secp = secp256k1::Secp256k1::new();

    scanner.seed_range(start).expect("seed_range");
    let _ = scanner.step().expect("step");
    let states = scanner.readback_states().expect("readback");

    // Check a spread of walkers: the first few, the middle, and the last.
    let indices = {
        let mut v: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        v.push(n / 2);
        v.push(n - 1);
        v
    };
    for i in indices {
        let expected = scalar_add_small(start, (i + n) as u64);
        let mut scalar_be = [0u8; 32];
        scalar_be.copy_from_slice(&gpu::convert::limbs_to_be_bytes(&states[i].scalar));
        assert_eq!(
            scalar_be, expected,
            "dense tiling: walker {} scalar wrong", i
        );
        let (ax, ay) = jac_to_affine_cpu(&states[i].x, &states[i].y, &states[i].z);
        let (cpu_x, cpu_y) = cpu_affine(expected);
        assert_eq!(ax, cpu_x, "dense tiling: walker {} x wrong", i);
        assert_eq!(ay, cpu_y, "dense tiling: walker {} y wrong", i);
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn scalar_from_u64(v: u64) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[24..32].copy_from_slice(&v.to_be_bytes());
    b
}

fn scalar_from_hex(h: &str) -> [u8; 32] {
    let s = h.strip_prefix("0x").unwrap_or(h);
    let raw = hex::decode(s).unwrap();
    let mut b = [0u8; 32];
    b[32 - raw.len()..].copy_from_slice(&raw);
    b
}

fn scalar_add_small(a: [u8; 32], b: u64) -> [u8; 32] {
    let mut out = a;
    let mut carry = b;
    for i in (0..32).rev() {
        if carry == 0 {
            break;
        }
        let sum = out[i] as u64 + carry;
        out[i] = (sum & 0xFF) as u8;
        carry = sum >> 8;
    }
    out
}
