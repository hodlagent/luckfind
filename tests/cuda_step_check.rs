//! CUDA per-step GPU-vs-CPU consistency checks — mirror of `gpu_step_check.rs`
//! for the CUDA backend (same two observation channels, same CPU reference).
//!
//!   Channel A — match output: when a walker lands on a candidate key the
//!   kernel reports the reconstructed private key (`scalar`), the affine point
//!   (`pubkey_x/pubkey_y` as LE limbs) and the `hash160`.  Comparing all of
//!   these to the CPU reference exercises the WHOLE pipeline end to end:
//!   point-add → batch-invert → affine-convert → compressed-serialize →
//!   SHA256 → RIPEMD160 → byte-swap, plus scalar tracking.
//!
//!   Channel B — state readback: after N steps we read the Jacobian (X,Y,Z) and
//!   scalar of any walker, affine-convert on the CPU (X/Z², Y/Z³) and compare
//!   to CPU's `(k+N·stride)·G`, and compare the scalar to `k+N·stride`.
//!
//! The whole file is compiled out unless the `cuda` feature is on, and every
//! test skips (returns early) when no CUDA device is present.
//!
//! NOTE: CudaScanner has no `set_initial_state` (unlike GpuScanner), so tests
//! place walkers via `seed_range` — walker 0 is seeded at `start`.  With
//! stride = 1 that ALSO seeds walker 1 at `start + 1`, so Channel A matches
//! can come from multiple walkers: candidates are found by filtering on
//! `thread_id == 0` rather than assuming matches[0] ordering.

#![cfg(feature = "cuda")]

use luckfind::cuda::CudaScanner;
use luckfind::gpu::{self, GpuMatchOutput};

/// Big-endian mod-p field prime.
const P: [u8; 32] = [
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2F,
];

fn new_scanner(stride: u32) -> Option<CudaScanner> {
    if !CudaScanner::probe() {
        eprintln!("cuda_step_check: no CUDA device, skipping");
        return None;
    }
    // Single candidate slot = target; everything else zero.
    let mut scanner = CudaScanner::new(&[[0u32; 5]; 78]).expect("CudaScanner::new");
    scanner.stride = stride;
    scanner.num_candidates = 1;
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

/// Pack the hash160 of a known scalar's compressed pubkey into candidate slot 0.
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

/// Reassemble the kernel's BE ripemd160 words into the canonical 20-byte hash.
fn reassemble_hash160(m: &GpuMatchOutput) -> [u8; 20] {
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

// ── Channel A: match output vs CPU, for many keys / steps ───────────────────

#[test]
fn channel_a_match_output_matches_cpu() {
    if !CudaScanner::probe() {
        eprintln!("cuda_step_check: no CUDA device, skipping");
        return;
    }

    // Distinct scalars, including a large one (high limbs != 0).
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
            let cand = candidate_for_scalar(target_sk);
            let mut sc = CudaScanner::new(&cand).unwrap();
            sc.stride = 1;
            sc.num_candidates = 1;
            sc.steps_per_call = steps;
            sc.seed_range(*sk_be).expect("seed_range");
            let matches = sc.step().expect("step");
            let m = matches
                .iter()
                .find(|m| m.thread_id == 0)
                .unwrap_or_else(|| {
                    panic!(
                        "channel A: no walker-0 match for sk={} steps={} ({} matches)",
                        hex::encode(sk_be),
                        steps,
                        matches.len()
                    )
                });

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
            let (cpu_x, cpu_y) = cpu_affine(target_sk);
            let gpu_x = gpu::convert::limbs_to_be_bytes(&m.pubkey_x);
            let gpu_y = gpu::convert::limbs_to_be_bytes(&m.pubkey_y);
            assert_eq!(gpu_x, cpu_x, "channel A: pubkey_x mismatch sk={} steps={}",
                hex::encode(sk_be), steps);
            assert_eq!(gpu_y, cpu_y, "channel A: pubkey_y mismatch sk={} steps={}",
                hex::encode(sk_be), steps);

            assert_eq!(m.candidate_index, 0, "channel A: wrong candidate_index");
        }
    }
}

// CUDA-only derivation test: the candidate hash160 comes from an EXTERNAL
// known vector (not computed on the CPU with secp256k1) — the kernel must
// reproduce the compressed pubkey and the pre-address RIPEMD-160 hash on its
// own.  Walker 0 sits exactly on the private key (stride=1, first check), so
// everything asserted here — affine convert, 02/03 prefix, SHA256, RIPEMD160,
// scalar tracking — is GPU work.
#[test]
fn cuda_only_derives_known_hash160_vector() {
    if !CudaScanner::probe() {
        eprintln!("cuda_step_check: no CUDA device, skipping");
        return;
    }
    let privkey_hex = "2259DA7B4E734DE2CB41CF5D4644F0F6CBBDD98157B3F9F786B9F40356CEB8CD";
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&hex::decode(privkey_hex).unwrap());

    // External vector: hash160 of the compressed pubkey of the above key.
    let want_h160: [u8; 20] = hex::decode("cdb5b024d031c56acfe6f00ae9a4f11f098ea085")
        .unwrap()
        .try_into()
        .unwrap();
    let mut cand = [[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(want_h160[4 * i..4 * i + 4].try_into().unwrap());
    }

    let mut sc = CudaScanner::new(&cand).unwrap();
    sc.stride = 1;
    sc.num_candidates = 1;
    sc.steps_per_call = 1;
    sc.seed_range(seed).expect("seed_range");
    let matches = sc.step().expect("step");
    let m = matches
        .iter()
        .find(|m| m.thread_id == 0)
        .unwrap_or_else(|| {
            panic!(
                "cuda-only derive: walker 0 must hit its own key ({} matches)",
                matches.len()
            )
        });

    // scalar must be the private key itself.
    let scalar_be = gpu::convert::limbs_to_be_bytes(&m.scalar);
    assert_eq!(
        hex::encode(scalar_be),
        privkey_hex.to_ascii_lowercase(),
        "cuda-only derive: scalar mismatch"
    );

    // hash160 must equal the external vector.
    assert_eq!(
        reassemble_hash160(m),
        want_h160,
        "cuda-only derive: hash160 mismatch"
    );

    // compressed pubkey = 02/03 + x, prefix from y parity — both GPU-derived.
    let x_be = gpu::convert::limbs_to_be_bytes(&m.pubkey_x);
    let y_be = gpu::convert::limbs_to_be_bytes(&m.pubkey_y);
    let mut pubkey = [0u8; 33];
    pubkey[0] = if y_be[31] & 1 == 1 { 0x03 } else { 0x02 };
    pubkey[1..].copy_from_slice(&x_be);
    assert_eq!(
        hex::encode(pubkey),
        "031a753c212489bcc6fbf96d0228ae6b1b6391442d31a924cead37309640999dc8",
        "cuda-only derive: compressed pubkey mismatch"
    );
}

// ── Channel B: state readback (point-add + scalar-add) vs CPU ──────────────

#[test]
fn channel_b_state_readback_matches_cpu() {
    for &stride in &[1u32, luckfind::cuda::NUM_GPU_THREADS] {
        let Some(mut scanner) = new_scanner(stride) else {
            return;
        };
        let scalars: Vec<[u8; 32]> = vec![
            scalar_from_u64(1),
            scalar_from_u64(1000),
            scalar_from_u64(u64::MAX),
            scalar_from_hex("8000000000000000000000000000000000000000000000000000000000000000"),
        ];
        for sk_be in &scalars {
            scanner.seed_range(*sk_be).expect("seed_range");
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
                let (cpu_x, cpu_y) = cpu_affine(carried);
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
    let n = luckfind::cuda::NUM_GPU_THREADS as usize;
    let Some(mut scanner) = new_scanner(luckfind::cuda::NUM_GPU_THREADS) else {
        return;
    };
    scanner.steps_per_call = 1;

    // start = scalar 500000 so start + (n-1) is well within range.
    let start = scalar_from_u64(500_000);

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

// Dense-tiling HIT path: with stride = N, seed_range(start) places walker i at
// (start + i)·G; the FIRST dispatch checks exactly [start, start+N) — a walker
// landing on the candidate in that first check must produce a match.  This
// exercises the kernel's stride-N check-then-advance path (channel A only
// covers stride = 1), which the puzzle worker depends on.
#[test]
fn dense_tiling_first_check_hits_candidate() {
    let n = luckfind::cuda::NUM_GPU_THREADS as usize;
    let Some(mut scanner) = new_scanner(luckfind::cuda::NUM_GPU_THREADS) else {
        return;
    };
    scanner.steps_per_call = 1;

    let start = scalar_from_u64(0x10afd);
    // A key inside [start, start+n): walker 6216.
    let target = scalar_from_u64(0x12345);
    let cand = candidate_for_scalar(target);
    // new_scanner built with zero candidates — rebuild with the real one.
    let mut sc = CudaScanner::new(&cand).unwrap();
    sc.stride = luckfind::cuda::NUM_GPU_THREADS;
    sc.num_candidates = 1;
    sc.steps_per_call = 1;
    sc.seed_range(start).expect("seed_range");
    let matches = sc.step().expect("step");
    let m = matches
        .iter()
        .find(|m| m.thread_id == 6216)
        .unwrap_or_else(|| {
            panic!(
                "dense hit: no match from walker 6216 ({} matches: {:?})",
                matches.len(),
                matches
                    .iter()
                    .map(|m| (m.thread_id, hex::encode(gpu::convert::limbs_to_be_bytes(&m.scalar))))
                    .collect::<Vec<_>>()
            )
        });
    assert_eq!(
        gpu::convert::limbs_to_be_bytes(&m.scalar),
        target,
        "dense hit: wrong scalar reported"
    );
    let _ = n; // walker index 6216 is hard-coded above
}

// Dense-tiling HIT with a REAL 256-bit key (high limbs non-zero) instead of a
// small test key: candidate hash160 = 2259DA7B...CEB8CD's compressed-pubkey
// hash160; start = key - 6216 so walker 6216 checks it on the FIRST dispatch.
#[test]
fn dense_tiling_hits_real_256bit_key() {
    let Some(_) = new_scanner(luckfind::cuda::NUM_GPU_THREADS) else {
        return;
    };
    let target = scalar_from_hex(
        "2259da7b4e734de2cb41cf5d4644f0f6cbbdd98157b3f9f786b9f40356ceb8cd",
    );
    let cand = candidate_for_scalar(target);
    // start = target - 6216 (walker 6216 lands exactly on the key first check).
    let start = scalar_sub_small(target, 6216);

    let mut sc = CudaScanner::new(&cand).unwrap();
    sc.stride = luckfind::cuda::NUM_GPU_THREADS;
    sc.num_candidates = 1;
    sc.steps_per_call = 1;
    sc.seed_range(start).expect("seed_range");
    let matches = sc.step().expect("step");
    let m = matches
        .iter()
        .find(|m| m.thread_id == 6216)
        .unwrap_or_else(|| {
            panic!(
                "real-256bit hit: no match from walker 6216 ({} matches)",
                matches.len()
            )
        });
    assert_eq!(
        gpu::convert::limbs_to_be_bytes(&m.scalar),
        target,
        "real-256bit hit: wrong scalar reported"
    );
    let got_h = reassemble_hash160(m);
    let mut want_h = [0u8; 20];
    want_h.copy_from_slice(&hex::decode("cdb5b024d031c56acfe6f00ae9a4f11f098ea085").unwrap());
    assert_eq!(got_h, want_h, "real-256bit hit: wrong hash160 reported");
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

fn scalar_sub_small(a: [u8; 32], b: u64) -> [u8; 32] {
    let mut out = a;
    let mut borrow = b;
    for i in (0..32).rev() {
        if borrow == 0 {
            break;
        }
        let sub = (borrow & 0xFF) as u8;
        let (d, under) = out[i].overflowing_sub(sub);
        out[i] = d;
        borrow = (borrow >> 8) + u64::from(under);
    }
    out
}
