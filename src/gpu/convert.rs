//! Byte/limb conversion for CPU↔GPU data exchange.
//!
//! GPU uses little-endian `[u32; 8]` limbs; secp256k1 uses big-endian `[u8; 32]`.
//! Adapted from kangaroo's `convert.rs` (stripped of k256 deps — luckfind uses
//! the `secp256k1` crate directly).

/// Convert GPU limbs (LE u32) → big-endian bytes.
pub fn limbs_to_be_bytes(limbs: &[u32; 8]) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for i in 0..8 {
        let be = limbs[7 - i].to_be_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&be);
    }
    bytes
}

/// Convert big-endian bytes → GPU limbs (LE u32).
pub fn be_bytes_to_limbs(bytes: &[u8; 32]) -> [u32; 8] {
    let mut limbs = [0u32; 8];
    for i in 0..8 {
        limbs[i] = u32::from_be_bytes([
            bytes[(7 - i) * 4],
            bytes[(7 - i) * 4 + 1],
            bytes[(7 - i) * 4 + 2],
            bytes[(7 - i) * 4 + 3],
        ]);
    }
    limbs
}

/// Convert 32-byte big-endian scalar → `[u32; 8]` LE limbs.
pub fn scalar_be_to_limbs(bytes: &[u8; 32]) -> [u32; 8] {
    be_bytes_to_limbs(bytes)
}

/// Compute `stride·G` as an affine step point (LE limbs).  The shader does
/// `P += step_point` and `scalar += stride` each step; choosing step_point =
/// stride·G keeps point and scalar in sync (= (scalar)·G at every step).
///
/// `stride = 1` reproduces the lottery step = G.  `stride = N` (NUM_GPU_THREADS)
/// gives puzzle dense-tiling: walkers seeded at `start + i` each stride N keys
/// and partition `[start, start + N·steps)` with zero overlap.
pub fn stride_step_point(stride: u32) -> ([u32; 8], [u32; 8]) {
    assert!(stride > 0 && stride < 2_000_000_000, "stride {stride} out of range");
    let secp = secp256k1::Secp256k1::new();
    // stride·G = the public key whose secret key is `stride`.  Uses the same
    // SecretKey → PublicKey path as init_random (stride < n always holds).
    let scalar_key = {
        let mut b = [0u8; 32];
        b[31] = stride as u8;
        b[30] = (stride >> 8) as u8;
        b[29] = (stride >> 16) as u8;
        b[28] = (stride >> 24) as u8;
        secp256k1::SecretKey::from_byte_array(b).expect("stride < curve order n")
    };
    let stepped = secp256k1::PublicKey::from_secret_key(&secp, &scalar_key);
    let encoded = stepped.serialize_uncompressed();
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&encoded[1..33]);
    y.copy_from_slice(&encoded[33..65]);
    (be_bytes_to_limbs(&x), be_bytes_to_limbs(&y))
}

/// Big-endian scalar add: `a + b` mod 2^256 (plain add with carry, no mod-n
/// reduction — callers guarantee the result < n).  Used to seed walker `i` at
/// `start + i` without a full scalar multiplication.
pub fn scalar_add_be(a: &[u8; 32], b: u64) -> [u8; 32] {
    let mut out = *a;
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

/// Big-endian scalar subtract: `a - b` (plain subtract with borrow, no mod-n
/// reduction — callers guarantee `a >= b`).  `b` is a small u64 (in practice 1),
/// so only the low 8 bytes change unless a borrow chains upward.  Used to seed a
/// reverse scan at `end - 1` without a full scalar multiplication.
pub fn scalar_sub_be_u64(a: &[u8; 32], b: u64) -> [u8; 32] {
    let mut out = *a;
    // Subtract `b` from the low 8-byte (big-endian) window as one u64, then
    // propagate the single-bit borrow (if any) through the upper 24 bytes.
    let low = u64::from_be_bytes(out[24..32].try_into().expect("24..32 is 8 bytes"));
    let (diff, borrowed) = low.overflowing_sub(b);
    out[24..32].copy_from_slice(&diff.to_be_bytes());
    if borrowed {
        for i in (0..24).rev() {
            if out[i] > 0 {
                out[i] -= 1;
                break;
            }
            out[i] = 0xFF;
        }
    }
    out
}

/// Pack a 20-byte hash160 into the GPU candidate buffer format.  The shader's
/// candidate binding is a fixed 78-slot array, so we return a 78-element vec
/// with slot 0 set to `h160` (LE u32 words) and the rest zero.  The caller sets
/// `num_candidates = 1` so only slot 0 is checked.
pub fn hash160_to_candidates(h160: &[u8; 20]) -> Vec<[u32; 5]> {
    let mut cand = vec![[0u32; 5]; 78];
    for i in 0..5 {
        cand[0][i] = u32::from_le_bytes(h160[4 * i..4 * i + 4].try_into().unwrap());
    }
    cand
}

/// Build a full 78-slot candidate buffer from a `PuzzleSet` — every puzzle's
/// hash160 fills its slot (in range order).  The shader's candidate array is
/// fixed at 78 slots; `num_candidates` = set length checks them all.  Used by the GPU
/// lottery worker, which scans against the full embedded puzzle set.
pub fn puzzle_set_to_candidates(ps: &crate::puzzles::PuzzleSet) -> Vec<[u32; 5]> {
    let nranges = ps.ranges().len();
    assert!(nranges <= 78, "more than 78 puzzle ranges cannot fit the GPU candidate buffer");
    let mut cand = vec![[0u32; 5]; 78];
    for (i, range) in ps.ranges().iter().enumerate() {
        for j in 0..5 {
            cand[i][j] =
                u32::from_le_bytes(range.hash160[4 * j..4 * j + 4].try_into().unwrap());
        }
    }
    cand
}

/// Big-endian comparison of two 32-byte keys.  `a < b` lexicographically as
/// unsigned integers.
pub fn be_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in 0..32 {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// Big-endian subtraction `a - b`, returning the difference as a `u64`.
/// Preconditions: `a >= b` and `(a - b) < 2^64`.  Used for the tiny tail of a
/// chunk (remaining < N·steps_per_call ≈ 6.4·10⁶), which always fits.
pub fn scalar_sub_be(a: &[u8; 32], b: &[u8; 32]) -> u64 {
    debug_assert!(!be_lt(a, b), "scalar_sub_be underflow");
    // Borrow propagates LSB→MSB (correct).  The reconstruction must then walk
    // the 8-byte BE window MSB-first — the old code accumulated
    // `diff = diff*256 + d` while still walking LSB-first, which reversed the
    // byte order and inflated small differences (e.g. 0xE0F2 → 0xF2E0…00),
    // turning a chunk tail of ~59k keys into ~2.8·10⁹ steps and hanging the GPU.
    let mut bytes = [0u8; 8];
    let mut borrow: i64 = 0;
    for (k, i) in (24..32).rev().enumerate() {
        let mut d = a[i] as i64 - b[i] as i64 - borrow;
        if d < 0 {
            d += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        bytes[k] = d as u8; // bytes[0] = byte 31 (LSB) … bytes[7] = byte 24 (MSB)
    }
    debug_assert!(borrow == 0, "scalar_sub_be high-byte borrow");
    let mut diff: u64 = 0;
    for &byte in bytes.iter().rev() {
        diff = diff * 256 + byte as u64;
    }
    diff
}

/// Convert a secp256k1 compressed pubkey to a GPU Jacobian state (z=1).
#[allow(dead_code)]
pub fn affine_compressed_to_state(
    compressed: &[u8; 33],
    scalar_be: [u8; 32],
) -> super::GpuState {
    let pk = secp256k1::PublicKey::from_slice(compressed).expect("valid pubkey");
    let encoded = pk.serialize_uncompressed();
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&encoded[1..33]);
    y.copy_from_slice(&encoded[33..65]);
    let (step_px, step_py) = stride_step_point(1); // default step = G (lottery)
    super::GpuState {
        x: be_bytes_to_limbs(&x),
        y: be_bytes_to_limbs(&y),
        z: [1, 0, 0, 0, 0, 0, 0, 0], // Jacobian z=1 (affine)
        scalar: scalar_be_to_limbs(&scalar_be),
        step_px,
        step_py,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limbs_to_be_bytes_roundtrip() {
        let original: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let bytes = limbs_to_be_bytes(&original);
        let recovered = be_bytes_to_limbs(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_limbs_to_be_bytes_value() {
        let mut limbs = [0u32; 8];
        limbs[0] = 0x01020304;
        let bytes = limbs_to_be_bytes(&limbs);
        assert_eq!(bytes[28], 0x01);
        assert_eq!(bytes[29], 0x02);
        assert_eq!(bytes[30], 0x03);
        assert_eq!(bytes[31], 0x04);
    }

    #[test]
    fn test_scalar_be_to_limbs() {
        let mut bytes = [0u8; 32];
        bytes[31] = 0x42;
        let limbs = scalar_be_to_limbs(&bytes);
        assert_eq!(limbs[0] & 0xFF, 0x42);
    }

    /// Regression: scalar_sub_be previously accumulated `diff = diff*256 + d`
    /// while walking the window LSB-first, reversing byte order and inflating
    /// small differences (0xE0F2 → 0xF2E0…00), which turned a ~59k-key chunk
    /// tail into ~2.8·10⁹ GPU steps and appeared to hang the worker.
    fn be32(value: u64) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[24..32].copy_from_slice(&value.to_be_bytes());
        b
    }

    #[test]
    fn test_scalar_sub_be_small_difference() {
        // end = 0x255ebf, start = 0x247dcd → 0xE0F2 = 57586.
        assert_eq!(scalar_sub_be(&be32(0x255EBF), &be32(0x247DCD)), 0xE0F2);
    }

    #[test]
    fn test_scalar_sub_be_zero() {
        assert_eq!(scalar_sub_be(&be32(0x200000), &be32(0x200000)), 0);
    }

    #[test]
    fn test_scalar_sub_be_multi_byte() {
        // 0x0000ABCD12345678 - 0x0000000000000001 = 0xABCD12345677.
        assert_eq!(scalar_sub_be(&be32(0xABCD12345678), &be32(1)), 0xABCD12345677);
    }

    #[test]
    fn test_scalar_add_be_carry() {
        // 0x255EBF + 0x186A0 (100000) = 0x26E55F.
        assert_eq!(&scalar_add_be(&be32(0x255EBF), 100_000)[24..32], &0x26E55Fu64.to_be_bytes());
    }

    #[test]
    fn test_scalar_sub_be_u64_basic() {
        // 0x255EBF - 1 = 0x255EBE (no borrow past the low window).
        assert_eq!(
            &scalar_sub_be_u64(&be32(0x255EBF), 1)[24..32],
            &0x255EBEu64.to_be_bytes()
        );
        // subtract a larger-but-still-low value: 0x255EBF - 100000 = 0x23D81F.
        assert_eq!(
            &scalar_sub_be_u64(&be32(0x255EBF), 100_000)[24..32],
            &0x23D81Fu64.to_be_bytes()
        );
        // b = 0 is a no-op.
        assert_eq!(scalar_sub_be_u64(&be32(0x1234), 0), be32(0x1234));
    }

    #[test]
    fn test_scalar_sub_be_u64_borrow_propagates() {
        // 0x1000 - 1 = 0x0FFF: borrow crosses the byte boundary inside the low
        // window; high bytes untouched.
        assert_eq!(&scalar_sub_be_u64(&be32(0x1000), 1)[24..32], &0x0FFFu64.to_be_bytes());

        // 2^64 - 1 in the low window (be32(0) with high=0xFFFFFFFF...) — feed a
        // value whose low window is exactly 0, so subtracting 1 borrows into the
        // upper 24 bytes: 0x01000000..0000 - 1 = 0x00FFFFFFFFFFFFFFFF.
        let a = {
            let mut b = [0u8; 32];
            b[23] = 0x01; // 2^64, byte 23 is the top of the low window
            b
        };
        let sub = scalar_sub_be_u64(&a, 1);
        assert_eq!(sub[23], 0x00);
        assert_eq!(&sub[24..32], &[0xFFu8; 8]);
    }

    #[test]
    fn test_scalar_sub_be_u64_matches_biguint() {
        // Property check against BigUint arithmetic for a few (a, b) with a >= b.
        use num_bigint::BigUint;
        for (hi, b) in [(0u8, 1u64), (0x40, 1), (0x80, 0x1234), (0xFF, 999_999)] {
            let mut a = [0u8; 32];
            a[23] = hi; // vary the high byte so subtraction crosses windows
            a[24] = 0x11;
            a[31] = 0x55;
            let expect = BigUint::from_bytes_be(&a) - BigUint::from(b);
            assert_eq!(
                scalar_sub_be_u64(&a, b),
                {
                    let mut out = [0u8; 32];
                    let be = expect.to_bytes_be();
                    out[32 - be.len()..].copy_from_slice(&be);
                    out
                },
                "a={a:?} b={b}"
            );
        }
        // Degenerate: a == b → 0.
        assert_eq!(scalar_sub_be_u64(&be32(0xABC), 0xABC), [0u8; 32]);
    }
}
