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
/// fixed at 78 slots; `num_candidates = 78` checks them all.  Used by the GPU
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
    let mut borrow: i64 = 0;
    let mut diff: u64 = 0;
    // High bytes (0..24) are equal (diff < 2^64), so borrow stays within the low
    // 8 bytes — propagate it through those and accumulate.
    for i in (24..32).rev() {
        let mut d = a[i] as i64 - b[i] as i64 - borrow;
        if d < 0 {
            d += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        diff = diff * 256 + d as u64;
    }
    debug_assert!(borrow == 0, "scalar_sub_be high-byte borrow");
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
}
