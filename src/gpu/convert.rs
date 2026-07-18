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

/// Convert a secp256k1 compressed pubkey to a GPU Jacobian state (z=1).
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
    super::GpuState {
        x: be_bytes_to_limbs(&x),
        y: be_bytes_to_limbs(&y),
        z: [1, 0, 0, 0, 0, 0, 0, 0], // Jacobian z=1 (affine)
        scalar: scalar_be_to_limbs(&scalar_be),
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
