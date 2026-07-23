//! Bitcoin address derivation helpers — compiles to ~5 ns/hash + ~30 ns/addr.
//!
//! Public API kept tidy — the inner address-format functions are deliberate:
//! they mirror `scripts/run.py`'s 5 address types and expose them for future
//! candidate lists with non-P2PK addresses (bc1*).
#![allow(dead_code)]

use sha2::Digest;

// ── secp256k1 generator point ────────────────────────────────────────────────

/// The secp256k1 generator (base point) G, encoded as a compressed 33-byte
/// public key.  This is the canonical constant from [SEC 2 §2.7.1]:
///
/// ```text
/// G = 04 79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798
///        483ADA77 26A3C465 5DA4FBFC 0E1108A8 FD17B448 A6855419 9C47D08F FB10D4B8
/// ```
///
/// Used by the scanner hot path to turn "advance key by 1" into a single point
/// addition `P_{n+1} = P_n + G` instead of a full scalar multiplication
/// `P_{n+1} = (sk + 1) * G`.  Point add ≈ 10-20× cheaper than scalar mult.
pub const GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

/// Parse the canonical generator point into a `PublicKey`.  Cheap (one affine
/// decode), but still worth doing once per worker rather than per key.
#[inline]
pub fn generator_public_key() -> secp256k1::PublicKey {
    secp256k1::PublicKey::from_slice(&GENERATOR_COMPRESSED)
        .expect("secp256k1 generator point is a valid constant")
}

#[inline(always)]
pub fn hash160(data: &[u8]) -> [u8; 20] {
    let sha = sha2::Sha256::digest(data);
    let out = ripemd::Ripemd160::digest(sha);
    let mut buf = [0u8; 20];
    buf.copy_from_slice(&out);
    buf
}

const B58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

fn bs58_encode(data: &[u8]) -> String {
    if data.is_empty() { return String::new(); }
    let mut digits: Vec<u8> = Vec::new();
    for &b in data {
        let mut carry = b as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) << 8;
            *d = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            digits.push((carry % 58) as u8);
            carry /= 58;
        }
    }
    for &b in data { if b == 0 { digits.push(0); } else { break; } }
    digits.reverse();
    let mapped: Vec<u8> = digits.iter().map(|&i| B58_ALPHABET[i as usize]).collect();
    String::from_utf8(mapped).unwrap_or_default()
}

/// Decode a base58-encoded string back into its raw bytes.
///
/// The inverse of `bs58_encode`: leading `'1'` characters map to leading `0x00`
/// bytes, and any character outside the base58 alphabet returns `None`.
fn bs58_decode(input: &str) -> Option<Vec<u8>> {
    if input.is_empty() { return Some(Vec::new()); }
    let mut digits: Vec<u8> = Vec::new();
    for ch in input.bytes() {
        let idx = B58_ALPHABET.iter().position(|&a| a == ch)?;
        let mut carry = idx as u32;
        for d in digits.iter_mut() {
            carry += (*d as u32) * 58;
            *d = (carry & 0xff) as u8;
            carry >>= 8;
        }
        while carry > 0 {
            digits.push((carry & 0xff) as u8);
            carry >>= 8;
        }
    }
    let leading_zeros = input.bytes().take_while(|&b| b == b'1').count();
    digits.reverse();
    let mut out = vec![0u8; leading_zeros];
    out.extend_from_slice(&digits);
    Some(out)
}

/// Parse a legacy Base58Check Bitcoin address (P2PKH starting with `1`, or P2SH
/// starting with `3`) and return the 20-byte RIPEMD-160 hash it commits to.
///
/// Returns `None` if the input is not valid base58, is not exactly 25 bytes
/// after decoding, or fails the 4-byte checksum verification.  The version byte
/// is discarded — callers that need it can read `address.as_bytes()[0]`
/// (`0x00` = P2PKH, `0x05` = P2SH).
pub fn legacy_address_hash160(address: &str) -> Option<[u8; 20]> {
    let decoded = bs58_decode(address)?;
    if decoded.len() != 25 {
        return None;
    }
    let (payload, checksum) = decoded.split_at(21);
    let c1 = sha2::Sha256::digest(payload);
    let c2 = sha2::Sha256::digest(c1);
    if c2[..4] != *checksum {
        return None;
    }
    let mut hash = [0u8; 20];
    hash.copy_from_slice(&payload[1..]);
    Some(hash)
}

#[inline(always)]
pub fn base58check(payload: &[u8]) -> String {
    let c1 = sha2::Sha256::digest(payload);
    let c2 = sha2::Sha256::digest(c1);
    let mut full = Vec::with_capacity(payload.len() + 4);
    full.extend_from_slice(payload);
    full.extend_from_slice(&c2[..4]);
    bs58_encode(&full)
}

#[inline(always)]
pub fn p2pkh_compressed(pub_key: &[u8]) -> String {
    let h = hash160(pub_key);
    let mut pld = Vec::with_capacity(21);
    pld.push(0x00);
    pld.extend_from_slice(&h);
    base58check(&pld)
}

#[inline(always)]
pub fn p2pkh_uncompressed(pub_key: &[u8]) -> String {
    p2pkh_compressed(pub_key)  // same pubkey format; bytes differ only in input
}

#[inline(always)]
pub fn p2sh_p2wpkh(pub_key: &[u8]) -> String {
    let h = hash160(pub_key);
    let mut redeem = Vec::with_capacity(22);
    redeem.extend_from_slice(&[0x00, 0x14]);
    redeem.extend_from_slice(&h);
    let sh = hash160(&redeem);
    let mut pld = Vec::with_capacity(21);
    pld.push(0x05);
    pld.extend_from_slice(&sh);
    base58check(&pld)
}

pub fn p2wpkh(pub_key: &[u8]) -> String {
    let h = hash160(pub_key);
    bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_0, &h).unwrap_or_default()
}

pub fn p2tr(pub_key: &[u8]) -> String {
    let x   = &pub_key[1..];
    let tag = sha2::Sha256::digest(b"TapTweak");
    let t   = sha2::Sha256::digest([tag.as_ref(), tag.as_ref(), x].concat());

    // BIP340: t must be a non-zero scalar < curve order n.
    // We treat t == 0 or t >= n as invalid — same as Python.
    if t.iter().all(|b| *b == 0) {
        return String::new();
    }
    let Ok(t_sk) = secp256k1::SecretKey::from_byte_array(t.into()) else {
        return String::new();
    };
    // SECRET-key → scalar (libsecp256k1 0.29 has Scalar::from（key)
    let t_scalar = secp256k1::Scalar::from(t_sk);

    let secp     = secp256k1::Secp256k1::new();
    let internal = match secp256k1::PublicKey::from_slice(pub_key) {
        Ok(p)  => p,
        Err(_) => return String::new(),
    };
    let tweaked = internal
        .add_exp_tweak(&secp, &t_scalar)
        .unwrap_or(internal);
    let ox = tweaked.x_only_public_key().0.serialize();
    bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_1, &ox)
        .unwrap_or_default()
}
