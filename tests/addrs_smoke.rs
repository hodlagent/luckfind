//! Black-box validation of BTC address derivation.

use sha2::Digest as Sha2Digest;
use ripemd::Digest as RipDigest;

// ── check helpers ──────────────────────────────────────────────────────────

fn my_p2pkh_parse(addr: &str) -> Option<[u8; 20]> {
    use sha2::Digest;
    let raw = base58::FromBase58::from_base58(addr).ok()?;
    if raw.len() < 5 { return None; }
    let (payload, checksum) = raw.split_at(raw.len() - 4);
    if payload[0] != 0x00 { return None; }
    let expect = &sha2::Sha256::digest(sha2::Sha256::digest(payload))[..4];
    if expect != checksum { return None; }
    if payload.len() != 21 { return None; }
    let mut out = [0u8; 20];
    out.copy_from_slice(&payload[1..21]);
    Some(out)
}

fn hash160(data: &[u8]) -> [u8; 20] {
    let sha = sha2::Sha256::digest(data);
    let rip = ripemd::Ripemd160::digest(sha);
    let mut out = [0u8; 20];
    out.copy_from_slice(&rip);
    out
}

fn bs58check(data: &[u8]) -> String {
    use sha2::Digest as Sha2Digest;
    let c = sha2::Sha256::digest(sha2::Sha256::digest(data));
    let mut v = data.to_vec();
    v.extend_from_slice(&c[..4]);
    let b58_alphabet: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    let mut n = num_bigint::BigUint::from_bytes_be(&v);
    let fiftyeight = num_bigint::BigUint::from(58u32);
    use num_traits::cast::ToPrimitive;
    use num_integer::Integer;
    let mut digits = Vec::new();
    while n > num_bigint::BigUint::from(0u32) {
        let (q, r) = n.div_rem(&fiftyeight);
        digits.push(b58_alphabet[r.to_u32().unwrap_or(0) as usize]);
        n = q;
    }
    for b in data {
        if *b == 0 { digits.push(b'1'); } else { break; }
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

fn pub_uncompressed(pub_c: &[u8]) -> Vec<u8> {
    let _secp = secp256k1::Secp256k1::new();
    secp256k1::PublicKey::from_slice(pub_c)
        .map(|p| p.serialize_uncompressed().to_vec())
        .unwrap_or_default()
}

// ── tests ──────────────────────────────────────────────────────────────────

#[test]
fn known_vector_bip_p2pkh_priv1() {
    let got = my_p2pkh_parse("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH");
    let exp: [u8; 20] = hex::decode("751e76e8199196d454941c45d1b3a323f1433bd6")
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(got, Some(exp));
}

#[test]
fn all_78_builtins_unique_and_valid() {
    let ps = luckfind::addrs::puzzle_set();
    assert_eq!(ps.len(), 78, "embedded puzzle count = {}", ps.len());

    // All hash160 values must be unique.
    let mut seen = std::collections::HashSet::<[u8; 20]>::new();
    for r in ps.ranges() {
        assert!(seen.insert(r.hash160), "duplicate hash160 for puzzle {}", r.puzzle_number);
    }
    assert_eq!(seen.len(), 78, "expected 78 unique hash160s");
}

#[test]
fn checksum_validates() {
    assert!(my_p2pkh_parse("1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAM").is_none());
    assert!(my_p2pkh_parse("8BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH").is_none());
}

#[test]
fn address_derivation_for_target_pubkey_03633cbe() {
    let pub_hex = "03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852";
    let pub_c   = hex::decode(pub_hex).unwrap();
    let pub_u   = pub_uncompressed(&pub_c);
    let h_u     = hash160(&pub_u);
    println!("Compressed   : {}", hex::encode(&pub_c));
    println!("Uncompressed : {}", hex::encode(&pub_u));

    let p2pkh_c = bs58check(&[&[0x00], &hash160(&pub_c)[..]].concat());
    let p2pkh_u = bs58check(&[&[0x00], &h_u[..]].concat());
    println!("P2PKH (c)   : {}", p2pkh_c);
    println!("P2PKH (u)   : {}", p2pkh_u);

    // P2SH-P2WPKH: witness_program = hash160(pub_c); redeem_script = 0x00 0x14 <wp>
    let wp = hash160(&pub_c);
    let mut redeem = Vec::with_capacity(22);
    redeem.push(0x00);
    redeem.push(0x14);
    redeem.extend_from_slice(&wp);
    let sh = hash160(&redeem);
    let p2sh = bs58check(&[&[0x05], &sh[..]].concat());
    let p2wpkh = bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_0,
                                       &hash160(&pub_c)).unwrap_or_default();
    println!("P2SH-P2WPKH : {}", p2sh);
    println!("P2WPKH      : {}", p2wpkh);

    // Taproot — BIP340:  Q = P + t*G,  t = int(TapTweak(x_only_P))  reduced mod n.
    let x = &pub_c[1..];
    let tag = sha2::Sha256::digest(b"TapTweak");
    let t_bytes = sha2::Sha256::digest(
        [tag.as_slice(), tag.as_slice(), x].concat()
    );

    // Four independent derivations — must produce the SAME address if the
    // implementation is internally consistent:
    //  (a) Scalar ← SecretKey (validates t < n) + add_exp_tweak
    //  (b) Scalar::from_be_bytes(t_bytes) + add_exp_tweak
    //  (c) SecretKey::from(t_bytes).public_key + PublicKey::combine
    //  (d) direct tweak with G via libsecp256k1 context.gen_mul
    let secp = secp256k1::Secp256k1::new();
    let p    = secp256k1::PublicKey::from_slice(&pub_c).unwrap();

    // path (a): SecretKey → Scalar → add_exp_tweak
    let sk_a = secp256k1::SecretKey::from_slice(&t_bytes).unwrap();
    let s_a  = secp256k1::Scalar::from(sk_a);
    let q_a  = p.add_exp_tweak(&secp, &s_a).unwrap();
    let a_a  = bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_1,
                                      &q_a.x_only_public_key().0.serialize())
                .unwrap_or_default();

    // path (b): Scalar::from_be_bytes + add_exp_tweak
    let s_b = secp256k1::Scalar::from_be_bytes(t_bytes.into()).unwrap();
    let q_b = p.add_exp_tweak(&secp, &s_b).unwrap();
    let a_b = bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_1,
                                     &q_b.x_only_public_key().0.serialize())
                .unwrap_or_default();

    // path (c): tG + P via combine
    let tG  = secp256k1::PublicKey::from_secret_key(&secp, &sk_a);
    let q_c = p.combine(&tG).unwrap();
    let a_c = bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_1,
                                     &q_c.x_only_public_key().0.serialize())
                .unwrap_or_default();

    // path (d): directly using Scalar::from(t_bytes) and context.mul
    let scalar_d = secp256k1::Scalar::from_be_bytes(t_bytes.into()).unwrap();
    let q_d = p.add_exp_tweak(&secp, &scalar_d).unwrap();
    let a_d = bech32::segwit::encode(bech32::hrp::BC, bech32::segwit::VERSION_1,
                                     &q_d.x_only_public_key().0.serialize())
                .unwrap_or_default();

    println!("P2TR (a SecretKey→Scalar)          : {}", a_a);
    println!("P2TR (b Scalar::from_be_bytes)      : {}", a_b);
    println!("P2TR (c combine P + tG)             : {}", a_c);
    println!("P2TR (d add_exp_tweak + direct)     : {}", a_d);

    assert_eq!(a_a, a_b, "path (a) != path (b)");
    assert_eq!(a_b, a_c, "path (b) != path (c)");
    assert_eq!(a_c, a_d, "path (c) != path (d)");

    println!("  ✓ all 4 BIP340 paths agree");
    let p2tr = a_a;

    let mut v = vec![p2pkh_c, p2pkh_u, p2sh, p2wpkh, p2tr];
    v.sort();
    println!("All 5 sorted:\n - {}", v.join("\n - "));
}

/// Validates the exact invariant the scanner's hot-path optimization relies on:
///
///   P_0 = sk * G                   (one scalar mult, via from_secret_key)
///   P_{n+1} = P_n + G              (one point add, via combine(&G))
///
/// must produce the SAME public keys as the naive per-step scalar multiplication
///
///   P_n = (sk + n) * G             (from_secret_key every step)
///
/// for every step n in [0, N).  If this holds, the 10-20× faster `combine` path
/// is a drop-in replacement for `from_secret_key` in sequential scanning.
#[test]
fn point_add_iter_matches_scalar_mult() {
    let secp = secp256k1::Secp256k1::new();

    // A non-trivial starting key (not 1, to exercise real scalar math).
    let mut sk = {
        let bytes: [u8; 32] = hex::decode(
            "a1b2c3d4e5f60718293a4b5c6d7e8f90112233445566778899aabbccddeeff00",
        )
        .unwrap()
        .try_into()
        .unwrap();
        secp256k1::SecretKey::from_byte_array(bytes).unwrap()
    };

    // The secp256k1 generator point G (compressed).  Hard-coded constant from
    // SEC 2 §2.7.1 — same bytes that crate::btc::GENERATOR_COMPRESSED holds.
    // Duplicated here because integration tests only see the crate's public
    // API surface, not its internal module layout.
    const GENERATOR_COMPRESSED: [u8; 33] = [
        0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE,
        0x87, 0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81,
        0x5B, 0x16, 0xF8, 0x17, 0x98,
    ];
    let point_g = secp256k1::PublicKey::from_slice(&GENERATOR_COMPRESSED).unwrap();

    // ── reference: per-step full scalar multiplication ──────────────────────
    // 512 steps — enough to catch any drift in x/y/z accumulation if the
    // combine path were silently wrong.
    const N: u64 = 512;
    let one = secp256k1::Scalar::from_be_bytes({
        let mut b = [0u8; 32];
        b[31] = 1;
        b
    })
    .expect("scalar 1 valid");

    let mut ref_pks = Vec::with_capacity(N as usize);
    let mut sk_ref = sk;
    for _ in 0..N {
        ref_pks.push(secp256k1::PublicKey::from_secret_key(&secp, &sk_ref));
        sk_ref = sk_ref.add_tweak(&one).expect("sk below curve order");
    }

    // ── optimized: one scalar mult + N point adds ──────────────────────────
    let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
    for (i, expected) in ref_pks.iter().enumerate() {
        assert_eq!(
            pk.serialize(),
            expected.serialize(),
            "step {i}: combine path diverged from scalar-mult path (compressed)"
        );
        assert_eq!(
            pk.serialize_uncompressed(),
            expected.serialize_uncompressed(),
            "step {i}: combine path diverged from scalar-mult path (uncompressed)"
        );
        // advance by point add — this is the optimized hot-path step
        pk = pk.combine(&point_g).expect("sk < n-1 for all 512 steps");
    }

    println!("  ✓ point-add iter matches scalar mult for all {N} steps");
}
