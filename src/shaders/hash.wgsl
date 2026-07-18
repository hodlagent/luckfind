// hash.wgsl — SHA256 + RIPEMD160 in WGSL (pure integer, no field arithmetic).
//
// SHA256: big-endian u32 words in, big-endian digest out.
// RIPEMD160: little-endian u32 words in, little-endian digest out.
// hash160(pubkey) = ripemd160(sha256(pubkey_bytes)).

// ── SHA256 (FIPS 180-4) ─────────────────────────────────────────────────────

const SHA256_K: array<u32, 64> = array<u32, 64>(
    0x428A2F98u, 0x71374491u, 0xB5C0FBCFu, 0xE9B5DBA5u,
    0x3956C25Bu, 0x59F111F1u, 0x923F82A4u, 0xAB1C5ED5u,
    0xD807AA98u, 0x12835B01u, 0x243185BEu, 0x550C7DC3u,
    0x72BE5D74u, 0x80DEB1FEu, 0x9BDC06A7u, 0xC19BF174u,
    0xE49B69C1u, 0xEFBE4786u, 0x0FC19DC6u, 0x240CA1CCu,
    0x2DE92C6Fu, 0x4A7484AAu, 0x5CB0A9DCu, 0x76F988DAu,
    0x983E5152u, 0xA831C66Du, 0xB00327C8u, 0xBF597FC7u,
    0xC6E00BF3u, 0xD5A79147u, 0x06CA6351u, 0x14292967u,
    0x27B70A85u, 0x2E1B2138u, 0x4D2C6DFCu, 0x53380D13u,
    0x650A7354u, 0x766A0ABBu, 0x81C2C92Eu, 0x92722C85u,
    0xA2BFE8A1u, 0xA81A664Bu, 0xC24B8B70u, 0xC76C51A3u,
    0xD192E819u, 0xD6990624u, 0xF40E3585u, 0x106AA070u,
    0x19A4C116u, 0x1E376C08u, 0x2748774Cu, 0x34B0BCB5u,
    0x391C0CB3u, 0x4ED8AA4Au, 0x5B9CCA4Fu, 0x682E6FF3u,
    0x748F82EEu, 0x78A5636Fu, 0x84C87814u, 0x8CC70208u,
    0x90BEFFFAu, 0xA4506CEBu, 0xBEF9A3F7u, 0xC67178F2u,
);

const SHA256_H0: array<u32, 8> = array<u32, 8>(
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
);

fn sha256_rotr(x: u32, n: u32) -> u32 { return (x >> n) | (x << (32u - n)); }
fn sha256_sigma1(e: u32) -> u32 { return sha256_rotr(e,6u) ^ sha256_rotr(e,11u) ^ sha256_rotr(e,25u); }
fn sha256_ch(e: u32, f: u32, g: u32) -> u32 { return (e & f) ^ (~e & g); }
fn sha256_sigma0(a: u32) -> u32 { return sha256_rotr(a,2u) ^ sha256_rotr(a,13u) ^ sha256_rotr(a,22u); }
fn sha256_maj(a: u32, b: u32, c: u32) -> u32 { return (a & b) ^ (a & c) ^ (b & c); }
fn sha256_gamma1(w: u32) -> u32 { return sha256_rotr(w,17u) ^ sha256_rotr(w,19u) ^ (w >> 10u); }
fn sha256_gamma0(w: u32) -> u32 { return sha256_rotr(w,7u) ^ sha256_rotr(w,18u) ^ (w >> 3u); }

fn sha256_block(msg: array<u32, 16>) -> array<u32, 8> {
    var w: array<u32, 64>;
    for (var i = 0u; i < 16u; i++) { w[i] = msg[i]; }
    for (var i = 16u; i < 64u; i++) {
        w[i] = sha256_gamma1(w[i-2u]) + w[i-7u] + sha256_gamma0(w[i-15u]) + w[i-16u];
    }
    var a = SHA256_H0[0]; var b = SHA256_H0[1]; var c = SHA256_H0[2]; var d = SHA256_H0[3];
    var e = SHA256_H0[4]; var f = SHA256_H0[5]; var g = SHA256_H0[6]; var h = SHA256_H0[7];
    for (var i = 0u; i < 64u; i++) {
        let t1 = h + sha256_sigma1(e) + sha256_ch(e,f,g) + SHA256_K[i] + w[i];
        let t2 = sha256_sigma0(a) + sha256_maj(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    var out: array<u32, 8>;
    out[0] = SHA256_H0[0] + a; out[1] = SHA256_H0[1] + b; out[2] = SHA256_H0[2] + c;
    out[3] = SHA256_H0[3] + d; out[4] = SHA256_H0[4] + e; out[5] = SHA256_H0[5] + f;
    out[6] = SHA256_H0[6] + g; out[7] = SHA256_H0[7] + h;
    return out;
}

fn rp_rotl(x: u32, n: u32) -> u32 { return (x << n) | (x >> (32u - n)); }

fn rp_f(j: u32, x: u32, y: u32, z: u32) -> u32 {
    if (j == 0u) { return x ^ y ^ z; }
    if (j == 1u) { return (x & y) | (~x & z); }
    if (j == 2u) { return (x | ~y) ^ z; }
    if (j == 3u) { return (x & z) | (y & ~z); }
    return x ^ (y | ~z);
}

const RP_H0: array<u32, 5> = array<u32, 5>(
    0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u,
);

fn ripemd160_block(m: array<u32, 16>) -> array<u32, 5> {
    var bb = RP_H0;
    var bbb = RP_H0;

    // LEFT LINE

    bb[0] = rp_rotl(bb[0] + rp_f(0u, bb[1], bb[2], bb[3]) + m[0] + 0u, 11u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(0u, bb[0], bb[1], bb[2]) + m[1] + 0u, 14u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(0u, bb[4], bb[0], bb[1]) + m[2] + 0u, 15u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(0u, bb[3], bb[4], bb[0]) + m[3] + 0u, 12u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(0u, bb[2], bb[3], bb[4]) + m[4] + 0u, 5u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(0u, bb[1], bb[2], bb[3]) + m[5] + 0u, 8u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(0u, bb[0], bb[1], bb[2]) + m[6] + 0u, 7u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(0u, bb[4], bb[0], bb[1]) + m[7] + 0u, 9u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(0u, bb[3], bb[4], bb[0]) + m[8] + 0u, 11u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(0u, bb[2], bb[3], bb[4]) + m[9] + 0u, 13u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(0u, bb[1], bb[2], bb[3]) + m[10] + 0u, 14u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(0u, bb[0], bb[1], bb[2]) + m[11] + 0u, 15u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(0u, bb[4], bb[0], bb[1]) + m[12] + 0u, 6u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(0u, bb[3], bb[4], bb[0]) + m[13] + 0u, 7u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(0u, bb[2], bb[3], bb[4]) + m[14] + 0u, 9u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(0u, bb[1], bb[2], bb[3]) + m[15] + 0u, 8u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(1u, bb[0], bb[1], bb[2]) + m[7] + 1518500249u, 7u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(1u, bb[4], bb[0], bb[1]) + m[4] + 1518500249u, 6u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(1u, bb[3], bb[4], bb[0]) + m[13] + 1518500249u, 8u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(1u, bb[2], bb[3], bb[4]) + m[1] + 1518500249u, 13u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(1u, bb[1], bb[2], bb[3]) + m[10] + 1518500249u, 11u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(1u, bb[0], bb[1], bb[2]) + m[6] + 1518500249u, 9u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(1u, bb[4], bb[0], bb[1]) + m[15] + 1518500249u, 7u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(1u, bb[3], bb[4], bb[0]) + m[3] + 1518500249u, 15u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(1u, bb[2], bb[3], bb[4]) + m[12] + 1518500249u, 7u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(1u, bb[1], bb[2], bb[3]) + m[0] + 1518500249u, 12u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(1u, bb[0], bb[1], bb[2]) + m[9] + 1518500249u, 15u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(1u, bb[4], bb[0], bb[1]) + m[5] + 1518500249u, 9u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(1u, bb[3], bb[4], bb[0]) + m[2] + 1518500249u, 11u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(1u, bb[2], bb[3], bb[4]) + m[14] + 1518500249u, 7u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(1u, bb[1], bb[2], bb[3]) + m[11] + 1518500249u, 13u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(1u, bb[0], bb[1], bb[2]) + m[8] + 1518500249u, 12u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(2u, bb[4], bb[0], bb[1]) + m[3] + 1859775393u, 11u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(2u, bb[3], bb[4], bb[0]) + m[10] + 1859775393u, 13u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(2u, bb[2], bb[3], bb[4]) + m[14] + 1859775393u, 6u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(2u, bb[1], bb[2], bb[3]) + m[4] + 1859775393u, 7u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(2u, bb[0], bb[1], bb[2]) + m[9] + 1859775393u, 14u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(2u, bb[4], bb[0], bb[1]) + m[15] + 1859775393u, 9u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(2u, bb[3], bb[4], bb[0]) + m[8] + 1859775393u, 13u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(2u, bb[2], bb[3], bb[4]) + m[1] + 1859775393u, 15u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(2u, bb[1], bb[2], bb[3]) + m[2] + 1859775393u, 14u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(2u, bb[0], bb[1], bb[2]) + m[7] + 1859775393u, 8u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(2u, bb[4], bb[0], bb[1]) + m[0] + 1859775393u, 13u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(2u, bb[3], bb[4], bb[0]) + m[6] + 1859775393u, 6u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(2u, bb[2], bb[3], bb[4]) + m[13] + 1859775393u, 5u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(2u, bb[1], bb[2], bb[3]) + m[11] + 1859775393u, 12u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(2u, bb[0], bb[1], bb[2]) + m[5] + 1859775393u, 7u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(2u, bb[4], bb[0], bb[1]) + m[12] + 1859775393u, 5u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(3u, bb[3], bb[4], bb[0]) + m[1] + 2400959708u, 11u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(3u, bb[2], bb[3], bb[4]) + m[9] + 2400959708u, 12u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(3u, bb[1], bb[2], bb[3]) + m[11] + 2400959708u, 14u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(3u, bb[0], bb[1], bb[2]) + m[10] + 2400959708u, 15u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(3u, bb[4], bb[0], bb[1]) + m[0] + 2400959708u, 14u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(3u, bb[3], bb[4], bb[0]) + m[8] + 2400959708u, 15u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(3u, bb[2], bb[3], bb[4]) + m[12] + 2400959708u, 9u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(3u, bb[1], bb[2], bb[3]) + m[4] + 2400959708u, 8u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(3u, bb[0], bb[1], bb[2]) + m[13] + 2400959708u, 9u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(3u, bb[4], bb[0], bb[1]) + m[3] + 2400959708u, 14u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(3u, bb[3], bb[4], bb[0]) + m[7] + 2400959708u, 5u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(3u, bb[2], bb[3], bb[4]) + m[15] + 2400959708u, 6u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(3u, bb[1], bb[2], bb[3]) + m[14] + 2400959708u, 8u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(3u, bb[0], bb[1], bb[2]) + m[5] + 2400959708u, 6u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(3u, bb[4], bb[0], bb[1]) + m[6] + 2400959708u, 5u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(3u, bb[3], bb[4], bb[0]) + m[2] + 2400959708u, 12u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(4u, bb[2], bb[3], bb[4]) + m[4] + 2840853838u, 9u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(4u, bb[1], bb[2], bb[3]) + m[0] + 2840853838u, 15u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(4u, bb[0], bb[1], bb[2]) + m[5] + 2840853838u, 5u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(4u, bb[4], bb[0], bb[1]) + m[9] + 2840853838u, 11u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(4u, bb[3], bb[4], bb[0]) + m[7] + 2840853838u, 6u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(4u, bb[2], bb[3], bb[4]) + m[12] + 2840853838u, 8u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(4u, bb[1], bb[2], bb[3]) + m[2] + 2840853838u, 13u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(4u, bb[0], bb[1], bb[2]) + m[10] + 2840853838u, 12u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(4u, bb[4], bb[0], bb[1]) + m[14] + 2840853838u, 5u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(4u, bb[3], bb[4], bb[0]) + m[1] + 2840853838u, 12u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(4u, bb[2], bb[3], bb[4]) + m[3] + 2840853838u, 13u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);
    bb[0] = rp_rotl(bb[0] + rp_f(4u, bb[1], bb[2], bb[3]) + m[8] + 2840853838u, 14u) + bb[4];
    bb[2] = rp_rotl(bb[2], 10u);
    bb[4] = rp_rotl(bb[4] + rp_f(4u, bb[0], bb[1], bb[2]) + m[11] + 2840853838u, 11u) + bb[3];
    bb[1] = rp_rotl(bb[1], 10u);
    bb[3] = rp_rotl(bb[3] + rp_f(4u, bb[4], bb[0], bb[1]) + m[6] + 2840853838u, 8u) + bb[2];
    bb[0] = rp_rotl(bb[0], 10u);
    bb[2] = rp_rotl(bb[2] + rp_f(4u, bb[3], bb[4], bb[0]) + m[15] + 2840853838u, 5u) + bb[1];
    bb[4] = rp_rotl(bb[4], 10u);
    bb[1] = rp_rotl(bb[1] + rp_f(4u, bb[2], bb[3], bb[4]) + m[13] + 2840853838u, 6u) + bb[0];
    bb[3] = rp_rotl(bb[3], 10u);

    // RIGHT LINE

    bbb[0] = rp_rotl(bbb[0] + rp_f(4u, bbb[1], bbb[2], bbb[3]) + m[5] + 1352829926u, 8u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(4u, bbb[0], bbb[1], bbb[2]) + m[14] + 1352829926u, 9u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(4u, bbb[4], bbb[0], bbb[1]) + m[7] + 1352829926u, 9u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(4u, bbb[3], bbb[4], bbb[0]) + m[0] + 1352829926u, 11u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(4u, bbb[2], bbb[3], bbb[4]) + m[9] + 1352829926u, 13u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(4u, bbb[1], bbb[2], bbb[3]) + m[2] + 1352829926u, 15u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(4u, bbb[0], bbb[1], bbb[2]) + m[11] + 1352829926u, 15u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(4u, bbb[4], bbb[0], bbb[1]) + m[4] + 1352829926u, 5u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(4u, bbb[3], bbb[4], bbb[0]) + m[13] + 1352829926u, 7u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(4u, bbb[2], bbb[3], bbb[4]) + m[6] + 1352829926u, 7u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(4u, bbb[1], bbb[2], bbb[3]) + m[15] + 1352829926u, 8u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(4u, bbb[0], bbb[1], bbb[2]) + m[8] + 1352829926u, 11u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(4u, bbb[4], bbb[0], bbb[1]) + m[1] + 1352829926u, 14u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(4u, bbb[3], bbb[4], bbb[0]) + m[10] + 1352829926u, 14u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(4u, bbb[2], bbb[3], bbb[4]) + m[3] + 1352829926u, 12u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(4u, bbb[1], bbb[2], bbb[3]) + m[12] + 1352829926u, 6u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(3u, bbb[0], bbb[1], bbb[2]) + m[6] + 1548603684u, 9u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(3u, bbb[4], bbb[0], bbb[1]) + m[11] + 1548603684u, 13u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(3u, bbb[3], bbb[4], bbb[0]) + m[3] + 1548603684u, 15u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(3u, bbb[2], bbb[3], bbb[4]) + m[7] + 1548603684u, 7u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(3u, bbb[1], bbb[2], bbb[3]) + m[0] + 1548603684u, 12u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(3u, bbb[0], bbb[1], bbb[2]) + m[13] + 1548603684u, 8u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(3u, bbb[4], bbb[0], bbb[1]) + m[5] + 1548603684u, 9u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(3u, bbb[3], bbb[4], bbb[0]) + m[10] + 1548603684u, 11u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(3u, bbb[2], bbb[3], bbb[4]) + m[14] + 1548603684u, 7u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(3u, bbb[1], bbb[2], bbb[3]) + m[15] + 1548603684u, 7u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(3u, bbb[0], bbb[1], bbb[2]) + m[8] + 1548603684u, 12u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(3u, bbb[4], bbb[0], bbb[1]) + m[12] + 1548603684u, 7u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(3u, bbb[3], bbb[4], bbb[0]) + m[4] + 1548603684u, 6u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(3u, bbb[2], bbb[3], bbb[4]) + m[9] + 1548603684u, 15u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(3u, bbb[1], bbb[2], bbb[3]) + m[1] + 1548603684u, 13u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(3u, bbb[0], bbb[1], bbb[2]) + m[2] + 1548603684u, 11u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(2u, bbb[4], bbb[0], bbb[1]) + m[15] + 1836072691u, 9u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(2u, bbb[3], bbb[4], bbb[0]) + m[5] + 1836072691u, 7u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(2u, bbb[2], bbb[3], bbb[4]) + m[1] + 1836072691u, 15u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(2u, bbb[1], bbb[2], bbb[3]) + m[3] + 1836072691u, 11u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(2u, bbb[0], bbb[1], bbb[2]) + m[7] + 1836072691u, 8u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(2u, bbb[4], bbb[0], bbb[1]) + m[14] + 1836072691u, 6u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(2u, bbb[3], bbb[4], bbb[0]) + m[6] + 1836072691u, 6u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(2u, bbb[2], bbb[3], bbb[4]) + m[9] + 1836072691u, 14u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(2u, bbb[1], bbb[2], bbb[3]) + m[11] + 1836072691u, 12u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(2u, bbb[0], bbb[1], bbb[2]) + m[8] + 1836072691u, 13u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(2u, bbb[4], bbb[0], bbb[1]) + m[12] + 1836072691u, 5u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(2u, bbb[3], bbb[4], bbb[0]) + m[2] + 1836072691u, 14u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(2u, bbb[2], bbb[3], bbb[4]) + m[10] + 1836072691u, 13u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(2u, bbb[1], bbb[2], bbb[3]) + m[0] + 1836072691u, 13u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(2u, bbb[0], bbb[1], bbb[2]) + m[4] + 1836072691u, 7u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(2u, bbb[4], bbb[0], bbb[1]) + m[13] + 1836072691u, 5u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(1u, bbb[3], bbb[4], bbb[0]) + m[8] + 2053994217u, 15u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(1u, bbb[2], bbb[3], bbb[4]) + m[6] + 2053994217u, 5u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(1u, bbb[1], bbb[2], bbb[3]) + m[4] + 2053994217u, 8u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(1u, bbb[0], bbb[1], bbb[2]) + m[1] + 2053994217u, 11u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(1u, bbb[4], bbb[0], bbb[1]) + m[3] + 2053994217u, 14u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(1u, bbb[3], bbb[4], bbb[0]) + m[11] + 2053994217u, 14u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(1u, bbb[2], bbb[3], bbb[4]) + m[15] + 2053994217u, 6u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(1u, bbb[1], bbb[2], bbb[3]) + m[0] + 2053994217u, 14u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(1u, bbb[0], bbb[1], bbb[2]) + m[5] + 2053994217u, 6u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(1u, bbb[4], bbb[0], bbb[1]) + m[12] + 2053994217u, 9u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(1u, bbb[3], bbb[4], bbb[0]) + m[2] + 2053994217u, 12u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(1u, bbb[2], bbb[3], bbb[4]) + m[13] + 2053994217u, 9u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(1u, bbb[1], bbb[2], bbb[3]) + m[9] + 2053994217u, 12u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(1u, bbb[0], bbb[1], bbb[2]) + m[7] + 2053994217u, 5u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(1u, bbb[4], bbb[0], bbb[1]) + m[10] + 2053994217u, 15u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(1u, bbb[3], bbb[4], bbb[0]) + m[14] + 2053994217u, 8u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(0u, bbb[2], bbb[3], bbb[4]) + m[12] + 0u, 8u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(0u, bbb[1], bbb[2], bbb[3]) + m[15] + 0u, 5u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(0u, bbb[0], bbb[1], bbb[2]) + m[10] + 0u, 12u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(0u, bbb[4], bbb[0], bbb[1]) + m[4] + 0u, 9u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(0u, bbb[3], bbb[4], bbb[0]) + m[1] + 0u, 12u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(0u, bbb[2], bbb[3], bbb[4]) + m[5] + 0u, 5u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(0u, bbb[1], bbb[2], bbb[3]) + m[8] + 0u, 14u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(0u, bbb[0], bbb[1], bbb[2]) + m[7] + 0u, 6u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(0u, bbb[4], bbb[0], bbb[1]) + m[6] + 0u, 8u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(0u, bbb[3], bbb[4], bbb[0]) + m[2] + 0u, 13u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(0u, bbb[2], bbb[3], bbb[4]) + m[13] + 0u, 6u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);
    bbb[0] = rp_rotl(bbb[0] + rp_f(0u, bbb[1], bbb[2], bbb[3]) + m[14] + 0u, 5u) + bbb[4];
    bbb[2] = rp_rotl(bbb[2], 10u);
    bbb[4] = rp_rotl(bbb[4] + rp_f(0u, bbb[0], bbb[1], bbb[2]) + m[0] + 0u, 15u) + bbb[3];
    bbb[1] = rp_rotl(bbb[1], 10u);
    bbb[3] = rp_rotl(bbb[3] + rp_f(0u, bbb[4], bbb[0], bbb[1]) + m[3] + 0u, 13u) + bbb[2];
    bbb[0] = rp_rotl(bbb[0], 10u);
    bbb[2] = rp_rotl(bbb[2] + rp_f(0u, bbb[3], bbb[4], bbb[0]) + m[9] + 0u, 11u) + bbb[1];
    bbb[4] = rp_rotl(bbb[4], 10u);
    bbb[1] = rp_rotl(bbb[1] + rp_f(0u, bbb[2], bbb[3], bbb[4]) + m[11] + 0u, 11u) + bbb[0];
    bbb[3] = rp_rotl(bbb[3], 10u);

    // Combine
    bbb[3] = bbb[3] + RP_H0[1] + bb[2];
    var h1 = RP_H0[2] + bb[3] + bbb[4];
    var h2 = RP_H0[3] + bb[4] + bbb[0];
    var h3 = RP_H0[4] + bb[0] + bbb[1];
    var h4 = RP_H0[0] + bb[1] + bbb[2];
    var h0 = bbb[3];
    return array<u32, 5>(h0, h1, h2, h3, h4);
}


