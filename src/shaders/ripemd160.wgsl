// =============================================================================
// RIPEMD-160 Implementation for WGSL
// =============================================================================
// Used for Bitcoin hash160 = RIPEMD160(SHA256(pubkey))
// From kangaroo v0.1.0 (https://crates.io/crates/kangaroo/0.1.0)

// RIPEMD-160 initial values
const RH0: u32 = 0x67452301u;
const RH1: u32 = 0xefcdab89u;
const RH2: u32 = 0x98badcfeu;
const RH3: u32 = 0x10325476u;
const RH4: u32 = 0xc3d2e1f0u;

// Left rotate
fn rotl(x: u32, n: u32) -> u32 {
    return (x << n) | (x >> (32u - n));
}

// RIPEMD-160 functions
fn rmd_f(x: u32, y: u32, z: u32) -> u32 { return x ^ y ^ z; }
fn rmd_g(x: u32, y: u32, z: u32) -> u32 { return (x & y) | (~x & z); }
fn rmd_h(x: u32, y: u32, z: u32) -> u32 { return (x | ~y) ^ z; }
fn rmd_i(x: u32, y: u32, z: u32) -> u32 { return (x & z) | (y & ~z); }
fn rmd_j(x: u32, y: u32, z: u32) -> u32 { return x ^ (y | ~z); }

// Constants
const KL: array<u32, 5> = array<u32, 5>(0x00000000u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu);
const KR: array<u32, 5> = array<u32, 5>(0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0x00000000u);

// Message word selection (left path)
const RL: array<u32, 80> = array<u32, 80>(
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u,
    7u, 4u, 13u, 1u, 10u, 6u, 15u, 3u, 12u, 0u, 9u, 5u, 2u, 14u, 11u, 8u,
    3u, 10u, 14u, 4u, 9u, 15u, 8u, 1u, 2u, 7u, 0u, 6u, 13u, 11u, 5u, 12u,
    1u, 9u, 11u, 10u, 0u, 8u, 12u, 4u, 13u, 3u, 7u, 15u, 14u, 5u, 6u, 2u,
    4u, 0u, 5u, 9u, 7u, 12u, 2u, 10u, 14u, 1u, 3u, 8u, 11u, 6u, 15u, 13u
);

// Message word selection (right path)
const RR: array<u32, 80> = array<u32, 80>(
    5u, 14u, 7u, 0u, 9u, 2u, 11u, 4u, 13u, 6u, 15u, 8u, 1u, 10u, 3u, 12u,
    6u, 11u, 3u, 7u, 0u, 13u, 5u, 10u, 14u, 15u, 8u, 12u, 4u, 9u, 1u, 2u,
    15u, 5u, 1u, 3u, 7u, 14u, 6u, 9u, 11u, 8u, 12u, 2u, 10u, 0u, 4u, 13u,
    8u, 6u, 4u, 1u, 3u, 11u, 15u, 0u, 5u, 12u, 2u, 13u, 9u, 7u, 10u, 14u,
    12u, 15u, 10u, 4u, 1u, 5u, 8u, 7u, 6u, 2u, 13u, 14u, 0u, 3u, 9u, 11u
);

// Rotation amounts (left path)
const SL: array<u32, 80> = array<u32, 80>(
    11u, 14u, 15u, 12u, 5u, 8u, 7u, 9u, 11u, 13u, 14u, 15u, 6u, 7u, 9u, 8u,
    7u, 6u, 8u, 13u, 11u, 9u, 7u, 15u, 7u, 12u, 15u, 9u, 11u, 7u, 13u, 12u,
    11u, 13u, 6u, 7u, 14u, 9u, 13u, 15u, 14u, 8u, 13u, 6u, 5u, 12u, 7u, 5u,
    11u, 12u, 14u, 15u, 14u, 15u, 9u, 8u, 9u, 14u, 5u, 6u, 8u, 6u, 5u, 12u,
    9u, 15u, 5u, 11u, 6u, 8u, 13u, 12u, 5u, 12u, 13u, 14u, 11u, 8u, 5u, 6u
);

// Rotation amounts (right path)
const SR: array<u32, 80> = array<u32, 80>(
    8u, 9u, 9u, 11u, 13u, 15u, 15u, 5u, 7u, 7u, 8u, 11u, 14u, 14u, 12u, 6u,
    9u, 13u, 15u, 7u, 12u, 8u, 9u, 11u, 7u, 7u, 12u, 7u, 6u, 15u, 13u, 11u,
    9u, 7u, 15u, 11u, 8u, 6u, 6u, 14u, 12u, 13u, 5u, 14u, 13u, 13u, 7u, 5u,
    15u, 5u, 8u, 11u, 14u, 14u, 6u, 14u, 6u, 9u, 12u, 9u, 12u, 5u, 15u, 8u,
    8u, 5u, 12u, 9u, 12u, 5u, 14u, 6u, 8u, 13u, 6u, 5u, 15u, 13u, 11u, 11u
);

// RIPEMD-160 hash result (5 x u32 = 160 bits)
struct Rmd160Hash {
    h: array<u32, 5>
}

// Get function result for round
fn rmd_func(round: u32, x: u32, y: u32, z: u32) -> u32 {
    switch (round) {
        case 0u: { return rmd_f(x, y, z); }
        case 1u: { return rmd_g(x, y, z); }
        case 2u: { return rmd_h(x, y, z); }
        case 3u: { return rmd_i(x, y, z); }
        default: { return rmd_j(x, y, z); }
    }
}

fn rmd_func_r(round: u32, x: u32, y: u32, z: u32) -> u32 {
    switch (round) {
        case 0u: { return rmd_j(x, y, z); }
        case 1u: { return rmd_i(x, y, z); }
        case 2u: { return rmd_h(x, y, z); }
        case 3u: { return rmd_g(x, y, z); }
        default: { return rmd_f(x, y, z); }
    }
}

// Process single 512-bit block
fn rmd160_process_block(state: ptr<function, array<u32, 5>>, x: array<u32, 16>) {
    // Left path
    var al = (*state)[0];
    var bl = (*state)[1];
    var cl = (*state)[2];
    var dl = (*state)[3];
    var el = (*state)[4];

    // Right path
    var ar = (*state)[0];
    var br = (*state)[1];
    var cr = (*state)[2];
    var dr = (*state)[3];
    var er = (*state)[4];

    // 80 rounds
    for (var i = 0u; i < 80u; i = i + 1u) {
        let round = i / 16u;

        // Left path
        let fl = rmd_func(round, bl, cl, dl);
        var tl = al + fl + x[RL[i]] + KL[round];
        tl = rotl(tl, SL[i]) + el;
        al = el;
        el = dl;
        dl = rotl(cl, 10u);
        cl = bl;
        bl = tl;

        // Right path
        let fr = rmd_func_r(round, br, cr, dr);
        var tr = ar + fr + x[RR[i]] + KR[round];
        tr = rotl(tr, SR[i]) + er;
        ar = er;
        er = dr;
        dr = rotl(cr, 10u);
        cr = br;
        br = tr;
    }

    // Final addition
    let t = (*state)[1] + cl + dr;
    (*state)[1] = (*state)[2] + dl + er;
    (*state)[2] = (*state)[3] + el + ar;
    (*state)[3] = (*state)[4] + al + br;
    (*state)[4] = (*state)[0] + bl + cr;
    (*state)[0] = t;
}

// RIPEMD-160 hash of 32 bytes (SHA-256 output for hash160)
fn ripemd160_32bytes(data: array<u32, 8>) -> Rmd160Hash {
    var state = array<u32, 5>(RH0, RH1, RH2, RH3, RH4);

    // Convert from big-endian (SHA256 output) to little-endian (RIPEMD160 input)
    var x: array<u32, 16>;

    // Byte-swap each word
    x[0] = ((data[0] & 0xFFu) << 24u) | ((data[0] & 0xFF00u) << 8u) |
           ((data[0] >> 8u) & 0xFF00u) | ((data[0] >> 24u) & 0xFFu);
    x[1] = ((data[1] & 0xFFu) << 24u) | ((data[1] & 0xFF00u) << 8u) |
           ((data[1] >> 8u) & 0xFF00u) | ((data[1] >> 24u) & 0xFFu);
    x[2] = ((data[2] & 0xFFu) << 24u) | ((data[2] & 0xFF00u) << 8u) |
           ((data[2] >> 8u) & 0xFF00u) | ((data[2] >> 24u) & 0xFFu);
    x[3] = ((data[3] & 0xFFu) << 24u) | ((data[3] & 0xFF00u) << 8u) |
           ((data[3] >> 8u) & 0xFF00u) | ((data[3] >> 24u) & 0xFFu);
    x[4] = ((data[4] & 0xFFu) << 24u) | ((data[4] & 0xFF00u) << 8u) |
           ((data[4] >> 8u) & 0xFF00u) | ((data[4] >> 24u) & 0xFFu);
    x[5] = ((data[5] & 0xFFu) << 24u) | ((data[5] & 0xFF00u) << 8u) |
           ((data[5] >> 8u) & 0xFF00u) | ((data[5] >> 24u) & 0xFFu);
    x[6] = ((data[6] & 0xFFu) << 24u) | ((data[6] & 0xFF00u) << 8u) |
           ((data[6] >> 8u) & 0xFF00u) | ((data[6] >> 24u) & 0xFFu);
    x[7] = ((data[7] & 0xFFu) << 24u) | ((data[7] & 0xFF00u) << 8u) |
           ((data[7] >> 8u) & 0xFF00u) | ((data[7] >> 24u) & 0xFFu);

    // Padding: 0x80 followed by zeros, then length
    x[8] = 0x00000080u;  // Little-endian 0x80
    x[9] = 0u;
    x[10] = 0u;
    x[11] = 0u;
    x[12] = 0u;
    x[13] = 0u;
    x[14] = 256u;  // Length in bits (little-endian)
    x[15] = 0u;

    rmd160_process_block(&state, x);

    // Convert from little-endian (RIPEMD160 internal) to big-endian (Bitcoin convention)
    var result: Rmd160Hash;
    for (var i = 0u; i < 5u; i = i + 1u) {
        let w = state[i];
        result.h[i] = ((w & 0xFFu) << 24u) | ((w & 0xFF00u) << 8u) |
                      ((w >> 8u) & 0xFF00u) | ((w >> 24u) & 0xFFu);
    }
    return result;
}

// Hash160 = RIPEMD160(SHA256(pubkey)) - returns 20 bytes as 5 x u32
fn hash160_from_sha256(sha: array<u32, 8>) -> Rmd160Hash {
    return ripemd160_32bytes(sha);
}
