// =============================================================================
// SHA-256 Implementation for WGSL
// =============================================================================
// Optimized for hashing Bitcoin public keys (33 or 65 bytes)
// From kangaroo v0.1.0 (https://crates.io/crates/kangaroo/0.1.0)

// SHA-256 initial hash values (first 32 bits of fractional parts of sqrt of first 8 primes)
const H0: u32 = 0x6a09e667u;
const H1: u32 = 0xbb67ae85u;
const H2: u32 = 0x3c6ef372u;
const H3: u32 = 0xa54ff53au;
const H4: u32 = 0x510e527fu;
const H5: u32 = 0x9b05688cu;
const H6: u32 = 0x1f83d9abu;
const H7: u32 = 0x5be0cd19u;

// SHA-256 round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
const K: array<u32, 64> = array<u32, 64>(
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
);

// Right rotate
fn rotr(x: u32, n: u32) -> u32 {
    return (x >> n) | (x << (32u - n));
}

// SHA-256 functions
fn ch(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ (~x & z);
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
    return (x & y) ^ (x & z) ^ (y & z);
}

fn sigma0(x: u32) -> u32 {
    return rotr(x, 2u) ^ rotr(x, 13u) ^ rotr(x, 22u);
}

fn sigma1(x: u32) -> u32 {
    return rotr(x, 6u) ^ rotr(x, 11u) ^ rotr(x, 25u);
}

fn gamma0(x: u32) -> u32 {
    return rotr(x, 7u) ^ rotr(x, 18u) ^ (x >> 3u);
}

fn gamma1(x: u32) -> u32 {
    return rotr(x, 17u) ^ rotr(x, 19u) ^ (x >> 10u);
}

// SHA-256 hash result (8 x u32 = 256 bits)
struct Sha256Hash {
    h: array<u32, 8>
}

// Process single 512-bit block
fn sha256_process_block(state: ptr<function, array<u32, 8>>, block: array<u32, 16>) {
    // Message schedule
    var w: array<u32, 64>;

    // First 16 words are the block itself
    w[0] = block[0]; w[1] = block[1]; w[2] = block[2]; w[3] = block[3];
    w[4] = block[4]; w[5] = block[5]; w[6] = block[6]; w[7] = block[7];
    w[8] = block[8]; w[9] = block[9]; w[10] = block[10]; w[11] = block[11];
    w[12] = block[12]; w[13] = block[13]; w[14] = block[14]; w[15] = block[15];

    // Extend to 64 words
    for (var i = 16u; i < 64u; i = i + 1u) {
        w[i] = gamma1(w[i - 2u]) + w[i - 7u] + gamma0(w[i - 15u]) + w[i - 16u];
    }

    // Working variables
    var a = (*state)[0];
    var b = (*state)[1];
    var c = (*state)[2];
    var d = (*state)[3];
    var e = (*state)[4];
    var f = (*state)[5];
    var g = (*state)[6];
    var h = (*state)[7];

    // 64 rounds
    for (var i = 0u; i < 64u; i = i + 1u) {
        let t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        let t2 = sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add to state
    (*state)[0] = (*state)[0] + a;
    (*state)[1] = (*state)[1] + b;
    (*state)[2] = (*state)[2] + c;
    (*state)[3] = (*state)[3] + d;
    (*state)[4] = (*state)[4] + e;
    (*state)[5] = (*state)[5] + f;
    (*state)[6] = (*state)[6] + g;
    (*state)[7] = (*state)[7] + h;
}

// SHA-256 hash of 33 bytes (compressed public key)
// Input: 33 bytes as array of u32 (9 words, last word partial)
fn sha256_33bytes(data: array<u32, 9>) -> Sha256Hash {
    var state = array<u32, 8>(H0, H1, H2, H3, H4, H5, H6, H7);

    // Build padded block (33 bytes + padding + length = 64 bytes = 1 block)
    // Data: 33 bytes = 8 full words + 1 byte
    // Padding: 0x80 after data, then zeros, then 64-bit length
    var block: array<u32, 16>;

    // First 8 words of data (big-endian)
    block[0] = data[0];
    block[1] = data[1];
    block[2] = data[2];
    block[3] = data[3];
    block[4] = data[4];
    block[5] = data[5];
    block[6] = data[6];
    block[7] = data[7];

    // 9th word: 1 byte of data + 0x80 padding
    block[8] = (data[8] & 0xFF000000u) | 0x00800000u;

    // Zeros
    block[9] = 0u;
    block[10] = 0u;
    block[11] = 0u;
    block[12] = 0u;
    block[13] = 0u;

    // Length in bits (33 * 8 = 264 = 0x108)
    block[14] = 0u;
    block[15] = 264u;

    sha256_process_block(&state, block);

    var result: Sha256Hash;
    result.h = state;
    return result;
}

// SHA-256 hash of 65 bytes (uncompressed public key)
fn sha256_65bytes(data: array<u32, 17>) -> Sha256Hash {
    var state = array<u32, 8>(H0, H1, H2, H3, H4, H5, H6, H7);

    // First block: 64 bytes of data
    var block1: array<u32, 16>;
    block1[0] = data[0]; block1[1] = data[1]; block1[2] = data[2]; block1[3] = data[3];
    block1[4] = data[4]; block1[5] = data[5]; block1[6] = data[6]; block1[7] = data[7];
    block1[8] = data[8]; block1[9] = data[9]; block1[10] = data[10]; block1[11] = data[11];
    block1[12] = data[12]; block1[13] = data[13]; block1[14] = data[14]; block1[15] = data[15];

    sha256_process_block(&state, block1);

    // Second block: 1 byte data + padding + length
    var block2: array<u32, 16>;
    block2[0] = (data[16] & 0xFF000000u) | 0x00800000u;
    block2[1] = 0u; block2[2] = 0u; block2[3] = 0u;
    block2[4] = 0u; block2[5] = 0u; block2[6] = 0u; block2[7] = 0u;
    block2[8] = 0u; block2[9] = 0u; block2[10] = 0u; block2[11] = 0u;
    block2[12] = 0u; block2[13] = 0u;
    block2[14] = 0u;
    block2[15] = 520u;  // 65 * 8 bits

    sha256_process_block(&state, block2);

    var result: Sha256Hash;
    result.h = state;
    return result;
}

// SHA-256 hash of 32 bytes (for double-SHA256 or hash160)
fn sha256_32bytes(data: array<u32, 8>) -> Sha256Hash {
    var state = array<u32, 8>(H0, H1, H2, H3, H4, H5, H6, H7);

    // Single block: 32 bytes data + 0x80 + zeros + length
    var block: array<u32, 16>;
    block[0] = data[0]; block[1] = data[1]; block[2] = data[2]; block[3] = data[3];
    block[4] = data[4]; block[5] = data[5]; block[6] = data[6]; block[7] = data[7];
    block[8] = 0x80000000u;  // Padding
    block[9] = 0u; block[10] = 0u; block[11] = 0u;
    block[12] = 0u; block[13] = 0u;
    block[14] = 0u;
    block[15] = 256u;  // 32 * 8 bits

    sha256_process_block(&state, block);

    var result: Sha256Hash;
    result.h = state;
    return result;
}

// SHA-256 hash of variable length data (up to 55 bytes, fits in one block)
// Input: data as array of 16 u32 (big-endian, pre-filled), length in bytes
// Returns: array of 8 u32 (hash result)
fn sha256_one_block(data: array<u32, 16>, len: u32) -> array<u32, 8> {
    var state = array<u32, 8>(H0, H1, H2, H3, H4, H5, H6, H7);

    // Build padded block
    var block: array<u32, 16>;

    // Copy data words
    let full_words = len / 4u;
    let remainder = len % 4u;

    // Copy full words
    for (var i = 0u; i < 16u; i = i + 1u) {
        if (i < full_words) {
            block[i] = data[i];
        } else if (i == full_words) {
            // Partial word + padding byte
            if (remainder == 0u) {
                block[i] = 0x80000000u;
            } else if (remainder == 1u) {
                block[i] = (data[i] & 0xFF000000u) | 0x00800000u;
            } else if (remainder == 2u) {
                block[i] = (data[i] & 0xFFFF0000u) | 0x00008000u;
            } else { // remainder == 3
                block[i] = (data[i] & 0xFFFFFF00u) | 0x00000080u;
            }
        } else {
            block[i] = 0u;
        }
    }

    // Length in bits (big-endian, 64-bit value in last 2 words)
    block[14] = 0u;
    block[15] = len * 8u;

    sha256_process_block(&state, block);

    return state;
}
