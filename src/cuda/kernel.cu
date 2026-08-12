// =============================================================================
// CUDA kernel for secp256k1 key scanning with batch Montgomery inversion.
//
// Mirrors the WGSL shader algorithm (src/shaders/luckfind.wgsl):
// 1 block = 128 threads = 128 independent walkers. Each step:
//   - Every thread does P += step_point (Jacobian mixed addition)
//   - Block batch-inverts 128 Z values (Montgomery trick: 1 fe_inv + 381 muls)
//   - Each thread converts to affine, hashes, compares against candidates
//
// Data layout matches the Rust GpuState / GpuConfig / GpuMatchOutput structs
// (see src/gpu/mod.rs) so the CPU side can share bytemuck casting.

// Compute-only kernel: uses the minimal device header rather than
// <cuda_runtime.h> so PTX generation does not pull in the host C++ standard
// library (which breaks CUDA 11.x against very new MSVC versions).
#include "minimal_cuda.h"

// =============================================================================
// Type definitions (match Rust bytemuck layout exactly)
// =============================================================================

typedef unsigned int u32;
typedef unsigned long long u64;
typedef int i32;

// Field element: 8 x u32 limbs, little-endian
typedef struct { u32 v[8]; } fe_t;

// Config: 16 bytes, matches GpuConfig
typedef struct {
    u32 num_threads;
    u32 steps_per_call;
    u32 num_candidates;
    u32 stride;
} GpuConfig;

// State: 192 bytes, matches GpuState
typedef struct {
    fe_t x, y, z;        // Jacobian point
    fe_t scalar;         // Private key (LE limbs)
    fe_t step_px, step_py; // Affine step point
} GpuState;

// Match output: 128 bytes, matches GpuMatchOutput
typedef struct {
    fe_t scalar;
    fe_t pubkey_x;
    fe_t pubkey_y;
    u32 hash160[5];
    u32 candidate_index;
    u32 thread_id;
    u32 _padding;
} MatchOutput;

// Atomic increment via inline PTX.  We deliberately do NOT call the builtin
// atomicAdd here: with the minimal headers (build.rs stub cuda_runtime.h) the
// builtin is never recognized by name, so nvcc emits a call to an extern
// function that the driver JIT cannot resolve.  Inline PTX lowers directly to
// the atom.add.u32 instruction on every arch that supports it (sm_1x+).
__device__ __forceinline__ u32 lucky_atomic_inc(u32* counter) {
    u32 result;
    asm("atom.add.u32 %0, [%1], 1;" : "=r"(result) : "l"(counter) : "memory");
    return result;
}

// =============================================================================
// Field arithmetic: secp256k1 prime p = 2^256 - 2^32 - 977
// =============================================================================

#define P0 0xFFFFFC2Fu
#define P1 0xFFFFFFFEu
#define P2 0xFFFFFFFFu
#define P3 0xFFFFFFFFu
#define P4 0xFFFFFFFFu
#define P5 0xFFFFFFFFu
#define P6 0xFFFFFFFFu
#define P7 0xFFFFFFFFu

__device__ __forceinline__ u32 adc(u32 a, u32 b, u32 carry, u32* carry_out) {
    u64 sum = (u64)a + (u64)b + (u64)carry;
    *carry_out = (u32)(sum >> 32);
    return (u32)sum;
}

__device__ __forceinline__ u32 sbb(u32 a, u32 b, u32 borrow, u32* borrow_out) {
    u64 rhs = (u64)b + (u64)borrow;
    u64 diff;
    if (a >= rhs) {
        diff = (u64)a - rhs;
        *borrow_out = 0;
    } else {
        diff = (u64)a + 0x100000000ULL - rhs;
        *borrow_out = 1;
    }
    return (u32)diff;
}

__device__ void fe_add(const fe_t* a, const fe_t* b, fe_t* c) {
    u32 carry = 0;
    for (int i = 0; i < 8; i++) {
        u64 sum = (u64)a->v[i] + (u64)b->v[i] + (u64)carry;
        c->v[i] = (u32)sum;
        carry = (u32)(sum >> 32);
    }
    // If overflow, reduce by adding 2^32 + 977 (because 2^256 ≡ 2^32 + 977 mod P)
    if (carry) {
        carry = 0;
        u64 sum = (u64)c->v[0] + 977ULL;
        c->v[0] = (u32)sum;
        carry = (u32)(sum >> 32);
        sum = (u64)c->v[1] + 1ULL + (u64)carry;
        c->v[1] = (u32)sum;
        carry = (u32)(sum >> 32);
        for (int i = 2; i < 8; i++) {
            sum = (u64)c->v[i] + (u64)carry;
            c->v[i] = (u32)sum;
            carry = (u32)(sum >> 32);
        }
    }
}

__device__ void fe_sub(const fe_t* a, const fe_t* b, fe_t* c) {
    u32 borrow = 0;
    for (int i = 0; i < 8; i++) {
        c->v[i] = sbb(a->v[i], b->v[i], borrow, &borrow);
    }
    // If borrow, add p back
    if (borrow) {
        u32 carry = 0;
        u64 sum = (u64)c->v[0] + (u64)P0;
        c->v[0] = (u32)sum;
        carry = (u32)(sum >> 32);
        sum = (u64)c->v[1] + (u64)P1 + (u64)carry;
        c->v[1] = (u32)sum;
        carry = (u32)(sum >> 32);
        for (int i = 2; i < 8; i++) {
            sum = (u64)c->v[i] + (u64)(P2) + (u64)carry;
            c->v[i] = (u32)sum;
            carry = (u32)(sum >> 32);
        }
    }
}

__device__ void fe_double_inplace(fe_t* a) {
    fe_t tmp;
    fe_add(a, a, &tmp);
    *a = tmp;
}

// 32x32 -> 64 bit multiplication
__device__ __forceinline__ void mul32(u32 a, u32 b, u32* lo, u32* hi) {
    u64 prod = (u64)a * (u64)b;
    *lo = (u32)prod;
    *hi = (u32)(prod >> 32);
}

// Multiply by 977 (secp256k1 reduction constant)
__device__ __forceinline__ void mul_977(u32 h, u32* lo, u32* hi) {
    u64 prod = (u64)h * 977ULL;
    *lo = (u32)prod;
    *hi = (u32)(prod >> 32);
}

__device__ void fe_mul(const fe_t* a, const fe_t* b, fe_t* result) {
    // Full schoolbook multiplication to get 16 limbs (512-bit product)
    u64 p[16] = {0};

    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        for (int j = 0; j < 8; j++) {
            u64 prod = (u64)a->v[i] * (u64)b->v[j] + p[i + j] + carry;
            p[i + j] = (u32)prod;
            carry = prod >> 32;
        }
        p[i + 8] = (u32)carry;
    }

    // Reduction using 2^256 ≡ 2^32 + 977 (mod p).
    // For a high limb h = p[idx] (idx >= 8):
    //   h·2^(32·idx) = h·(2^(32·(idx-8)))·2^256 ≡ h·2^(32·(idx-7)) + h·977·2^(32·(idx-8))
    // Fold h·977 into p[idx-8]/p[idx-7], add h into p[idx-7], then PROPAGATE
    // the carry fully up through p[idx-1] into p[idx] (which is re-folded by a
    // later iteration or the p8 loop).  Mirrors field.wgsl exactly — an earlier
    // in-place reduction dropped carries for the top limbs, producing wrong
    // results whenever any Jacobian z accumulated a >1-limb value.
    u32 mul_lo, mul_hi;
    for (int idx = 15; idx >= 9; idx--) {
        u32 h = (u32)p[idx];
        if (h == 0) continue;
        p[idx] = 0;

        mul_977(h, &mul_lo, &mul_hi);
        u64 sum = (u64)p[idx - 8] + (u64)mul_lo;
        p[idx - 8] = (u32)sum;
        u32 c = (u32)(sum >> 32);

        sum = (u64)p[idx - 7] + (u64)mul_hi + (u64)c;
        p[idx - 7] = (u32)sum;
        c = (u32)(sum >> 32);

        sum = (u64)p[idx - 7] + (u64)h;
        p[idx - 7] = (u32)sum;
        c += (u32)(sum >> 32);

        for (int k = idx - 6; k <= idx - 1; k++) {
            sum = (u64)p[k] + (u64)c;
            p[k] = (u32)sum;
            c = (u32)(sum >> 32);
        }
        p[idx] = (u32)c;
    }

    // Fold p[8]: one pass can regenerate a carry into p8 (p0..p7 within 2^64 of
    // overflow), so loop until it vanishes (≤3 iterations in practice).
    while (p[8] != 0) {
        u32 h = (u32)p[8];
        p[8] = 0;

        mul_977(h, &mul_lo, &mul_hi);
        u64 sum = (u64)p[0] + (u64)mul_lo;
        p[0] = (u32)sum;
        u32 c = (u32)(sum >> 32);

        sum = (u64)p[1] + (u64)mul_hi + (u64)c;
        p[1] = (u32)sum;
        c = (u32)(sum >> 32);

        sum = (u64)p[1] + (u64)h;
        p[1] = (u32)sum;
        c += (u32)(sum >> 32);

        for (int k = 2; k < 8; k++) {
            sum = (u64)p[k] + (u64)c;
            p[k] = (u32)sum;
            c = (u32)(sum >> 32);
        }
        p[8] = (u32)c;
    }

    for (int i = 0; i < 8; i++) result->v[i] = (u32)p[i];
}

__device__ void fe_square(const fe_t* a, fe_t* result) {
    // Square is just mul(a, a)
    fe_mul(a, a, result);
}

__device__ void fe_inv(const fe_t* a, fe_t* result) {
    // Fermat's little theorem: a^(-1) = a^(p-2) mod p
    // Addition chain from libsecp256k1 (255 squarings + 15 multiplications)
    fe_t a2, x2, x3, x11, x22, x44, x88, t;

    fe_square(a, &a2);              // a^2
    fe_mul(&a2, a, &x2);           // a^3

    fe_square(&x2, &t);
    fe_mul(&t, a, &x3);            // a^7

    t = x3;
    for (int i = 0; i < 3; i++) fe_square(&t, &t);
    fe_mul(&t, &x3, &t);           // a^(2^6-1)
    for (int i = 0; i < 3; i++) fe_square(&t, &t);
    fe_mul(&t, &x3, &t);           // a^(2^9-1)
    for (int i = 0; i < 2; i++) fe_square(&t, &t);
    fe_mul(&t, &x2, &x11);         // a^(2^11-1)

    t = x11;
    for (int i = 0; i < 11; i++) fe_square(&t, &t);
    fe_mul(&t, &x11, &x22);        // a^(2^22-1)

    t = x22;
    for (int i = 0; i < 22; i++) fe_square(&t, &t);
    fe_mul(&t, &x22, &x44);        // a^(2^44-1)

    t = x44;
    for (int i = 0; i < 44; i++) fe_square(&t, &t);
    fe_mul(&t, &x44, &x88);        // a^(2^88-1)

    // Assembly
    t = x88;
    for (int i = 0; i < 88; i++) fe_square(&t, &t);
    fe_mul(&t, &x88, &t);          // a^(2^176-1)
    for (int i = 0; i < 44; i++) fe_square(&t, &t);
    fe_mul(&t, &x44, &t);          // a^(2^220-1)
    for (int i = 0; i < 3; i++) fe_square(&t, &t);
    fe_mul(&t, &x3, &t);           // a^(2^223-1)

    for (int i = 0; i < 23; i++) fe_square(&t, &t);
    fe_mul(&t, &x22, &t);

    // bits 9-0 = 0000_10_11_01
    fe_square(&t, &t);
    fe_square(&t, &t);
    fe_square(&t, &t);
    fe_square(&t, &t);  // 4 zeros
    fe_square(&t, &t);
    fe_square(&t, &t);
    fe_mul(&t, &a2, &t);  // window 10 = a^2
    fe_square(&t, &t);
    fe_square(&t, &t);
    fe_mul(&t, &x2, &t);  // window 11 = a^3
    fe_square(&t, &t);
    fe_square(&t, &t);
    fe_mul(&t, a, &t);    // window 01 = a^1

    *result = t;
}

__device__ __forceinline__ bool fe_is_zero(const fe_t* a) {
    return (a->v[0] | a->v[1] | a->v[2] | a->v[3] | a->v[4] | a->v[5] | a->v[6] | a->v[7]) == 0;
}

// =============================================================================
// Scalar addition (256-bit, no mod-p reduction)
// =============================================================================

__device__ void scalar_add_256(const fe_t* a, const fe_t* b, fe_t* c) {
    u64 carry = 0;
    for (int i = 0; i < 8; i++) {
        u64 sum = (u64)a->v[i] + (u64)b->v[i] + carry;
        c->v[i] = (u32)sum;
        carry = sum >> 32;
    }
}

// =============================================================================
// Jacobian point operations (secp256k1: y² = x³ + 7)
// =============================================================================

typedef struct { fe_t x, y, z; } JacobianPoint;

__device__ void jac_add_affine(const JacobianPoint* p, const fe_t* qx, const fe_t* qy, JacobianPoint* r) {
    fe_t z1z1, z1z1z1, u2, s2, h;

    fe_square(&p->z, &z1z1);           // Z1²
    fe_mul(&z1z1, &p->z, &z1z1z1);     // Z1³
    fe_mul(qx, &z1z1, &u2);            // U2 = X2 * Z1²
    fe_mul(qy, &z1z1z1, &s2);          // S2 = Y2 * Z1³
    fe_sub(&u2, &p->x, &h);            // H = U2 - X1

    if (fe_is_zero(&h)) {
        fe_t s_diff;
        fe_sub(&s2, &p->y, &s_diff);
        if (fe_is_zero(&s_diff)) {
            // P == Q: double (not expected in our use case)
            fe_t a, b, c, d, e, f;
            fe_square(&p->x, &a);
            fe_square(&p->y, &b);
            fe_square(&b, &c);
            fe_t xpb, xpb2, d_inner;
            fe_add(&p->x, &b, &xpb);
            fe_square(&xpb, &xpb2);
            fe_sub(&xpb2, &a, &d_inner);
            fe_sub(&d_inner, &c, &d_inner);
            fe_double_inplace(&d_inner);
            d = d_inner;
            fe_t aa;
            fe_add(&a, &a, &aa);
            fe_add(&aa, &a, &e);
            fe_square(&e, &f);
            fe_t dd = d;
            fe_double_inplace(&dd);
            fe_sub(&f, &dd, &r->x);
            // Y3 = M·(S - X3) - 8·Y1⁴.  Compute (S - X3) FIRST, then scale by M
            // — the earlier code computed M·S - X3, which is wrong (X3 is not
            // M·X3).  This branch only fires when Q == P (H = 0), i.e. walker
            // at k=1 doubling G with stride-1 (lottery) steps, which the tests
            // exercise but real puzzle chunks never hit (P == Q needs
            // start+i == stride, impossible at 2^70+ magnitudes).
            fe_t v;
            fe_t sx;
            fe_sub(&d, &r->x, &sx);  // sx = S - X3
            fe_mul(&e, &sx, &v);     // v = M·(S - X3)
            fe_t eight_c = c;
            fe_double_inplace(&eight_c);
            fe_double_inplace(&eight_c);
            fe_double_inplace(&eight_c);
            fe_sub(&v, &eight_c, &r->y);
            fe_t y1z1;
            fe_mul(&p->y, &p->z, &y1z1);
            fe_double_inplace(&y1z1);
            r->z = y1z1;
        } else {
            // P == -Q: infinity
            r->x.v[0] = 1; r->x.v[1] = 0; r->x.v[2] = 0; r->x.v[3] = 0;
            r->x.v[4] = 0; r->x.v[5] = 0; r->x.v[6] = 0; r->x.v[7] = 0;
            r->y.v[0] = 1; r->y.v[1] = 0; r->y.v[2] = 0; r->y.v[3] = 0;
            r->y.v[4] = 0; r->y.v[5] = 0; r->y.v[6] = 0; r->y.v[7] = 0;
            r->z.v[0] = 0; r->z.v[1] = 0; r->z.v[2] = 0; r->z.v[3] = 0;
            r->z.v[4] = 0; r->z.v[5] = 0; r->z.v[6] = 0; r->z.v[7] = 0;
        }
        return;
    }

    fe_t r_val, hh, i, j, v;
    fe_t s2my;
    fe_sub(&s2, &p->y, &s2my);
    fe_double_inplace(&s2my);
    r_val = s2my;
    fe_square(&h, &hh);            // hh = H²  (kept for Z3 = ... - H²)
    fe_add(&hh, &hh, &i);          // i = 2·hh
    fe_double_inplace(&i);         // i = 4·H²
    fe_mul(&h, &i, &j);
    fe_mul(&p->x, &i, &v);

    fe_t r2, vv = v;
    fe_double_inplace(&vv);
    fe_square(&r_val, &r2);
    fe_sub(&r2, &j, &r->x);
    fe_sub(&r->x, &vv, &r->x);

    fe_t vmx, y1j;
    fe_sub(&v, &r->x, &vmx);
    fe_mul(&r_val, &vmx, &r->y);
    fe_mul(&p->y, &j, &y1j);
    fe_double_inplace(&y1j);
    fe_sub(&r->y, &y1j, &r->y);

    fe_t z1ph, z1ph2;
    fe_add(&p->z, &h, &z1ph);
    fe_square(&z1ph, &z1ph2);
    fe_sub(&z1ph2, &z1z1, &r->z);
    fe_sub(&r->z, &hh, &r->z);     // subtract H² (hh is still H²)
}

// =============================================================================
// SHA-256 (for 33-byte compressed pubkey)
// =============================================================================

__device__ __forceinline__ u32 rotr(u32 x, u32 n) {
    return (x >> n) | (x << (32u - n));
}
__device__ __forceinline__ u32 ch(u32 x, u32 y, u32 z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ u32 maj(u32 x, u32 y, u32 z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ u32 sigma0(u32 x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
__device__ __forceinline__ u32 sigma1(u32 x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
__device__ __forceinline__ u32 gamma0(u32 x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
__device__ __forceinline__ u32 gamma1(u32 x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

__constant__ static u32 K[64] = {
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
};

__device__ void sha256_33bytes(const u32 data[9], u32 out[8]) {
    u32 state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                     0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    u32 w[64];
    // First 8 words
    for (int i = 0; i < 8; i++) w[i] = data[i];
    // 9th word: 1 byte of data + 0x80
    w[8] = (data[8] & 0xFF000000u) | 0x00800000u;
    for (int i = 9; i < 14; i++) w[i] = 0;
    w[14] = 0;
    w[15] = 264;  // 33 * 8 bits

    // Extend
    for (int i = 16; i < 64; i++) {
        w[i] = gamma1(w[i-2]) + w[i-7] + gamma0(w[i-15]) + w[i-16];
    }

    // Rounds
    u32 a = state[0], b = state[1], c = state[2], d = state[3];
    u32 e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        u32 t1 = h + sigma1(e) + ch(e, f, g) + K[i] + w[i];
        u32 t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    for (int i = 0; i < 8; i++) out[i] = state[i];
}

// =============================================================================
// RIPEMD-160 (for 32-byte SHA-256 output -> hash160)
// =============================================================================

__device__ __forceinline__ u32 rotl(u32 x, u32 n) { return (x << n) | (x >> (32u - n)); }

__device__ __forceinline__ u32 rmd_f(u32 x, u32 y, u32 z) { return x ^ y ^ z; }
__device__ __forceinline__ u32 rmd_g(u32 x, u32 y, u32 z) { return (x & y) | (~x & z); }
__device__ __forceinline__ u32 rmd_h(u32 x, u32 y, u32 z) { return (x | ~y) ^ z; }
__device__ __forceinline__ u32 rmd_i(u32 x, u32 y, u32 z) { return (x & z) | (y & ~z); }
__device__ __forceinline__ u32 rmd_j(u32 x, u32 y, u32 z) { return x ^ (y | ~z); }

__device__ void ripemd160_32bytes(const u32 data[8], u32 out[5]) {
    u32 state[5] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};

    // Convert from big-endian to little-endian and build message block
    u32 x[16];
    for (int i = 0; i < 8; i++) {
        u32 w = data[i];
        x[i] = ((w & 0xFFu) << 24) | ((w & 0xFF00u) << 8) |
               ((w >> 8) & 0xFF00u) | ((w >> 24) & 0xFFu);
    }
    x[8] = 0x00000080u;
    for (int i = 9; i < 14; i++) x[i] = 0;
    x[14] = 256;  // Length in bits
    x[15] = 0;

    // Constants
    u32 KL[5] = {0x00000000u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu};
    u32 KR[5] = {0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0x00000000u};

    u32 RL[80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                  7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
                  3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
                  1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
                  4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    u32 RR[80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
                  6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
                  15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
                  8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
                  12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    u32 SL[80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
                  7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
                  11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
                  11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
                  9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    u32 SR[80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
                  9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
                  9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
                  15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
                  8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};

    u32 al = state[0], bl = state[1], cl = state[2], dl = state[3], el = state[4];
    u32 ar = state[0], br = state[1], cr = state[2], dr = state[3], er = state[4];

    for (int i = 0; i < 80; i++) {
        u32 round = i / 16;
        u32 fl, fr, tl, tr;

        // Left path
        switch (round) {
            case 0: fl = rmd_f(bl, cl, dl); break;
            case 1: fl = rmd_g(bl, cl, dl); break;
            case 2: fl = rmd_h(bl, cl, dl); break;
            case 3: fl = rmd_i(bl, cl, dl); break;
            default: fl = rmd_j(bl, cl, dl); break;
        }
        tl = al + fl + x[RL[i]] + KL[round];
        tl = rotl(tl, SL[i]) + el;
        al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = tl;

        // Right path
        switch (round) {
            case 0: fr = rmd_j(br, cr, dr); break;
            case 1: fr = rmd_i(br, cr, dr); break;
            case 2: fr = rmd_h(br, cr, dr); break;
            case 3: fr = rmd_g(br, cr, dr); break;
            default: fr = rmd_f(br, cr, dr); break;
        }
        tr = ar + fr + x[RR[i]] + KR[round];
        tr = rotl(tr, SR[i]) + er;
        ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = tr;
    }

    u32 t = state[1] + cl + dr;
    state[1] = state[2] + dl + er;
    state[2] = state[3] + el + ar;
    state[3] = state[4] + al + br;
    state[4] = state[0] + bl + cr;
    state[0] = t;

    // Convert from little-endian to big-endian
    for (int i = 0; i < 5; i++) {
        u32 w = state[i];
        out[i] = ((w & 0xFFu) << 24) | ((w & 0xFF00u) << 8) |
                 ((w >> 8) & 0xFF00u) | ((w >> 24) & 0xFFu);
    }
}

// =============================================================================
// Helper: limbs (LE) -> big-endian words for SHA256
// =============================================================================

__device__ void limbs_to_be_words(const fe_t* limbs, u32 be[8]) {
    // Limbs are numeric u32 values in base-2^32 LE order (limbs[0] least
    // significant).  Reversing the index yields the SHA-256 big-endian words
    // directly — the word value already matches the message byte sequence,
    // so NO byte swap is performed here (mirrors luckfind.wgsl).
    for (int i = 0; i < 8; i++) {
        be[i] = limbs->v[7 - i];
    }
}

__device__ void compressed_pubkey_to_words(u32 prefix, const u32 x_be[8], u32 out[9]) {
    out[0] = (prefix << 24) | (x_be[0] >> 8);
    for (int i = 1; i < 8; i++) {
        out[i] = ((x_be[i - 1] & 0xFFu) << 24) | (x_be[i] >> 8);
    }
    out[8] = (x_be[7] & 0xFFu) << 24;
}

// =============================================================================
// Batch Montgomery inversion in shared memory
// =============================================================================

#define WORKGROUP_SIZE 128
#define MAX_MATCHES 256

extern "C" __global__ void luckfind_kernel(
    GpuConfig config,
    GpuState* states,
    const u32 candidates[78 * 5],  // 78 candidates, each 5 x u32 LE
    MatchOutput* matches,
    u32* match_count
) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= config.num_threads) return;

    u32 lid = threadIdx.x;  // 0..127 within block

    __shared__ fe_t shared_prod[WORKGROUP_SIZE];
    __shared__ fe_t shared_save[WORKGROUP_SIZE];

    // Stride as LE limbs
    fe_t stride_le = {{config.stride, 0, 0, 0, 0, 0, 0, 0}};

    GpuState s = states[tid];

    for (u32 step = 0; step < config.steps_per_call; step++) {
        // ── 1. Batch-invert 128 Z values ───────────────────────────────
        shared_prod[lid] = s.z;
        __syncthreads();

        // UP-SWEEP
        if ((lid & 1u) == 1u) {
            shared_save[lid >> 1] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 1], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 3u) == 3u) {
            shared_save[64 + (lid >> 2)] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 2], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 7u) == 7u) {
            shared_save[96 + (lid >> 3)] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 4], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 15u) == 15u) {
            shared_save[112 + (lid >> 4)] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 8], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 31u) == 31u) {
            shared_save[120 + (lid >> 5)] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 16], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 63u) == 63u) {
            shared_save[124 + (lid >> 6)] = shared_prod[lid];
            fe_mul(&shared_prod[lid - 32], &shared_prod[lid], &shared_prod[lid]);
        }
        __syncthreads();
        if (lid == 127u) {
            shared_save[126] = shared_prod[127];
            fe_mul(&shared_prod[63], &shared_prod[127], &shared_prod[127]);
        }
        __syncthreads();

        // INVERT root
        if (lid == 0u) {
            fe_inv(&shared_prod[127], &shared_prod[127]);
        }
        __syncthreads();

        // DOWN-SWEEP
        if (lid == 127u) {
            fe_t inv_p = shared_prod[127];
            fe_t left = shared_prod[63];
            fe_t right = shared_save[126];
            fe_mul(&inv_p, &right, &shared_prod[63]);
            fe_mul(&inv_p, &left, &shared_prod[127]);
        }
        __syncthreads();
        if ((lid & 63u) == 63u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 32];
            fe_t right = shared_save[124 + (lid >> 6)];
            fe_mul(&inv_p, &right, &shared_prod[lid - 32]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 31u) == 31u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 16];
            fe_t right = shared_save[120 + (lid >> 5)];
            fe_mul(&inv_p, &right, &shared_prod[lid - 16]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 15u) == 15u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 8];
            fe_t right = shared_save[112 + (lid >> 4)];
            fe_mul(&inv_p, &right, &shared_prod[lid - 8]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 7u) == 7u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 4];
            fe_t right = shared_save[96 + (lid >> 3)];
            fe_mul(&inv_p, &right, &shared_prod[lid - 4]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 3u) == 3u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 2];
            fe_t right = shared_save[64 + (lid >> 2)];
            fe_mul(&inv_p, &right, &shared_prod[lid - 2]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();
        if ((lid & 1u) == 1u) {
            fe_t inv_p = shared_prod[lid];
            fe_t left = shared_prod[lid - 1];
            fe_t right = shared_save[lid >> 1];
            fe_mul(&inv_p, &right, &shared_prod[lid - 1]);
            fe_mul(&inv_p, &left, &shared_prod[lid]);
        }
        __syncthreads();

        // ── 2. Affine conversion + hash ────────────────────────────────
        fe_t z_inv = shared_prod[lid];
        fe_t z2_inv, z3_inv, x_affine, y_affine;
        fe_square(&z_inv, &z2_inv);
        fe_mul(&z2_inv, &z_inv, &z3_inv);
        fe_mul(&s.x, &z2_inv, &x_affine);
        fe_mul(&s.y, &z3_inv, &y_affine);

        // Compressed pubkey prefix
        u32 prefix = ((x_affine.v[0] | y_affine.v[0]) & 1u) == 0 ? 0x02u : 0x03u;
        // Actually use y_affine parity
        prefix = (y_affine.v[0] & 1u) == 0 ? 0x02u : 0x03u;

        u32 x_be[8];
        limbs_to_be_words(&x_affine, x_be);
        u32 cpk[9];
        compressed_pubkey_to_words(prefix, x_be, cpk);

        // SHA256
        u32 sha_out[8];
        sha256_33bytes(cpk, sha_out);

        // RIPEMD160
        u32 h160_be[5];
        ripemd160_32bytes(sha_out, h160_be);

        // Convert to LE for comparison
        u32 h160_le[5];
        for (int i = 0; i < 5; i++) {
            u32 w = h160_be[i];
            h160_le[i] = ((w & 0xFFu) << 24) | ((w & 0xFF00u) << 8) |
                         ((w >> 8) & 0xFF00u) | ((w >> 24) & 0xFFu);
        }

        // Candidate match
        u32 m = 0xFFFFFFFFu;
        for (u32 i = 0; i < config.num_candidates; i++) {
            if (h160_le[0] == candidates[i * 5 + 0] &&
                h160_le[1] == candidates[i * 5 + 1] &&
                h160_le[2] == candidates[i * 5 + 2] &&
                h160_le[3] == candidates[i * 5 + 3] &&
                h160_le[4] == candidates[i * 5 + 4]) {
                m = i;
                break;
            }
        }

        if (m != 0xFFFFFFFFu) {
            u32 idx = lucky_atomic_inc(match_count);
            if (idx < MAX_MATCHES) {
                MatchOutput mo;
                mo.scalar = s.scalar;
                mo.pubkey_x = x_affine;
                mo.pubkey_y = y_affine;
                for (int i = 0; i < 5; i++) mo.hash160[i] = h160_be[i];
                mo.candidate_index = m;
                mo.thread_id = tid;
                mo._padding = 0;
                matches[idx] = mo;
            }
        }

        // ── 3. Advance to next key ──────────────────────────────────────
        JacobianPoint jp;
        jp.x = s.x; jp.y = s.y; jp.z = s.z;
        JacobianPoint result;
        jac_add_affine(&jp, &s.step_px, &s.step_py, &result);
        s.x = result.x; s.y = result.y; s.z = result.z;
        scalar_add_256(&s.scalar, &stride_le, &s.scalar);
    }

    states[tid] = s;
}
