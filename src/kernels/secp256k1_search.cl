// ============================================================================
//  secp256k1 GPU key-search kernel (OpenCL C)
//
//  Each global invocation `gid` computes:
//      d   = base + gid                       (256-bit scalar add)
//      Q   = d * G                            (Jacobian windowed scalar mult)
//      (x,y) = Q.to_affine()                  (one modular inverse)
//      comp / unc pubkey bytes + hash160
//      binary-search the sorted candidate table
//      on hit: atomic_inc a counter, write priv/comp/unc/wid
//
//  NOTE: hand-rolled field arithmetic. Validate against the CPU path on a
//  small range before trusting results in production.
// ============================================================================

// ---- secp256k1 domain parameters ------------------------------------------
// p = 2^256 - 2^32 - 977
// big-endian: FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
__constant uint P[8]    = {0xFFFFFC2F,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,
                           0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFE};
// p - 2 (for Fermat inversion)
__constant uint PM2[8]   = {0xFFFFFC2D,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,
                           0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFE};

// ---- field element: 8 x uint32, little-endian limbs ------------------------
typedef uint fe[8];

typedef struct { fe X; fe Y; fe Z; } jp;   // Jacobian point

// ---------------------------------------------------------------------------
// field helpers
// ---------------------------------------------------------------------------
static int fe_ge(const fe a, const fe b){
    for (int i = 7; i >= 0; i--){
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}
static int fe_is_zero(const fe a){
    uint s = 0; for (int i = 0; i < 8; i++) s |= a[i];
    return s == 0;
}
static void fe_copy(fe r, const fe a){ for (int i = 0; i < 8; i++) r[i] = a[i]; }
static void fe_zero(fe r){ for (int i = 0; i < 8; i++) r[i] = 0; }
static void fe_set_u32(fe r, uint v){ r[0] = v; for (int i = 1; i < 8; i++) r[i] = 0; }

static void fe_sub(fe r, const fe a, const fe b){
    int borrow = 0;
    for (int i = 0; i < 8; i++){
        int32_t s = (int32_t)a[i] - (int32_t)b[i] - borrow;
        if (s < 0){ s += (int32_t)0x100000000ULL; borrow = 1; } else borrow = 0;
        r[i] = (uint)s;
    }
    if (borrow){ // result negative -> add p
        uint c = 0;
        for (int i = 0; i < 8; i++){ uint s = (uint)r[i] + P[i] + c; c = (s < r[i]) ? 1u : 0u; r[i] = s; }
    }
}
static void fe_add(fe r, const fe a, const fe b){
    uint c = 0;
    for (int i = 0; i < 8; i++){ uint s = (uint)a[i] + b[i] + c; c = (s < a[i]) ? 1u : 0u; r[i] = s; }
    if (c || fe_ge(r, P)) fe_sub(r, r, P);
}

// reduce a 512-bit little-endian value (16 limbs) mod p
static void reduce512(fe r, uint w[16]){
    // value = low(0..7) + high(8..15)*2^256  ;  2^256 = 2^32 + 977 (mod p)
    ulong t[17]; for (int i = 0; i < 17; i++) t[i] = 0;
    for (int i = 0; i < 8;  i++) t[i]     = w[i];                 // low
    for (int i = 0; i < 8;  i++) t[9+i]  += (ulong)w[8+i];        // high * 2^32  (limb shift)
    for (int i = 0; i < 8;  i++) t[8+i]  += (ulong)w[8+i] * 977ULL; // high * 977
    ulong carry = 0; uint v[17];
    for (int i = 0; i < 17; i++){ ulong s = t[i] + carry; v[i] = (uint)(s & 0xFFFFFFFFULL); carry = s >> 32; }

    // second fold (high part is now tiny)
    ulong u[18]; for (int i = 0; i < 18; i++) u[i] = 0;
    for (int i = 0; i < 8;  i++) u[i]     = v[i];
    for (int i = 0; i < 9;  i++) u[9+i]  += (ulong)v[8+i];
    for (int i = 0; i < 9;  i++) u[8+i]  += (ulong)v[8+i] * 977ULL;
    ulong c2 = 0; uint z[18];
    for (int i = 0; i < 17; i++){ ulong s = u[i] + c2; z[i] = (uint)(s & 0xFFFFFFFFULL); c2 = s >> 32; }
    z[17] = (uint)c2;

    for (int i = 0; i < 8; i++) r[i] = z[i];
    while (fe_ge(r, P)) fe_sub(r, r, P);
}

static void fe_mul(fe r, const fe a, const fe b){
    ulong t[16]; for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++){
        ulong carry = 0;
        for (int j = 0; j < 8; j++){
            ulong prod = (ulong)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFFULL;
            carry    = prod >> 32;
        }
        t[i + 8] += carry;
    }
    ulong carry = 0; uint w[16];
    for (int i = 0; i < 16; i++){ ulong s = t[i] + carry; w[i] = (uint)(s & 0xFFFFFFFFULL); carry = s >> 32; }
    reduce512(r, w);
}
static void fe_sqr(fe r, const fe a){ fe_mul(r, a, a); }

// a^(p-2) via binary exponentiation (Fermat)
static void fe_inv(fe out, const fe a){
    fe r; fe_set_u32(r, 1);
    fe base; fe_copy(base, a);
    for (int i = 7; i >= 0; i--){
        uint limb = PM2[i];
        for (int b = 31; b >= 0; b--){
            fe_sqr(r, r);
            if ((limb >> b) & 1u) fe_mul(r, r, base);
        }
    }
    fe_copy(out, r);
}

// ---------------------------------------------------------------------------
// Jacobian point ops (a = 0 for secp256k1)
// ---------------------------------------------------------------------------
static void jac_double(jp* R, const jp* P){
    fe a, b, c, d, e, t1, c2;
    fe_sqr(a, P->X);            // A = X^2
    fe_sqr(b, P->Y);            // B = Y^2
    fe_sqr(c, b);               // C = B^2
    fe_add(d, P->X, b); fe_sqr(d, d); fe_sub(d, d, a); fe_sub(d, d, c); fe_add(d, d, d); // D = 2((X+B)^2-A-C)
    fe_add(e, a, a); fe_add(e, e, a);   // E = 3A
    fe_sqr(e, e);                       // F = E^2
    fe_sub(R->X, e, d); fe_sub(R->X, R->X, d);   // X3 = F - 2D
    fe_sub(t1, d, R->X); fe_mul(R->Y, t1, e);     // E*(D-X3)
    fe_add(c2, c, c); fe_add(c2, c2, c2);          // 4C
    fe_sub(R->Y, R->Y, c2); fe_sub(R->Y, R->Y, c2); // - 8C
    fe_mul(R->Z, P->Y, P->Z); fe_add(R->Z, R->Z, R->Z); // Z3 = 2*Y*Z
}

static void jac_add(jp* R, const jp* P, const jp* Q){
    if (fe_is_zero(P->Z)){ fe_copy(R->X,Q->X); fe_copy(R->Y,Q->Y); fe_copy(R->Z,Q->Z); return; }
    if (fe_is_zero(Q->Z)){ fe_copy(R->X,P->X); fe_copy(R->Y,P->Y); fe_copy(R->Z,P->Z); return; }
    fe z1z1, z2z2, u1, u2, s1, s2, h, r, hh, ts;
    fe_sqr(z1z1, P->Z);
    fe_sqr(z2z2, Q->Z);
    fe_mul(u1, P->X, z2z2);
    fe_mul(u2, Q->X, z1z1);
    fe_mul(s1, P->Y, z2z2); fe_mul(s1, s1, Q->Z);
    fe_mul(s2, Q->Y, z1z1); fe_mul(s2, s2, P->Z);
    fe_sub(h, u2, u1);
    if (fe_is_zero(h)){
        fe_sub(ts, s2, s1);
        if (fe_is_zero(ts)){ jac_double(R, P); return; }
        fe_zero(R->X); fe_zero(R->Y); fe_zero(R->Z); return; // P == -Q
    }
    fe_sub(r, s2, s1);
    fe_sqr(hh, h);
    fe_mul(u1, u1, hh);
    fe_mul(hh, hh, h);
    fe_mul(s2, s1, hh);
    fe_sqr(R->X, r);
    fe_sub(R->X, R->X, hh);
    fe_sub(R->X, R->X, u1);
    fe_sub(R->X, R->X, u1);
    fe_sub(R->Y, u1, R->X);
    fe_mul(R->Y, R->Y, r);
    fe_sub(R->Y, R->Y, s2);
    fe_sub(R->Y, R->Y, s2);
    fe_mul(R->Z, P->Z, Q->Z);
    fe_mul(R->Z, R->Z, h);
}

// ---------------------------------------------------------------------------
// scalar decode / table read / affine conversion
// ---------------------------------------------------------------------------
static void bytes_to_fe(fe r, const uchar* b){ // 32 bytes big-endian -> LE limbs
    for (int i = 0; i < 8; i++){
        int bi = 4 * i;
        r[7 - i] = ((uint)b[bi] << 24) | ((uint)b[bi+1] << 16) | ((uint)b[bi+2] << 8) | b[bi+3];
    }
}
static void fe_to_bytes(uchar* out, const fe x){ // LE limbs -> 32 bytes big-endian
    for (int i = 0; i < 8; i++){
        uint limb = x[7 - i];
        out[4*i]   = (uchar)(limb >> 24);
        out[4*i+1] = (uchar)(limb >> 16);
        out[4*i+2] = (uchar)(limb >> 8);
        out[4*i+3] = (uchar)limb;
    }
}
static void read_table(jp* T, __global const uchar* table, int idx){
    const uchar* e = table + idx * 96;
    bytes_to_fe(T->X, e);
    bytes_to_fe(T->Y, e + 32);
    bytes_to_fe(T->Z, e + 64);
}
static void jac_to_affine(fe* x, fe* y, const jp* P){
    if (fe_is_zero(P->Z)){ fe_zero(*x); fe_zero(*y); return; }
    fe z1, z2, z3;
    fe_inv(z1, P->Z);
    fe_sqr(z2, z1);
    fe_mul(*x, P->X, z2);
    fe_mul(z3, z2, z1);
    fe_mul(*y, P->Y, z3);
}

// windowed (4-bit) scalar multiplication: R = d * G
static void scalar_mult(jp* R, __global const uchar* table, const fe* d){
    fe_zero(R->X); fe_zero(R->Y); fe_zero(R->Z); // infinity
    for (int win = 63; win >= 0; win--){
        int bitpos = win * 4;
        int limb   = bitpos / 32;
        int shift  = bitpos % 32;
        uint digit = (d[limb] >> shift) & 0xF;
        for (int k = 0; k < 4; k++){
            if (!fe_is_zero(R->Z)) jac_double(R, R);
        }
        if (digit != 0){
            jp T; read_table(&T, table, digit - 1);
            jac_add(R, R, &T);
        }
    }
}

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------
#define ROR(x,n) ((x >> n) | (x << (32 - n)))
static uint S0(uint x){ return ROR(x,2)  ^ ROR(x,13) ^ ROR(x,22); }
static uint S1(uint x){ return ROR(x,6)  ^ ROR(x,11) ^ ROR(x,25); }
static uint s0(uint x){ return ROR(x,7)  ^ ROR(x,18) ^ (x >> 3); }
static uint s1(uint x){ return ROR(x,17) ^ ROR(x,19) ^ (x >> 10); }
static uint Ch(uint x,uint y,uint z){ return (x & y) ^ (~x & z); }
static uint Maj(uint x,uint y,uint z){ return (x & y) ^ (x & z) ^ (y & z); }

static void sha256(const uchar* data, uint len, uchar out[32]){
    uint k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
    uint h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                 0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};

    ulong bitlen = (ulong)len * 8UL;
    uint total   = len + 1 + ((56u - (len % 64u)) % 64u) + 8u;
    uint blocks  = total / 64u;
    uint w[64];

    for (uint block = 0; block < blocks; block++){
        uchar buf[64];
        for (uint i = 0; i < 64; i++){
            uint pos = block * 64u + i;
            uchar v = 0;
            if (pos < len) v = data[pos];
            else if (pos == len) v = 0x80;
            buf[i] = v;
        }
        if (block == blocks - 1u){
            for (int i = 0; i < 8; i++)
                buf[56 + i] = (uchar)(bitlen >> (56 - 8 * i));
        }
        for (int i = 0; i < 16; i++)
            w[i] = ((uint)buf[4*i]<<24)|((uint)buf[4*i+1]<<16)|((uint)buf[4*i+2]<<8)|buf[4*i+3];
        for (int i = 16; i < 64; i++)
            w[i] = s1(w[i-2]) + w[i-7] + s0(w[i-15]) + w[i-16];

        uint a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++){
            uint t1 = hh + S1(e) + Ch(e,f,g) + k[i] + w[i];
            uint t2 = S0(a) + Maj(a,b,c);
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    for (int i = 0; i < 8; i++){
        out[4*i]   = (uchar)(h[i] >> 24);
        out[4*i+1] = (uchar)(h[i] >> 16);
        out[4*i+2] = (uchar)(h[i] >> 8);
        out[4*i+3] = (uchar)h[i];
    }
}

// ---------------------------------------------------------------------------
// RIPEMD-160
// ---------------------------------------------------------------------------
#define RROL(x,n) ((x << n) | (x >> (32 - n)))
static uint f1(uint x,uint y,uint z){ return x ^ y ^ z; }
static uint f2(uint x,uint y,uint z){ return (x & y) | (~x & z); }
static uint f3(uint x,uint y,uint z){ return (x | ~y) ^ z; }
static uint f4(uint x,uint y,uint z){ return (x & z) | (y & ~z); }
static uint f5(uint x,uint y,uint z){ return x ^ (y | ~z); }

static void ripemd160(const uchar* data, uint len, uchar out[20]){
    int r1[16]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    int r2[16]={7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8};
    int r3[16]={3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12};
    int r4[16]={1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2};
    int r5[16]={4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    int rp1[16]={5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12};
    int rp2[16]={6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2};
    int rp3[16]={15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13};
    int rp4[16]={8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14};
    int rp5[16]={12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    int s1[16]={11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8};
    int s2[16]={7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12};
    int s3[16]={11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5};
    int s4[16]={11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12};
    int s5[16]={9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    int sp1[16]={8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6};
    int sp2[16]={9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11};
    int sp3[16]={9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5};
    int sp4[16]={15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8};
    int sp5[16]={8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    uint kl[5]={0,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E};
    uint kr[5]={0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0};

    uint h[5]={0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};

    uint total = len + 1 + ((56u - (len % 64u)) % 64u) + 8u;
    uint blocks = total / 64u;
    for (uint block = 0; block < blocks; block++){
        uchar buf[64];
        for (uint i = 0; i < 64; i++){
            uint pos = block*64u + i;
            uchar v = 0;
            if (pos < len) v = data[pos];
            else if (pos == len) v = 0x80;
            buf[i] = v;
        }
        if (block == blocks - 1u){
            ulong bl = (ulong)len * 8UL;
            for (int i = 0; i < 8; i++) buf[56+i] = (uchar)(bl >> (56 - 8*i));
        }
        uint X[16];
        for (int i = 0; i < 16; i++)
            X[i] = ((uint)buf[4*i]<<24)|((uint)buf[4*i+1]<<16)|((uint)buf[4*i+2]<<8)|buf[4*i+3];

        uint al=h[0],bl=h[1],cl=h[2],dl=h[3],el=h[4];
        uint ar=al,br=bl,cr=cl,dr=dl,er=el;
        uint t;
        // --- left stream ---
        for (int i=0;i<16;i++){ t=RROL(al+f1(bl,cl,dl)+X[r1[i]]+kl[0],s1[i])+el; al=el;el=dl;dl=cl;cl=bl;bl=t; }
        for (int i=0;i<16;i++){ t=RROL(al+f2(bl,cl,dl)+X[r2[i]]+kl[1],s2[i])+el; al=el;el=dl;dl=cl;cl=bl;bl=t; }
        for (int i=0;i<16;i++){ t=RROL(al+f3(bl,cl,dl)+X[r3[i]]+kl[2],s3[i])+el; al=el;el=dl;dl=cl;cl=bl;bl=t; }
        for (int i=0;i<16;i++){ t=RROL(al+f4(bl,cl,dl)+X[r4[i]]+kl[3],s4[i])+el; al=el;el=dl;dl=cl;cl=bl;bl=t; }
        for (int i=0;i<16;i++){ t=RROL(al+f5(bl,cl,dl)+X[r5[i]]+kl[4],s5[i])+el; al=el;el=dl;dl=cl;cl=bl;bl=t; }
        // --- right stream ---
        for (int i=0;i<16;i++){ t=RROL(ar+f1(br,cr,dr)+X[rp1[i]]+kr[0],sp1[i])+er; ar=er;er=dr;dr=cr;cr=br;br=t; }
        for (int i=0;i<16;i++){ t=RROL(ar+f2(br,cr,dr)+X[rp2[i]]+kr[1],sp2[i])+er; ar=er;er=dr;dr=cr;cr=br;br=t; }
        for (int i=0;i<16;i++){ t=RROL(ar+f3(br,cr,dr)+X[rp3[i]]+kr[2],sp3[i])+er; ar=er;er=dr;dr=cr;cr=br;br=t; }
        for (int i=0;i<16;i++){ t=RROL(ar+f4(br,cr,dr)+X[rp4[i]]+kr[3],sp4[i])+er; ar=er;er=dr;dr=cr;cr=br;br=t; }
        for (int i=0;i<16;i++){ t=RROL(ar+f5(br,cr,dr)+X[rp5[i]]+kr[4],sp5[i])+er; ar=er;er=dr;dr=cr;cr=br;br=t; }

        t = h[1] + cl + dr; h[1] = h[2] + dl + er; h[2] = h[3] + el + ar;
        h[3] = h[4] + al + br; h[4] = h[0] + bl + cr; h[0] = t;
    }
    for (int i = 0; i < 5; i++){
        out[4*i]   = (uchar)(h[i]);
        out[4*i+1] = (uchar)(h[i] >> 8);
        out[4*i+2] = (uchar)(h[i] >> 16);
        out[4*i+3] = (uchar)(h[i] >> 24);
    }
}

// ---------------------------------------------------------------------------
// candidate lookup (binary search over sorted 20-byte hashes)
// ---------------------------------------------------------------------------
static int binsearch(__global const uchar* targets, uint n, const uchar* key){
    int lo = 0, hi = (int)n - 1;
    while (lo <= hi){
        int mid = (lo + hi) >> 1;
        __global const uchar* p = targets + (ulong)mid * 20UL;
        int cmp = 0;
        for (int i = 0; i < 20; i++){
            if (p[i] < key[i]){ cmp = -1; break; }
            if (p[i] > key[i]){ cmp =  1; break; }
        }
        if (cmp == 0) return 1;
        if (cmp < 0) lo = mid + 1; else hi = mid - 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main kernel
// ---------------------------------------------------------------------------
__kernel void search(
    __global const uchar* table,    // 15 * 96 bytes (X,Y,Z per entry, BE)
    __global const uchar* targets,  // sorted 20-byte hashes, ascending
    uint ntargets,
    __global const uchar* base,     // 32 bytes, big-endian (secret_bytes order)
    uint batch,
    uint capacity,
    uint wid,
    __global uint* counter,         // atomic match counter
    __global uchar* out_priv,       // capacity * 32
    __global uchar* out_comp,       // capacity * 33
    __global uchar* out_unc,        // capacity * 65
    __global uint*  out_wid)        // capacity
{
    uint gid = get_global_id(0);
    if (gid >= batch) return;

    // d = base + gid  (base is big-endian; LSB lives in the last 4 bytes)
    uchar d_bytes[32];
    for (int i = 0; i < 32; i++) d_bytes[i] = base[i];
    ulong acc = ((ulong)d_bytes[28] << 24) | ((ulong)d_bytes[29] << 16) |
                ((ulong)d_bytes[30] << 8)  |  (ulong)d_bytes[31];
    acc += (ulong)gid;
    for (int i = 31; i >= 28; i--){ d_bytes[i] = (uchar)(acc & 0xFF); acc >>= 8; }
    if (acc > 0){ int i = 27; while (acc > 0 && i >= 0){ acc += d_bytes[i]; d_bytes[i] = (uchar)(acc & 0xFF); acc >>= 8; i--; } }

    fe d; bytes_to_fe(d, d_bytes);

    jp R; scalar_mult(&R, table, &d);
    fe X, Y; jac_to_affine(&X, &Y, &R);

    uchar xb[32], yb[32];
    fe_to_bytes(xb, X); fe_to_bytes(yb, Y);

    uchar comp[33]; comp[0] = (Y[0] & 1u) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) comp[i+1] = xb[i];
    uchar unc[65]; unc[0] = 0x04;
    for (int i = 0; i < 32; i++){ unc[1+i] = xb[i]; unc[33+i] = yb[i]; }

    uchar hc[20], tmp[32];
    sha256(comp, 33, tmp); ripemd160(tmp, 32, hc);
    if (binsearch(targets, ntargets, hc)){
        uint idx = atomic_inc(counter);
        if (idx < capacity){
            for (int i = 0; i < 32; i++) out_priv[idx*32 + i] = d_bytes[i];
            for (int i = 0; i < 33; i++) out_comp[idx*33 + i] = comp[i];
            for (int i = 0; i < 65; i++) out_unc [idx*65 + i] = unc[i];
            out_wid[idx] = wid;
        }
    }
    sha256(unc, 65, tmp); ripemd160(tmp, 32, hc);
    if (binsearch(targets, ntargets, hc)){
        uint idx = atomic_inc(counter);
        if (idx < capacity){
            for (int i = 0; i < 32; i++) out_priv[idx*32 + i] = d_bytes[i];
            for (int i = 0; i < 33; i++) out_comp[idx*33 + i] = comp[i];
            for (int i = 0; i < 65; i++) out_unc [idx*65 + i] = unc[i];
            out_wid[idx] = wid;
        }
    }
}
