// luckfind.wgsl — GPU kernel for sequential secp256k1 key scanning with batch inversion.
//
// 1 workgroup = 128 threads = 128 independent walkers. Each step:
//   - Every thread does P += G (Jacobian)
//   - Workgroup batch-inverts 128 Z values (Montgomery trick: 1 fe_inv + 381 muls)
//   - Each thread converts to affine, hashes, compares against candidates
//
// Per-step cost: ~12 (add) + 3 (amortized batch) + hash ≈ 15 field muls/point.
// Target: 80-150 Mkeys/s on M4 GPU with 100k parallel threads.

override WORKGROUP_SIZE: u32 = 128u;

fn jacobian(x: array<u32, 8>, y: array<u32, 8>, z: array<u32, 8>) -> JacobianPoint {
    var p: JacobianPoint; p.x = x; p.y = y; p.z = z; return p;
}

struct GpuConfig {
    num_threads: u32, steps_per_call: u32, num_candidates: u32, stride: u32,
}
struct GpuState {
    x: array<u32, 8>, y: array<u32, 8>, z: array<u32, 8>, scalar: array<u32, 8>,
    // Per-walker step point (affine, LE limbs).  Lottery mode: = G (stride 1).
    // Puzzle mode: = N*G (stride N) so walkers, seeded at start+i, interleave
    // across the chunk with zero overlap.
    step_px: array<u32, 8>, step_py: array<u32, 8>,
}
struct MatchOutput {
    scalar: array<u32, 8>, pubkey_x: array<u32, 8>, pubkey_y: array<u32, 8>,
    hash160: array<u32, 5>, candidate_index: u32, thread_id: u32, _padding: u32,
}

// Generator point G (affine), LE limbs.  Matches convert::be_bytes_to_limbs:
//   limbs[0] = lowest 32 bits of integer = BE bytes[28..32] as big-endian u32
const GX: array<u32, 8> = array<u32, 8>(
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
);
const GY: array<u32, 8> = array<u32, 8>(
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
);

// Batch-inversion workgroup scratch: 2 × 4 KB = 8 KB, well under Metal's
// 32 KB / workgroup cap.
var<workgroup> shared_prod: array<array<u32, 8>, 128>;
var<workgroup> shared_save: array<array<u32, 8>, 128>;

@group(0) @binding(0) var<uniform> config: GpuConfig;
@group(0) @binding(1) var<storage, read_write> states: array<GpuState>;
@group(0) @binding(2) var<storage, read> candidates: array<array<u32, 5>, 78>;
@group(0) @binding(3) var<storage, read_write> matches: array<MatchOutput>;
@group(0) @binding(4) var<storage, read_write> match_count: atomic<u32>;

fn limbs_to_be_words(limbs: array<u32, 8>) -> array<u32, 8> {
    var w: array<u32, 8>;
    for (var i = 0u; i < 8u; i++) { w[i] = limbs[7u - i]; }
    return w;
}

// Pack a compressed pubkey (1 prefix byte + 32 x-coordinate bytes) into the
// 9 big-endian u32 words that sha256_33bytes expects.  Bitcoin's canonical
// layout is [prefix, X0..X31] (prefix FIRST), so the prefix byte lands in the
// top byte of word[0] and the 33rd byte (X31) lands in the top byte of word[8].
// sha256_33bytes reads word[8] & 0xFF000000 for that last data byte, then pads.
fn compressed_pubkey_to_words(prefix: u32, x_be: array<u32, 8>) -> array<u32, 9> {
    var out: array<u32, 9>;
    out[0] = (prefix << 24u) | (x_be[0] >> 8u);
    for (var i = 1u; i < 8u; i++) {
        out[i] = ((x_be[i - 1u] & 0xFFu) << 24u) | (x_be[i] >> 8u);
    }
    out[8] = (x_be[7] & 0xFFu) << 24u;
    return out;
}

// kangaroo's ripemd160_32bytes returns hash160 packed as big-endian u32 words.
// The candidate buffer is filled on the CPU with little-endian u32 words
// (see tests/gpu_batch_inv.rs), so byte-swap each word before comparing.
fn hash160_be_to_le(h: Rmd160Hash) -> array<u32, 5> {
    var out: array<u32, 5>;
    for (var i = 0u; i < 5u; i = i + 1u) {
        let w = h.h[i];
        out[i] = ((w & 0xFFu) << 24u) | ((w & 0xFF00u) << 8u)
               | ((w >> 8u) & 0xFF00u) | ((w >> 24u) & 0xFFu);
    }
    return out;
}

fn candidate_match(h: array<u32, 5>) -> u32 {
    // Bound the scan by `num_candidates` (set to 78 for the full lottery set,
    // 1 for puzzle mode) so unused trailing slots — which read as zero and
    // would false-match an all-zero hash160 — are never examined.
    for (var i = 0u; i < config.num_candidates; i = i + 1u) {
        if (h[0]==candidates[i][0] && h[1]==candidates[i][1] && h[2]==candidates[i][2]
         && h[3]==candidates[i][3] && h[4]==candidates[i][4]) { return i; }
    }
    return 0xFFFFFFFFu;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= config.num_threads) { return; }
    let lid = gid.x & 127u;  // tid % 128

    // Declared outside the loop so the final writeback (last line) sees it.
    var s: GpuState;
    // Stride as an LE limb array for scalar_add_256.  `stride` fits in one limb
    // (it's either 1 for lottery or N=100000 for puzzle), so limb[0]=stride and
    // the high limbs are zero.
    let stride_le = array<u32,8>(config.stride, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    for (var step = 0u; step < config.steps_per_call; step = step + 1u) {
        // ── 1. read current state + batch-invert 128 Z values ───────────────
        // Hash-then-advance ordering: we check the key the walker is currently
        // ON, then stride past it.  This makes deterministic tiling gap-free
        // — dispatch d covers [start + d·S·N, start + (d+1)·S·N) contiguously.
        // (For the random lottery the ordering is immaterial.)
        s = states[tid];
        shared_prod[lid] = s.z;
        workgroupBarrier();

        // UP-SWEEP
        if ((lid & 1u) == 1u) {
            shared_save[lid >> 1u] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 1u], shared_prod[lid]);
        }
        workgroupBarrier();
        if ((lid & 3u) == 3u) {
            shared_save[64u + (lid >> 2u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 2u], shared_prod[lid]);
        }
        workgroupBarrier();
        if ((lid & 7u) == 7u) {
            shared_save[96u + (lid >> 3u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 4u], shared_prod[lid]);
        }
        workgroupBarrier();
        if ((lid & 15u) == 15u) {
            shared_save[112u + (lid >> 4u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 8u], shared_prod[lid]);
        }
        workgroupBarrier();
        if ((lid & 31u) == 31u) {
            shared_save[120u + (lid >> 5u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 16u], shared_prod[lid]);
        }
        workgroupBarrier();
        if ((lid & 63u) == 63u) {
            shared_save[124u + (lid >> 6u)] = shared_prod[lid];
            shared_prod[lid] = fe_mul(shared_prod[lid - 32u], shared_prod[lid]);
        }
        workgroupBarrier();
        if (lid == 127u) {
            shared_save[126u] = shared_prod[127u];
            shared_prod[127u] = fe_mul(shared_prod[63u], shared_prod[127u]);
        }
        workgroupBarrier();

        // INVERT root
        if (lid == 0u) {
            shared_prod[127u] = fe_inv(shared_prod[127u]);
        }
        workgroupBarrier();

        // DOWN-SWEEP
        if (lid == 127u) {
            let inv_p = shared_prod[127u];
            let left = shared_prod[63u];
            let right = shared_save[126u];
            shared_prod[63u] = fe_mul(inv_p, right);
            shared_prod[127u] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 63u) == 63u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 32u];
            let right = shared_save[124u + (lid >> 6u)];
            shared_prod[lid - 32u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 31u) == 31u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 16u];
            let right = shared_save[120u + (lid >> 5u)];
            shared_prod[lid - 16u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 15u) == 15u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 8u];
            let right = shared_save[112u + (lid >> 4u)];
            shared_prod[lid - 8u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 7u) == 7u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 4u];
            let right = shared_save[96u + (lid >> 3u)];
            shared_prod[lid - 4u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 3u) == 3u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 2u];
            let right = shared_save[64u + (lid >> 2u)];
            shared_prod[lid - 2u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();
        if ((lid & 1u) == 1u) {
            let inv_p = shared_prod[lid];
            let left = shared_prod[lid - 1u];
            let right = shared_save[lid >> 1u];
            shared_prod[lid - 1u] = fe_mul(inv_p, right);
            shared_prod[lid] = fe_mul(inv_p, left);
        }
        workgroupBarrier();

        // ── 3. Affine conversion (pre-advance state) + hash ────────────────
        // `s` is still the state we read at the top of this step (before any
        // advance) — that's the key we are checking now.  The Jacobian X/Y/Z
        // were untouched by the inversion tree (which only used shared memory).
        let z_inv = shared_prod[lid];   // 1/Z from the inversion tree
        let z2_inv = fe_square(z_inv);
        let z3_inv = fe_mul(z2_inv, z_inv);
        let x_affine = fe_mul(s.x, z2_inv);
        let y_affine = fe_mul(s.y, z3_inv);

        // 4. Serialize compressed pubkey as 9 big-endian u32 words.
        var prefix: u32;
        if ((y_affine[0] & 1u) == 0u) { prefix = 0x02u; } else { prefix = 0x03u; }
        let x_be = limbs_to_be_words(x_affine);
        let cpk = compressed_pubkey_to_words(prefix, x_be);

        // 5-6. SHA256 + RIPEMD160 (hash160) via kangaroo's verified compressors.
        let sha = sha256_33bytes(cpk);
        let h160 = hash160_from_sha256(sha.h);
        // Candidate buffer is little-endian; flip kangaroo's big-endian output.
        let h160_le = hash160_be_to_le(h160);

        // 7. Candidate match
        let m = candidate_match(h160_le);
        if (m != 0xFFFFFFFFu) {
            let idx = atomicAdd(&match_count, 1u);
            if (idx < 256u) {
                matches[idx] = MatchOutput(s.scalar, x_affine, y_affine, h160.h, m, tid, 0u);
            }
        }

        // ── 8. Advance to the next key ───────────────────────────────────────
        // Done AFTER the hash so the seed key itself is checked (gap-free
        // tiling).  Point and scalar advance by the stride; the scalar tracks
        // the private key for match reporting.
        let p = jac_add_affine(jacobian(s.x, s.y, s.z), s.step_px, s.step_py);
        s.x = p.x; s.y = p.y; s.z = p.z;
        s.scalar = scalar_add_256(s.scalar, stride_le);
        states[tid] = s;
    }
}