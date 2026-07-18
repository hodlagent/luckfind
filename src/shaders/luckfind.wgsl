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
    num_threads: u32, steps_per_call: u32, num_candidates: u32, _padding: u32,
}
struct GpuState {
    x: array<u32, 8>, y: array<u32, 8>, z: array<u32, 8>, scalar: array<u32, 8>,
}
struct MatchOutput {
    scalar: array<u32, 8>, pubkey_x: array<u32, 8>, pubkey_y: array<u32, 8>,
    hash160: array<u32, 5>, candidate_index: u32, thread_id: u32, _padding: u32,
}

// Generator point G (affine), LE limbs.
const GX: array<u32, 8> = array<u32, 8>(
    0x16F81798u, 0x59F2815Bu, 0xD928CE2Du, 0xDBFC9B02u,
    0x070B87CEu, 0x9562A055u, 0xACBBDCF9u, 0x7E66BE79u
);
const GY: array<u32, 8> = array<u32, 8>(
    0xFB10D4B8u, 0x99C47D08u, 0x8A685541u, 0x8FD17B44u,
    0x8A10E1C0u, 0xBF4FDA55u, 0x463C6A72u, 0x0483ADA7u
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

fn pubkey_to_sha256_block(prefix: u32, x_be: array<u32, 8>) -> array<u32, 16> {
    var m: array<u32, 16>;
    m[0] = (prefix << 24u) | (x_be[0] >> 8u);
    for (var i = 0u; i < 7u; i++) {
        m[i + 1u] = ((x_be[i] & 0xFFu) << 24u) | (x_be[i + 1u] >> 8u);
    }
    m[8] = ((x_be[7] & 0xFFu) << 24u) | 0x800000u;
    for (var i = 9u; i < 14u; i++) { m[i] = 0u; }
    m[14] = 256u; m[15] = 0u;    // length in bits = 32 * 8
    return m;
}

fn sha256_to_ripemd160_block(sha: array<u32, 8>) -> array<u32, 16> {
    var m: array<u32, 16>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let be = sha[i];
        m[i] = ((be & 0xFFu) << 24u) | ((be & 0xFF00u) << 8u)
             | ((be & 0xFF0000u) >> 8u) | ((be & 0xFF000000u) >> 24u);
    }
    m[8] = 0x00000080u;
    for (var i = 9u; i < 15u; i++) { m[i] = 0u; }
    m[15] = 256u;
    return m;
}

fn candidate_match(h: array<u32, 5>) -> u32 {
    for (var i = 0u; i < 78u; i = i + 1u) {
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

    var s = states[tid];

    for (var step = 0u; step < config.steps_per_call; step = step + 1u) {
        // 1. P += G (Jacobian mixed add — ~12 field muls)
        let p = jac_add_affine(jacobian(s.x, s.y, s.z), GX, GY);
        s.x = p.x; s.y = p.y; s.z = p.z;
        s.scalar = scalar_add_256(s.scalar, array<u32,8>(1u,0u,0u,0u,0u,0u,0u,0u));

        // 2. Batch-invert 128 Z values (Montgomery trick: 1 fe_inv + 381 muls).
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

        // 3. Affine conversion: shared_prod[lid] = 1/Z
        let z_inv = shared_prod[lid];
        let z2_inv = fe_square(z_inv);
        let z3_inv = fe_mul(z2_inv, z_inv);
        let x_affine = fe_mul(s.x, z2_inv);
        let y_affine = fe_mul(s.y, z3_inv);

        // 4. Serialize compressed pubkey
        var prefix: u32;
        if ((y_affine[0] & 1u) == 0u) { prefix = 0x02u; } else { prefix = 0x03u; }
        let x_be = limbs_to_be_words(x_affine);

        // 5. SHA256
        let sha = sha256_block(pubkey_to_sha256_block(prefix, x_be));

        // 6. RIPEMD160
        let h = ripemd160_block(sha256_to_ripemd160_block(sha));

        // 7. Candidate match
        let m = candidate_match(h);
        if (m != 0xFFFFFFFFu) {
            let idx = atomicAdd(&match_count, 1u);
            if (idx < 256u) {
                matches[idx] = MatchOutput(s.scalar, x_affine, y_affine, h, m, tid, 0u);
            }
        }
    }

    states[tid] = s;
}