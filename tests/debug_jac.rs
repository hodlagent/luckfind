// Temporary debug: read back Jacobian after one step, affine-convert on CPU.
use luckfind::gpu::{self, GpuScanner};
use num_bigint::BigUint;
use num_traits::FromPrimitive;
use num_bigint::BigInt;

const P: [u32; 8] = [0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF];

fn limbs_to_big(l: &[u32; 8]) -> num_bigint::BigUint {
    let mut n = num_bigint::BigUint::from_u32(l[7]).unwrap();
    for i in (0..7).rev() { n <<= 32; n += num_bigint::BigUint::from_u32(l[i]).unwrap(); }
    n
}

#[test]
fn one_step_jacobian_is_2g() {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(_) => { eprintln!("no Metal, skipping"); return; }
    };
    let secp = secp256k1::Secp256k1::new();
    let two = secp256k1::SecretKey::from_byte_array({ let mut b=[0u8;32]; b[31]=2; b }).unwrap();
    let pk_2g = secp256k1::PublicKey::from_secret_key(&secp, &two).serialize();
    let expected_x = &pk_2g[1..33];

    let cand = vec![[0u32; 5]; 78];
    let mut scanner = GpuScanner::new(ctx, &cand).expect("new");
    scanner.init_random(luckfind::puzzles::puzzle_set()).expect("init");
    let g = secp256k1::PublicKey::from_slice(&luckfind::btc::GENERATOR_COMPRESSED).unwrap();
    let ge = g.serialize_uncompressed();
    let mut gx = [0u8;32]; let mut gy = [0u8;32];
    gx.copy_from_slice(&ge[1..33]); gy.copy_from_slice(&ge[33..65]);
    let (step_px, step_py) = gpu::convert::stride_step_point(1);
    scanner.set_initial_state(0, [1u32,0,0,0,0,0,0,0], gpu::GpuState {
        x: gpu::convert::be_bytes_to_limbs(&gx),
        y: gpu::convert::be_bytes_to_limbs(&gy),
        z: [1,0,0,0,0,0,0,0],
        scalar: [1,0,0,0,0,0,0,0],
        step_px,
        step_py,
    }).expect("set");
    scanner.steps_per_call = 1;
    let _ = scanner.step().expect("step");

    let states = scanner.readback_states().expect("readback");
    let s = &states[0];
    let p = limbs_to_big(&[0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
    let x = limbs_to_big(&s.x); let y = limbs_to_big(&s.y); let z = limbs_to_big(&s.z);
    let z2 = (&z * &z) % &p;
    let z2_inv = z2.modpow(&(&p - 2u32), &p);
    let x_affine = (&x * &z2_inv) % &p;
    println!("affine x = {:064x}", x_affine);
    println!("expected = {}", hex::encode(expected_x));
    assert_eq!(x_affine, BigUint::from_bytes_be(expected_x),
               "GPU point-add + inversion produced wrong affine x");
}

// Stage 1: point-add Jacobian Z must equal 2*GY mod p.
#[test]
fn stage1_point_add_z() {
    let ctx = match gpu::GpuContext::new_blocking(0) {
        Ok(c) => c, Err(_) => { eprintln!("skip"); return; }
    };
    let cand = vec![[0u32; 5]; 78];
    let mut s = GpuScanner::new(ctx, &cand).expect("new");
    s.init_random(luckfind::puzzles::puzzle_set()).expect("init");
    let g = secp256k1::PublicKey::from_slice(&luckfind::btc::GENERATOR_COMPRESSED).unwrap();
    let ge = g.serialize_uncompressed();
    let mut gx = [0u8;32]; let mut gy = [0u8;32];
    gx.copy_from_slice(&ge[1..33]); gy.copy_from_slice(&ge[33..65]);
    let (step_px, step_py) = gpu::convert::stride_step_point(1);
    s.set_initial_state(0, [1u32,0,0,0,0,0,0,0], gpu::GpuState {
        x: gpu::convert::be_bytes_to_limbs(&gx), y: gpu::convert::be_bytes_to_limbs(&gy),
        z: [1,0,0,0,0,0,0,0], scalar: [1,0,0,0,0,0,0,0],
        step_px, step_py,
    }).expect("set");
    s.steps_per_call = 1;
    let _ = s.step().expect("step");
    let st = s.readback_states().expect("readback");
    let z_gpu = limbs_to_big(&st[0].z);
    let p = limbs_to_big(&P);
    // Gy as LE limbs (limb[0] = lowest).  limbs_to_big treats limb[7] as most
    // significant, so ordering here is LEAST-significant-first.
    let gy = limbs_to_big(&[0xFB10D4B8,0x9C47D08F,0xA6855419,0xFD17B448,0x0E1108A8,0x5DA4FBFC,0x26A3C465,0x483ADA77]);
    let z_exp = (2u32 * &gy) % &p;
    println!("GPU Z = {:064x}", z_gpu);
    println!("CPU Z = {:064x}", z_exp);
    assert_eq!(z_gpu, z_exp, "stage1 point-add Z wrong");
}
