//! Temporary helper: generate a puzzle JSON (puzzle_number 16, range
//! [2^16, 2^17)) whose private key is known (0x12345), for end-to-end testing
//! of puzzle mode with --gpu_framework cuda.

fn main() {
    let secp = secp256k1::Secp256k1::new();
    // 0x12345 = 5 hex digits → bytes [01 23 45] end at b[31], so start at b[29].
    // (b[30]=0x01, b[31]=0x23 would be 0x123 = 291 — outside the range!)
    let mut b = [0u8; 32];
    b[24..32].copy_from_slice(&0x12345u64.to_be_bytes());
    let sk = secp256k1::SecretKey::from_byte_array(b).unwrap();
    let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
    let compressed = pk.serialize();
    let addr = luckfind::btc::p2pkh_compressed(&compressed);
    let h160 = luckfind::btc::hash160(&compressed);
    let start = "0000000000000000000000000000000000000000000000000000000000010000";
    let end = "0000000000000000000000000000000000000000000000000000000000020000";
    println!(
        "sk_hex=0000000000000000000000000000000000000000000000000000000000012345"
    );
    println!("address={addr}");
    println!("hash160={}", hex::encode(h160));
    println!(
        "JSON={{\"puzzle_number\":16,\"total_bits\":16,\"target\":\"{addr}\",\"hash160\":\"{}\",\"start_hex\":\"{start}\",\"end_hex\":\"{end}\",\"next_id\":2,\"chunks\":[{{\"id\":1,\"current_hex\":\"{start}\",\"end_hex\":\"{end}\",\"status\":\"pending\"}}]}}",
        hex::encode(h160)
    );
}
