//! Generate a tiny puzzle worklist whose target address is derived from a
//! *known* private key — exercises 命中即停 end-to-end: [HIT] print (含私钥)
//! → aman_<TS>.txt → sqlite 最后更新。
//!
//! The worklist chunk is a SINGLE key wide (`[key, key+1)`) so the random-split
//! strategy never fragments it — the worker claims it and must hit immediately.
//!
//! ```
//! cargo run --release --example make_hit_worklist /tmp/hit.json
//! mkdir -p /tmp/hit-test
//! target/release/luckfind --puzzle /tmp/hit.json -w 1 -o /tmp/hit-test
//! ```
//!
//! Expected: a `[HIT] 🎯 … sk_hex=…` line appears immediately, the run stops
//! right away, `/tmp/hit-test/aman_<TS>.txt` contains the private key + 5
//! derived addresses, and the sibling `/tmp/hit.db` marks the chunk finished.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.get(1).map(String::as_str).unwrap_or("/tmp/hit.json");

    // Known hit key (small, unambiguous).
    let key: u64 = 1 << 20;

    let mut kbytes = [0u8; 32];
    kbytes[24..32].copy_from_slice(&key.to_be_bytes());
    let secp = secp256k1::Secp256k1::new();
    let sk = secp256k1::SecretKey::from_byte_array(kbytes).unwrap();
    let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
    let address = luckfind::btc::p2pkh_compressed(&pk.serialize());
    // Round-trip sanity: the target address must decode back to the key's hash160.
    assert_eq!(
        luckfind::btc::legacy_address_hash160(&address),
        Some(luckfind::btc::hash160(&pk.serialize())),
        "target address does not decode to the hit key's hash160"
    );

    // Single-key chunk [key, key+1) — never split, hit at idx=0.
    let start_hex = format!("{key:x}");
    let end_hex = format!("{:x}", key + 1);

    let json = format!(
        r#"{{
  "puzzle_number": 20,
  "total_bits": 20,
  "chunk_bits_used": 20,
  "total_chunks": 1,
  "completed_chunks": 0,
  "target": "{address}",
  "start_hex": "{start_hex}",
  "end_hex": "{end_hex}",
  "next_id": 2,
  "chunks": [
    {{ "id": 1, "current_hex": "{start_hex}", "end_hex": "{end_hex}", "status": "pending" }}
  ]
}}"#
    );
    std::fs::write(out, json).unwrap();
    println!("wrote {out}");
    println!("target address = {address}");
    println!("hit key hex    = {}", hex::encode(kbytes));
}
