//! Terminal report + per-match `aman_<UTC>.txt` flush.

use std::path::Path;
use std::time::Instant;

use crate::btc;
use crate::progress::Progress;
use crate::workers::MatchEvent;

pub fn final_report(progress: &Progress, matches: &[MatchEvent], start: &Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let total = progress.checked.load(std::sync::atomic::Ordering::Relaxed);
    let rate = if elapsed > 1.0 { total as f64 / elapsed } else { 0.0 };

    println!("──────────────────────────────────────────────────");
    println!("  RESULTS");
    println!("──────────────────────────────────────────────────");
    if matches.is_empty() {
        println!("  ❌ No match found.");
    } else {
        println!("  ✅ {} match(es) total:", matches.len());
        for (i, m) in matches.iter().enumerate() {
            let puzzle_label = match m.puzzle_number {
                Some(n) => format!(" puzzle={n}"),
                None    => String::new(),
            };
            println!(
                "     #{}  worker #{}  idx={}{}  pk_hex={}",
                i + 1,
                m.worker_id,
                crate::workers::fmt_comma(m.key_index),
                puzzle_label,
                hex::encode(m.private_key),
            );
        }
    }
    println!("  Keys checked:  {}", crate::workers::fmt_comma(total));
    println!("  Total time:    {:.2}s ({:.2}m)", elapsed, elapsed / 60.0);
    println!("  Average rate:  {} keys/sec", crate::workers::fmt_comma(rate as u64));
    println!();
}

/// Write one `aman_<UTC>.txt` per match (per-hit file, never overwrites).
///
/// Each file carries the private key (hex + decimal), compressed/uncompressed
/// pubkeys, and all 5 derived address types.  Address derivation reuses the
/// existing `btc` helpers — no duplicated hashing here.
pub fn flush_match_files(matches: &[MatchEvent], output_dir: Option<&Path>) {
    let dir = output_dir.unwrap_or_else(|| Path::new("."));
    for m in matches {
        let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let path = dir.join(format!("aman_{ts}.txt"));
        match std::fs::write(&path, match_block(m)) {
            Ok(_) => eprintln!("  📄 Saved     : {}", path.display()),
            Err(e) => eprintln!("  ⚠️  cannot write {}: {}", path.display(), e),
        }
    }
}

/// Human-readable block for one match — mirrors the old Python orchestrator's
/// `aman_<UTC>.txt`.  The 5 address types come straight from `btc.rs`.
fn match_block(m: &MatchEvent) -> String {
    let mut s = String::new();
    s.push_str("================================================================\n");
    s.push_str("  BITCOIN DORMANT ADDRESS LOTTERY — MATCH FOUND\n");
    s.push_str("================================================================\n");
    s.push_str(&format!("  Worker    : #{}\n", m.worker_id));
    if let Some(id) = m.chunk_id {
        s.push_str(&format!("  Chunk     : #{id}\n"));
    }
    if let Some(n) = m.puzzle_number {
        s.push_str(&format!("  Puzzle    : #{n}\n"));
    }
    s.push_str(&format!(
        "  Index     : {}\n",
        crate::workers::fmt_comma(m.key_index)
    ));
    s.push_str(&format!("  Time      : {:.2}s\n", m.elapsed));
    s.push('\n');
    s.push_str("  ── PRIVATE KEY ──────────────────────────────────────\n");
    s.push_str(&format!("  Hex     : {}\n", hex::encode(m.private_key)));
    s.push_str(&format!(
        "  Decimal : {}\n",
        num_bigint::BigUint::from_bytes_be(&m.private_key)
    ));
    s.push('\n');
    s.push_str("  ── PUBLIC KEY ────────────────────────────────────────\n");
    s.push_str(&format!("  Compressed  : {}\n", hex::encode(&m.compressed)));
    s.push_str(&format!("  Uncompressed: {}\n", hex::encode(&m.uncompressed)));
    s.push('\n');
    s.push_str("  ── ADDRESSES ─────────────────────────────────────────\n");
    s.push_str(&format!(
        "  P2PKH (compressed)   : {}\n",
        btc::p2pkh_compressed(&m.compressed)
    ));
    s.push_str(&format!(
        "  P2PKH (uncompressed) : {}\n",
        btc::p2pkh_uncompressed(&m.uncompressed)
    ));
    s.push_str(&format!(
        "  P2SH-P2WPKH          : {}\n",
        btc::p2sh_p2wpkh(&m.compressed)
    ));
    s.push_str(&format!("  P2WPKH               : {}\n", btc::p2wpkh(&m.compressed)));
    s.push_str(&format!("  P2TR                 : {}\n", btc::p2tr(&m.compressed)));
    s
}
