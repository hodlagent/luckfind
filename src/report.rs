//! Terminal report + got.txt append flush.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

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
            println!(
                "     #{}  worker #{}  idx={}  pk_hex={}",
                i + 1,
                m.worker_id,
                crate::workers::fmt_comma(m.key_index),
                hex::encode(m.private_key),
            );
        }
    }
    println!("  Keys checked:  {}", crate::workers::fmt_comma(total));
    println!("  Total time:    {:.2}s ({:.2}m)", elapsed, elapsed / 60.0);
    println!("  Average rate:  {} keys/sec", crate::workers::fmt_comma(rate as u64));
    println!();
}

pub fn flush_got(matches: &[MatchEvent], output_dir: Option<&Path>) {
    let dir = output_dir.unwrap_or_else(|| Path::new("."));
    if matches.is_empty() {
        return;
    }
    let path = dir.join("got.txt");
    let mut f = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("  ⚠️  cannot open got.txt: {}", e);
            return;
        }
    };
    for m in matches {
        let chunk_label = match m.chunk_id {
            Some(id) => format!(" chunk={id}"),
            None      => String::new(),
        };
        let _ = writeln!(
            f,
            "ts={} worker={}{} idx={} sk_hex={} pk_c={} pk_u={} elapsed={:.2}s",
            chrono::Utc::now().to_rfc3339(),
            m.worker_id,
            chunk_label,
            m.key_index,
            hex::encode(m.private_key),
            hex::encode(&m.compressed),
            hex::encode(&m.uncompressed),
            m.elapsed,
        );
    }
}
