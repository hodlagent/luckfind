//! Puzzle-mode scanner.
//!
//! Reads a worklist file (JSON or SQLite `.db`) that subdivides a puzzle
//! key-range into sub-range `chunks`.  Each chunk defines:
//!
//! - `id`:           stable identifier (monotonic; new ids are assigned as
//!                    chunks are split).
//! - `current_hex`:  the effective left bound AND the resume position.  May be
//!                    < 64 hex chars; it is always zero-padded on the *left*
//!                    (high bytes) to a full 32-byte secp256k1 key.  Scanning
//!                    advances this; on resume the worker starts here.  Always
//!                    set (never null).
//! - `end_hex`:      exclusive upper bound of the *unscanned* region.  Scanned
//!                    forward it stays fixed; scanned in reverse it retreats as
//!                    keys are consumed from the top.  Either way `[current, end)`
//!                    is always exactly the not-yet-scanned keys.
//! - `status`:       `"pending"`, `"running"`, or `"finished"`.
//!
//! Per worker: pick a random *pending* chunk, claim it (status = "running"),
//! then walk every key `current .. end` by scalar +1 (the same tight `+= 1`
//! loop used in lottery mode for speed).  Each claim randomly flips the scan
//! direction: forward (from `current`, `+1`) or reverse (from `end - 1`, `-1`).
//! Both cover the identical key set `[current, end)` — only the in-order
//! traversal differs — so coverage, zero-dedup, and resume are unchanged; the
//! direction is a per-claim coin flip, never persisted.  When the whole
//! sub-range is done, the chunk is marked `finished`.  On SIGINT (Ctrl+C) every
//! worker flushes
//! its current scanning position into the SQLite worklist DB, reverts the
//! chunk to `"pending"`, and exits — so a later invocation resumes cleanly.
//!
//! **Random-split strategy.**  The worklist starts with one chunk per
//! puzzle-range subdivision and grows dynamically: when a worker claims a
//! pending chunk `[x, y)` and the worklist is below the cap
//! (`MAX_CHUNKS = 2^24 = 4096 × 4096`), it splits the chunk at a random point
//! `d ∈ (x, y)`, scans the upper half `[d, y)`, and parks the lower half
//! `[x, d)` as a fresh pending chunk with a new id.  Once the cap is reached
//! workers simply scan the picked chunk directly.  Sub-ranges narrower than
//! `ROTATION_BUDGET = 2^27` keys are scanned to completion in one claim;
//! wider ones use the rotation mechanism (park after the per-claim
//! `rotate_keys` budget — `2^27` on CPU, `2^31` on GPU — and resume later).
//! This gives a "jump around the key space" scan with the same
//! zero-dedup guarantee as a sequential sweep.
//!
//! Bytes saved per chunk:
//!
//! ```json
//! {
//!   "id": 42,
//!   "current_hex": "8200000000000000000",
//!   "end_hex": "8400000000000000000",
//!   "status": "pending"
//! }
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use num_bigint::BigUint;
use num_traits::One;
use rand::TryRng;
use serde::{Deserialize, Serialize};

use crate::btc;
use crate::progress::Progress;
use crate::workers::{fmt_comma, MatchEvent};

/// Maximum number of sub-ranges the worklist may grow to via splitting.
/// 2^24 = 16_777_216 = 4096 × 4096.  Once reached, workers stop splitting and
/// just scan the picked sub-range (with rotation if it exceeds the per-claim
/// budget).
const MAX_CHUNKS: usize = 1 << 24;

/// Per-claim scan budget (CPU workers).  Sub-ranges narrower than this are
/// scanned to completion in one claim; wider ones are parked after the CPU
/// `rotate_keys` budget and resumed later (the "rotation" mechanism).
///
/// This is the CPU-side boundary only.  The GPU worker scans with its own
/// rotation budget (see `gpu_rotate_keys` in `run`) and never calls
/// `scan_budget`, so this can be tuned independently of the GPU cadence.
pub(crate) const ROTATION_BUDGET: u64 = 1u64 << 27;

// ── terminal output coordination ─────────────────────────────────────────────
// The heartbeat rewrites a single status line in place with `\r` (no trailing
// newline).  Full log lines (claims, errors, final report) must first
// terminate any open status line so they start at column 0 instead of the
// cursor's mid-line position.  `TERM_LINE` serializes the two so a claim line
// can't be clobbered by a status rewrite (or vice versa).

struct TermLine {
    open: bool,
}

static TERM_LINE: Mutex<TermLine> = Mutex::new(TermLine { open: false });

/// Rewrite the in-place status line (carriage return + ANSI clear-to-EOL).
pub(crate) fn term_status(s: &str) {
    let mut g = TERM_LINE.lock().unwrap_or_else(|e| e.into_inner());
    eprint!("\r{s}\x1b[K");
    g.open = true;
}

/// Emit a full log line, terminating any open status line first.
pub(crate) fn term_line(s: &str) {
    let mut g = TERM_LINE.lock().unwrap_or_else(|e| e.into_inner());
    if g.open {
        eprint!("\r\n"); // drop the cursor to column 0 of the next line
        g.open = false;
    }
    eprintln!("{s}");
}

// ── worklist JSON shape ──────────────────────────────────────────────────────

/// On-disk puzzle worklist.  The overall puzzle range is `[start_hex, end_hex)`;
/// `chunks` holds the live sub-ranges that partition (part of) that range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleFile {
    pub puzzle_number: u32,
    pub total_bits: u32,
    #[serde(default)]
    pub chunk_bits_used: u32,
    #[serde(default)]
    pub total_chunks: u32,
    #[serde(default)]
    pub completed_chunks: u32,
    pub target: String,
    /// Optional expected RIPEMD-160 hash (40 hex chars) of `target`.  When
    /// present it is checked against the hash decoded from `target` — a mismatch
    /// means the JSON is inconsistent and we abort.  Absent ⇒ skipped.
    pub hash160: Option<String>,
    pub chunks: Vec<Chunk>,

    // ── new fields (random-split strategy) ─────────────────────────────────────
    /// Overall puzzle range `[start_hex, end_hex)`.  For puzzle 76 this is
    /// `[2^75, 2^76)`.  Derived from `total_bits` on migration when absent.
    #[serde(default)]
    pub start_hex: String,
    #[serde(default)]
    pub end_hex: String,
    /// Monotonic counter for assigning ids to newly split-off chunks.
    #[serde(default)]
    pub next_id: u32,
}

/// A single sub-range in the puzzle worklist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    #[serde(default)]
    pub id: u32,
    /// Most-recent scan position AND the effective left bound of the sub-range.
    /// Scanning advances this; on resume the worker starts here.  Always set.
    #[serde(default)]
    pub current_hex: String,
    /// Exclusive upper bound.  Never changes for a given chunk.
    #[serde(default)]
    pub end_hex: String,
    pub status: String,

    // ── legacy fields (old worklist format); cleared on migration.
    // Skipped on serialization so the on-disk format stays clean (new chunks
    // never carry them).  `#[serde(default)]` lets old JSON deserialize.
    #[serde(default, skip_serializing_if = "u32_is_zero")]
    pub chunk_index: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub start_hex: String,
    #[serde(default, skip_serializing_if = "u32_is_zero")]
    pub range_bits: u32,
}

fn u32_is_zero(v: &u32) -> bool {
    *v == 0
}

// ── runtime state ────────────────────────────────────────────────────────────

/// Shared state across workers: the parsed worklist and the path to persist to.
/// All runtime saves go to `db_path` (SQLite); the JSON worklist (if any) is
/// only a one-time import and never written again.
///
/// `dirty` tracks chunk IDs whose in-memory state has changed since the last
/// save and need to be flushed to the DB.  This lets us UPDATE/INSERT only the
/// handful of chunks that actually changed (on claim, split, park, finalize)
/// instead of rewriting the entire worklist table on every save.
#[derive(Debug)]
struct PuzzleCtx {
    file: PuzzleFile,
    db_path: PathBuf,
    dirty: HashSet<u32>,
}

impl PuzzleCtx {
    fn new(file: PuzzleFile, db_path: PathBuf) -> Self {
        Self {
            file,
            db_path,
            dirty: HashSet::new(),
        }
    }

    /// Mark a chunk dirty so the next `save_dirty` flushes it to the DB.
    fn mark_dirty(&mut self, id: u32) {
        self.dirty.insert(id);
    }

    /// Mark every chunk dirty.  Used after migration when loading from DB: the
    /// upgraded in-memory state must be rewritten to the DB on the first save.
    fn mark_all_dirty(&mut self) {
        for c in &self.file.chunks {
            self.dirty.insert(c.id);
        }
    }
}

// ── migration: old worklist format → new random-split format ─────────────────

/// Migrate an old-format worklist (with `start_hex` + `range_bits` per chunk) to
/// the new random-split format (`current_hex` + `end_hex`).  Detection heuristic:
/// a chunk is "old format" iff `range_bits != 0` (old chunks always carry
/// `range_bits: 65`; new chunks default to 0) or `end_hex` is empty.  Idempotent:
/// new-format chunks are left untouched.
///
/// Returns `true` if any chunk was migrated (so the caller can mark all chunks
/// dirty and persist the upgraded state on the next save).
fn migrate_puzzle_file(file: &mut PuzzleFile) -> bool {
    // Assign ids from a fresh monotonic counter seeded above every existing
    // id.  Using a counter (rather than chunk_index) means a re-run after a
    // killed partial-migration can never collide with an id already written:
    // every new id is strictly greater than all pre-existing ones, regardless
    // of chunk_index values or how far the previous run got.
    let mut next_free = file
        .chunks
        .iter()
        .map(|c| c.id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let mut changed = false;
    for c in &mut file.chunks {
        let legacy = c.range_bits != 0 || c.end_hex.is_empty();
        if !legacy {
            continue;
        }
        changed = true;

        let base = parse_hex_key(&c.start_hex);
        let end = chunk_end(&base, c.range_bits);

        c.end_hex = hex_encode_key(&end);
        c.id = next_free;
        next_free = next_free.saturating_add(1);

        // Resume position: keep the old current_hex iff it lies in [base, end);
        // otherwise self-heal to the chunk's nominal start (preserving the
        // original short-hex representation, matching the existing resume
        // logic).
        if c.current_hex.is_empty() {
            c.current_hex = c.start_hex.clone();
        } else {
            let cur = parse_hex_key(&c.current_hex);
            let in_range = cur >= base && crate::gpu::convert::be_lt(&cur, &end);
            if !in_range {
                c.current_hex = c.start_hex.clone();
            }
        }

        // Zero legacy fields so they don't serialize on the next save and don't
        // re-trigger migration.
        c.range_bits = 0;
        c.start_hex = String::new();
        c.chunk_index = 0;
    }

    file.next_id = next_free;
    file.total_chunks = file.chunks.len() as u32;

    // Derive overall puzzle range [2^total_bits, 2^(total_bits+1)) when absent.
    if file.start_hex.is_empty() {
        let start = BigUint::one() << file.total_bits; // 2^total_bits
        let end = BigUint::one() << (file.total_bits + 1); // 2^(total_bits+1)
        file.start_hex = hex_encode_key(&biguint_to_32be(&start));
        file.end_hex = hex_encode_key(&biguint_to_32be(&end));
    }

    changed
}

// ── SQLite persistence (.db format) ──────────────────────────────────────────

/// Status codes stored in the SQLite `chunks.status` column.  Shared convention
/// with the Python `scripts/split.py` writer.
const STATUS_PENDING: i64 = 0;
const STATUS_RUNNING: i64 = 1;
const STATUS_FINISHED: i64 = 2;

fn status_str_to_int(s: &str) -> i64 {
    match s {
        "pending" => STATUS_PENDING,
        "running" => STATUS_RUNNING,
        "finished" => STATUS_FINISHED,
        _ => STATUS_PENDING,
    }
}

fn status_int_to_str(v: i64) -> &'static str {
    match v {
        STATUS_PENDING => "pending",
        STATUS_RUNNING => "running",
        STATUS_FINISHED => "finished",
        _ => "pending",
    }
}

/// Schema version written into the `meta` table.  Lets us detect/format-mismatch
/// on load and migrate later if the schema evolves.
const DB_SCHEMA_VERSION: i64 = 1;

/// Create a fresh SQLite worklist DB from an in-memory `PuzzleFile`.  This is the
/// "JSON import" path: the first run from a `.json` worklist writes a `.db`
/// sibling and all subsequent saves go there.
fn create_db_from_file(path: &Path, file: &PuzzleFile) -> Result<(), String> {
    let conn =
        rusqlite::Connection::open(path).map_err(|e| format!("create {}: {}", path.display(), e))?;

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS meta (
            puzzle_number    INTEGER NOT NULL,
            total_bits       INTEGER NOT NULL,
            chunk_bits_used  INTEGER NOT NULL,
            total_chunks     INTEGER NOT NULL,
            completed_chunks INTEGER NOT NULL,
            target           TEXT NOT NULL,
            hash160          TEXT,
            range_start      BLOB(32) NOT NULL,
            range_end        BLOB(32) NOT NULL,
            next_id          INTEGER NOT NULL,
            schema_version   INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY,
            current BLOB(32) NOT NULL,
            end     BLOB(32) NOT NULL,
            status  INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);
        ",
    )
    .map_err(|e| format!("create schema in {}: {}", path.display(), e))?;

    let range_start = parse_hex_key(&file.start_hex);
    let range_end = parse_hex_key(&file.end_hex);

    conn.execute(
        "INSERT INTO meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rusqlite::params![
            file.puzzle_number as i64,
            file.total_bits as i64,
            file.chunk_bits_used as i64,
            file.total_chunks as i64,
            file.completed_chunks as i64,
            &file.target,
            &file.hash160,
            &range_start[..],
            &range_end[..],
            file.next_id as i64,
            DB_SCHEMA_VERSION,
        ],
    )
    .map_err(|e| format!("insert meta into {}: {}", path.display(), e))?;

    let mut stmt = conn
        .prepare("INSERT INTO chunks (id, current, end, status) VALUES (?, ?, ?, ?)")
        .map_err(|e| e.to_string())?;
    for chunk in &file.chunks {
        let cur = parse_hex_key(&chunk.current_hex);
        let end = parse_hex_key(&chunk.end_hex);
        stmt.execute(rusqlite::params![
            chunk.id as i64,
            &cur[..],
            &end[..],
            status_str_to_int(&chunk.status),
        ])
        .map_err(|e| format!("insert chunk #{} into {}: {}", chunk.id, path.display(), e))?;
    }

    Ok(())
}

/// Load a `PuzzleFile` from a SQLite worklist DB.
fn load_from_db(path: &Path) -> Result<PuzzleFile, String> {
    let conn =
        rusqlite::Connection::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;

    // Read the single meta row.
    let (puzzle_number, total_bits, chunk_bits_used, total_chunks, completed_chunks, target, hash160, range_start_blob, range_end_blob, next_id, _schema_ver) = conn
        .query_row(
            "SELECT puzzle_number, total_bits, chunk_bits_used, total_chunks, \
                    completed_chunks, target, hash160, range_start, range_end, \
                    next_id, schema_version FROM meta",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Vec<u8>>(7)?,
                    row.get::<_, Vec<u8>>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                ))
            },
        )
        .map_err(|e| format!("read meta from {}: {}", path.display(), e))?;

    let mut range_start = [0u8; 32];
    range_start
        .copy_from_slice(&range_start_blob);
    let mut range_end = [0u8; 32];
    range_end.copy_from_slice(&range_end_blob);

    // Read all chunks.
    let mut stmt = conn
        .prepare("SELECT id, current, end, status FROM chunks ORDER BY id")
        .map_err(|e| e.to_string())?;
    let chunk_rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, Vec<u8>>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(|e| e.to_string())?;

    let mut chunks = Vec::new();
    for row in chunk_rows {
        let (id, cur_blob, end_blob, status_int) = row.map_err(|e| e.to_string())?;
        let mut cur = [0u8; 32];
        cur.copy_from_slice(&cur_blob);
        let mut end = [0u8; 32];
        end.copy_from_slice(&end_blob);
        chunks.push(Chunk {
            id: id as u32,
            current_hex: hex_encode_key(&cur),
            end_hex: hex_encode_key(&end),
            status: status_int_to_str(status_int).to_string(),
            chunk_index: 0,
            start_hex: String::new(),
            range_bits: 0,
        });
    }

    Ok(PuzzleFile {
        puzzle_number: puzzle_number as u32,
        total_bits: total_bits as u32,
        chunk_bits_used: chunk_bits_used as u32,
        total_chunks: total_chunks as u32,
        completed_chunks: completed_chunks as u32,
        target,
        hash160,
        chunks,
        start_hex: hex_encode_key(&range_start),
        end_hex: hex_encode_key(&range_end),
        next_id: next_id as u32,
    })
}

/// Full-sync the entire `PuzzleFile` to SQLite (DELETE all chunks + re-INSERT).
/// Not on the hot path — `save_dirty` handles incremental saves.  Kept as a
/// primitive for tests, DB compaction, and the `save_all` escape hatch.
#[allow(dead_code)]
fn save_to_db(path: &Path, file: &PuzzleFile) -> Result<(), String> {
    let mut conn =
        rusqlite::Connection::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;

    tx.execute("DELETE FROM chunks", [])
        .map_err(|e| e.to_string())?;
    {
        let mut stmt = tx
            .prepare("INSERT INTO chunks (id, current, end, status) VALUES (?, ?, ?, ?)")
            .map_err(|e| e.to_string())?;
        for chunk in &file.chunks {
            let cur = parse_hex_key(&chunk.current_hex);
            let end = parse_hex_key(&chunk.end_hex);
            stmt.execute(rusqlite::params![
                chunk.id as i64,
                &cur[..],
                &end[..],
                status_str_to_int(&chunk.status),
            ])
            .map_err(|e| format!("write chunk #{} to {}: {}", chunk.id, path.display(), e))?;
        }
    }

    tx.execute(
        "UPDATE meta SET total_chunks = ?, completed_chunks = ?, next_id = ?",
        rusqlite::params![
            file.total_chunks as i64,
            file.completed_chunks as i64,
            file.next_id as i64,
        ],
    )
    .map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

/// Incremental save: flush only the chunks whose IDs are in `dirty` to the DB.
/// Each dirty chunk is written with INSERT OR REPLACE (covers both "existing
/// chunk was updated" and "brand-new chunk was split off").  Uses a transaction
/// so a crash mid-write cannot corrupt the DB.  Clears `dirty` on success.
///
/// This is the hot-path save: a claim touches 1-2 chunks, a rotation touches 1,
/// a Ctrl+C finalizes ≤ n_workers.  Even at the 16M-chunk cap we write O(1)
/// rows instead of O(N).
fn save_dirty(ctx: &mut PuzzleCtx) -> Result<(), String> {
    if ctx.dirty.is_empty() {
        return Ok(());
    }

    // Snapshot the dirty IDs and clear the set up front.  This way, if a chunk
    // is modified again while we're mid-write (e.g. the ticker fires during a
    // worker's claim), the new modification will be caught by the *next* save
    // rather than lost.
    let dirty_ids: Vec<u32> = ctx.dirty.drain().collect();

    let mut conn = rusqlite::Connection::open(&ctx.db_path)
        .map_err(|e| format!("open {}: {}", ctx.db_path.display(), e))?;
    let tx = conn.transaction().map_err(|e| e.to_string())?;

    {
        let mut stmt = tx
            .prepare(
                "INSERT OR REPLACE INTO chunks (id, current, end, status) \
                 VALUES (?, ?, ?, ?)",
            )
            .map_err(|e| e.to_string())?;

        for id in &dirty_ids {
            // Resolve the chunk by id.  A chunk may have been marked dirty and
            // then removed (shouldn't happen with the current logic, but be
            // defensive) — skip it.
            let chunk = match ctx.file.chunks.iter().find(|c| c.id == *id) {
                Some(c) => c,
                None => continue,
            };
            let cur = parse_hex_key(&chunk.current_hex);
            let end = parse_hex_key(&chunk.end_hex);
            stmt.execute(rusqlite::params![
                chunk.id as i64,
                &cur[..],
                &end[..],
                status_str_to_int(&chunk.status),
            ])
            .map_err(|e| format!("write chunk #{} to {}: {}", chunk.id, ctx.db_path.display(), e))?;
        }
    }

    // Meta fields (total_chunks, completed_chunks, next_id) are mutable and may
    // have changed alongside the dirty chunks — keep them in sync.
    tx.execute(
        "UPDATE meta SET total_chunks = ?, completed_chunks = ?, next_id = ?",
        rusqlite::params![
            ctx.file.total_chunks as i64,
            ctx.file.completed_chunks as i64,
            ctx.file.next_id as i64,
        ],
    )
    .map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

// ── public entry point ──────────────────────────────────────────────────────

/// Run the puzzle loop.  Returns total keys checked and any match events.
///
/// `rotate_keys` enables *random-rotation* mode for the CPU workers: each claim
/// scans at most this many keys before the chunk is parked (status ←
/// "pending", `current_hex` saved) and the worker moves on to a fresh random
/// pending chunk.  `gpu_rotate_keys` is the same per-claim budget for the
/// (single) GPU worker, so the two backends can rotate at different cadences.
/// `None` preserves classic behaviour — a chunk is scanned to completion per
/// claim.  When set, `n_workers` slots each churn through random chunks, giving
/// a "jump around the worklist" effect instead of sweeping it in order.
pub fn run(
    path: &Path,
    n_workers: usize,
    heartbeat_secs: f64,
    rotate_keys: Option<u64>,
    gpu_rotate_keys: Option<u64>,
    output_dir: Option<&Path>,
    framework: crate::framework::GpuFramework,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    // ── 1. load the worklist ────────────────────────────────────────────────
    //
    // Format detection:
    //   • `.db`  → load directly from SQLite.
    //   • `.json` → if a sibling `{n}.db` already exists, load from that
    //               (JSON is ignored — it was only the initial import);
    //               otherwise load from JSON and create the `.db` for all
    //               future saves.
    //
    // JSON is thus a one-time import format; the DB is the runtime source of
    // truth and the only thing we ever write back to.
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    let (mut file, db_path) = match ext.as_deref() {
        Some("db") => {
            let file = load_from_db(path).unwrap_or_else(|e| {
                eprintln!("[puzzle] {e}");
                std::process::exit(2);
            });
            (file, path.to_path_buf())
        }
        Some("json") => {
            // Derive the sibling `.db` path: same directory & stem, `.db` ext.
            let db_path = path.with_extension("db");
            if db_path.exists() {
                eprintln!(
                    "[puzzle] found existing database {} — loading from it \
                     (ignoring {})",
                    db_path.display(),
                    path.display()
                );
                let file = load_from_db(&db_path).unwrap_or_else(|e| {
                    eprintln!("[puzzle] {e}");
                    std::process::exit(2);
                });
                (file, db_path)
            } else {
                // First run from JSON: parse it, then create the DB.
                let file: PuzzleFile = serde_json::from_reader(std::io::BufReader::new(
                    std::fs::File::open(path).unwrap_or_else(|e| {
                        eprintln!("[puzzle] cannot open {}: {}", path.display(), e);
                        std::process::exit(2);
                    }),
                ))
                .unwrap_or_else(|e| {
                    eprintln!("[puzzle] invalid JSON in {}: {}", path.display(), e);
                    std::process::exit(2);
                });
                (file, db_path)
            }
        }
        _ => {
            eprintln!(
                "[puzzle] unsupported file extension (expected .json or .db): {}",
                path.display()
            );
            std::process::exit(2);
        }
    };

    // Convert the target BTC address to its 20-byte hash160 for fast compare.
    // This decode happens once at startup; both CPU workers and the GPU worker
    // reuse the resulting 20-byte value — no Base58 work on the hot path.
    let target_h160 = btc::legacy_address_hash160(&file.target).unwrap_or_else(|| {
        eprintln!("[puzzle] target {} is not a valid P2PKH address", file.target);
        std::process::exit(2);
    });

    // If the worklist ships an expected hash160, sanity-check it against the
    // value we just decoded from `target`.  A mismatch means the worklist is
    // internally inconsistent — we abort rather than scan the wrong set.
    // Absent ⇒ no check (backward-compatible with files that omit the field).
    if let Some(ref h160_hex) = file.hash160 {
        match hash160_from_hex(h160_hex) {
            Some(expected) if expected == target_h160 => {
                eprintln!("[puzzle] hash160 OK ({h160_hex})");
            }
            Some(_) => {
                eprintln!(
                    "[puzzle] hash160 MISMATCH: JSON says {h160_hex}, \
                     target {} decodes to {}",
                    file.target,
                    hex::encode(target_h160)
                );
                std::process::exit(2);
            }
            None => {
                eprintln!("[puzzle] hash160 in worklist is not valid hex: {h160_hex}");
                std::process::exit(2);
            }
        }
    }

    // Migrate an old-format worklist (start_hex + range_bits) to the new
    // random-split format (current_hex + end_hex).  Idempotent on new files.
    let migrated = migrate_puzzle_file(&mut file);

    // Crash-recovery: any chunk left "running" from a previous killed run was
    // in-flight — revert it to pending so it can be reclaimed.
    for c in file.chunks.iter_mut() {
        if c.status == "running" {
            c.status = "pending".to_string();
        }
    }

    // If we loaded from JSON and the DB doesn't exist yet, create it now so all
    // subsequent saves go to the DB.  The JSON is never written again.
    if !db_path.exists() {
        eprintln!("[puzzle] creating database {} (future saves go here)", db_path.display());
        create_db_from_file(&db_path, &file).unwrap_or_else(|e| {
            eprintln!("[puzzle] failed to create database: {e}");
            std::process::exit(2);
        });
    }

    // Build the shared state.  If migration changed any chunk and we loaded
    // from DB (not JSON import, which already wrote everything fresh above),
    // the DB still holds the pre-migration state — mark every chunk dirty so
    // the first save_dirty rewrites them all in the new format.
    let mut ctx = PuzzleCtx::new(file, db_path);
    if migrated {
        ctx.mark_all_dirty();
    }

    let summary = chunk_summary(&ctx.file);
    println!(
        "[puzzle #{}] target={}  chunks={}  (pending={}, running={}, finished={})",
        ctx.file.puzzle_number, ctx.file.target, ctx.file.total_chunks, summary.0, summary.1, summary.2,
    );

    if summary.0 + summary.1 == 0 {
        println!("[puzzle] all chunks already finished — nothing to do.");
        return (Arc::new(Progress::new(0)), Vec::new());
    }

    // ── 2. shared state + SIGINT flag ───────────────────────────────────────
    let ctx = Arc::new(Mutex::new(ctx));
    let progress = Arc::new(Progress::new(n_workers as u64));
    let matches = Arc::new(Mutex::new(Vec::<MatchEvent>::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    // 命中即停：`hit_flag` 标记首个命中者（赢家负责打印 [HIT]）。`stop_flag` 由
    // 命中或首次 Ctrl+C 置位，只负责让 worker/heartbeat 停下。
    let hit_flag = Arc::new(AtomicBool::new(false));
    // `sigint_flag` 单独记录「是否已经按过一次 Ctrl+C」——与 `stop_flag` 解耦，
    // 否则命中后按 Ctrl+C 会被误判为「第二次」而 exit(130)，跳过 aman 落盘/保存。
    let sigint_flag = Arc::new(AtomicBool::new(false));

    let sigint_handler = sigint_flag.clone();
    let stop_handler = stop_flag.clone();
    ctrlc::set_handler(move || {
        // First Ctrl+C → request graceful shutdown (hit 置位的 stop_flag 不影响此判断)。
        if sigint_handler
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            stop_handler.store(true, Ordering::SeqCst);
            term_line("[puzzle] Ctrl+C — saving progress and stopping workers …");
        } else {
            // Second Ctrl+C hard-aborts (default behaviour).
            term_line("[puzzle] second Ctrl+C — aborting immediately");
            std::process::exit(130);
        }
    })
    .expect("[puzzle] failed to install SIGINT handler");

    // ── 3. launch worker threads ─────────────────────────────────────────────
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_workers);
    for wid in 0..n_workers {
        handles.push(std::thread::spawn({
            let ctx = ctx.clone();
            let progress = progress.clone();
            let matches = matches.clone();
            let stop_flag = stop_flag.clone();
            let hit_flag = hit_flag.clone();
            move || {
                puzzle_worker(
                    wid as u32,
                    target_h160,
                    ctx,
                    &progress,
                    &matches,
                    &stop_flag,
                    &hit_flag,
                    start,
                    rotate_keys,
                )
            }
        }));
    }

    // ── 3b. GPU worker thread ────────────────────────────────────────────────
    // One additional worker that claims whatever pending chunks the CPU workers
    // haven't taken and scans them with the GPU (100k strided walkers, dense
    // zero-overlap tiling, per-dispatch checkpoint).  Falls back to CPU-only if
    // the selected backend is unavailable.  Rotation is passed through so the
    // GPU worker participates in the same per-claim park-and-rotate strategy as
    // the CPUs.
    let gpu_ctx = ctx.clone();
    let gpu_progress = progress.clone();
    let gpu_matches = matches.clone();
    let gpu_stop = stop_flag.clone();
    let gpu_hit = hit_flag.clone();
    let gpu_handle = std::thread::spawn(move || {
        match framework {
            crate::framework::GpuFramework::WebGpu => gpu_puzzle_worker(
                target_h160,
                gpu_ctx,
                &gpu_progress,
                &gpu_matches,
                &gpu_stop,
                &gpu_hit,
                start,
                gpu_rotate_keys,
            ),
            crate::framework::GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    cuda_puzzle_worker(
                        target_h160,
                        gpu_ctx,
                        &gpu_progress,
                        &gpu_matches,
                        &gpu_stop,
                        &gpu_hit,
                        start,
                        gpu_rotate_keys,
                    );
                }
                #[cfg(not(feature = "cuda"))]
                {
                    term_line("[puzzle] CUDA feature not compiled — running CPU-only.");
                }
            }
            crate::framework::GpuFramework::Auto => {
                // Resolved to a concrete backend in main() before we get here.
                unreachable!("framework Auto must be resolved before puzzle::run")
            }
        }
    });

    // ── 4. heartbeat ticker: status line only (no periodic DB save) ──────────
    let hb_ctx = ctx.clone();
    let hb_stop = stop_flag.clone();
    let hb_progress = progress.clone();
    let hb_handle = std::thread::spawn(move || {
        let mut prev_total = 0u64;
        let mut prev_time = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs_f64(heartbeat_secs));
            if hb_stop.load(Ordering::Relaxed) {
                break;
            }

            let now = Instant::now();
            let total = hb_progress.checked.load(Ordering::Relaxed);
            let dt = now.duration_since(prev_time).as_secs_f64();
            let rate = if dt > 0.1 {
                (total - prev_total) as f64 / dt
            } else {
                0.0
            };
            prev_total = total;
            prev_time = now;

            let s = {
                let ctx = hb_ctx.lock().unwrap_or_else(|e| e.into_inner());

                let summary = chunk_summary(&ctx.file);
                let done_pct = if ctx.file.total_chunks > 0 {
                    summary.2 as f64 / ctx.file.total_chunks as f64 * 100.0
                } else {
                    100.0
                };
                // Chunk indices currently held by a worker (status="running").
                // Cheap to collect here: ticker already holds ctx for the
                // save-throttle check, and the list is tiny (≤ n_workers).
                let running_idxs: Vec<u32> = ctx
                    .file
                    .chunks
                    .iter()
                    .filter(|c| c.status == "running")
                    .map(|c| c.id)
                    .collect();
                let idxs_label = if running_idxs.len() <= 8 {
                    format!("{:?}", running_idxs)
                } else {
                    format!(
                        "{} chunks: {:?} … {:?}",
                        running_idxs.len(),
                        &running_idxs[..4],
                        &running_idxs[running_idxs.len() - 4..]
                    )
                };
                format!(
                    "[puzzle] pend={} running={} done={} ({:.1}%) \
                     idxs={}  keys={}  rate={}/s",
                    summary.0,
                    summary.1,
                    summary.2,
                    done_pct,
                    idxs_label,
                    fmt_comma(total),
                    fmt_comma(rate.round() as u64),
                )
            };
            // In-place status update — rewrites the same line; no newline.
            term_status(&format!("  {s}"));
        }
    });

    // ── 5. join workers, then do one final save ─────────────────────────────
    for h in handles {
        drop(h.join());
    }
    drop(gpu_handle.join()); // GPU worker (no-op if it fell back to CPU-only)
    drop(hb_handle); // ticker sees stop flag on its next tick and exits

    let (final_file, final_matches) = {
        let ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
        let fm = match Arc::try_unwrap(matches) {
            Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        };
        (ctx.file.clone(), fm)
    };

    // 命中即停收尾顺序：终端已打印 [HIT] → 这里先落盘 aman_<TS>.txt → SQLite 最后更新。
    crate::report::flush_match_files(&final_matches, output_dir);
    {
        let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = ctx.save() {
            term_line(&format!("[puzzle] final save failed: {e}"));
        }
    }

    // ── 6. final report ─────────────────────────────────────────────────────
    // `term_line("")` terminates any open in-place status line (if the last
    // output was a claim, it's a no-op) and emits the separating blank line.
    term_line("");
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("  PUZZLE SCAN COMPLETE  (#{})", final_file.puzzle_number);
    eprintln!("──────────────────────────────────────────────────");
    let summary = chunk_summary(&final_file);
    eprintln!("  Target     : {}", final_file.target);
    eprintln!(
        "  Chunks     : {} total — pending {}, running {}, finished {}",
        final_file.total_chunks, summary.0, summary.1, summary.2,
    );
    eprintln!(
        "  Keys       : {}",
        fmt_comma(progress.checked.load(Ordering::Relaxed)),
    );
    eprintln!(
        "  Duration   : {:.2}s ({:.2}m)",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() / 60.0,
    );
    if final_matches.is_empty() {
        eprintln!("  Match      : none");
    } else {
        eprintln!(
            "  Match      : {} event(s) — see aman_<TS>.txt",
            final_matches.len()
        );
        for m in &final_matches {
            let chunk_label = match m.chunk_id {
                Some(id) => format!(" chunk={id}"),
                None => String::new(),
            };
            eprintln!(
                "    worker={}{} idx={} sk_hex={}",
                m.worker_id,
                chunk_label,
                fmt_comma(m.key_index),
                hex::encode(m.private_key),
            );
        }
    }
    eprintln!();

    (progress, final_matches)
}

// ── one worker loop ─────────────────────────────────────────────────────────

/// Reverse-step constants for one worker: the scalar `-1 mod n` and the point
/// `-G`, both derived from `n-1 = CURVE_ORDER - 1`.  Stepping `sk += (n-1)`
/// (via `add_tweak`) and `pk += (-G)` (via `combine`) is the exact mirror of
/// the forward `+1`/`+G` step, at identical cost — so scanning a claimed
/// sub-range in reverse is free.  `pk + (-G) == (sk - 1)·G` keeps the
/// `pk == sk·G` invariant intact while walking downward.
fn reverse_step(
    secp: &secp256k1::Secp256k1<secp256k1::All>,
) -> (secp256k1::Scalar, secp256k1::PublicKey) {
    let n_minus_1: [u8; 32] = {
        let mut b = secp256k1::constants::CURVE_ORDER;
        for i in (0..32).rev() {
            if b[i] > 0 {
                b[i] -= 1;
                break;
            }
            b[i] = 0xFF;
        }
        b
    };
    let neg_one = secp256k1::Scalar::from_be_bytes(n_minus_1)
        .expect("n-1 is a valid scalar (in [1, n-1])");
    let neg_g = secp256k1::PublicKey::from_secret_key(
        secp,
        &secp256k1::SecretKey::from_byte_array(n_minus_1)
            .expect("n-1 is a valid secret key"),
    );
    (neg_one, neg_g)
}

// ── shared chunk-scanning core (local puzzle_worker + remote worker) ─────────

/// Scan direction for a claimed sub-range.  Forward walks `start` upward
/// (`key += 1`); reverse walks `end - 1` downward (`key -= 1`).  Both cover the
/// identical key set `[start, end)` — only the in-order traversal differs, so
/// coverage, zero-dedup, and resume are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScanDir {
    Forward,
    Reverse,
}

/// Resume position for a chunk at a scan checkpoint.  Matches the on-disk
/// semantics: `[current, end)` is always exactly the not-yet-scanned keys.
/// Forward: `current` = next key to scan, `end` unchanged.  Reverse: `end`
/// retreats to `sk + 1`, `current` stays at `start`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResumePosition {
    pub current: [u8; 32],
    pub end: [u8; 32],
}

/// Options for `scan_chunk`: a claimed sub-range plus the shared run state.
pub(crate) struct ScanChunkOptions<'a> {
    pub target_h160: [u8; 20],
    pub puzzle_number: Option<u32>,
    pub worker_id: u32,
    pub chunk_id: Option<u32>,
    /// Inclusive first key of the claimed sub-range.
    pub start: [u8; 32],
    /// Exclusive upper bound of the claimed sub-range.
    pub end: [u8; 32],
    pub dir: ScanDir,
    /// Per-claim rotation budget for wide chunks (`None` = scan to completion).
    pub rotate_keys: Option<u64>,
    pub progress: &'a Progress,
    pub matches: &'a Mutex<Vec<MatchEvent>>,
    pub stop_flag: &'a AtomicBool,
    pub hit_flag: &'a AtomicBool,
    /// Optional early-abort signal, checked on the 2048-key cadence right
    /// after `on_position`.  The remote worker sets it when a heartbeat comes
    /// back 404/409 (lease lost) so the rest of the claim is abandoned within
    /// one cadence instead of being scanned to the rotation budget.  Local
    /// mode passes `None`.
    pub abort_flag: Option<&'a AtomicBool>,
    pub start_elapsed: Instant,
    /// Persistence sink invoked on the 2048-key housekeeping cadence.  The
    /// caller stores the resume position (local: in-memory `PuzzleCtx` +
    /// dirty flag; remote: throttled heartbeat).  Never called on the per-key
    /// hot path.
    pub on_position: &'a mut dyn FnMut(&ResumePosition),
}

/// Result of scanning a claimed sub-range.
pub(crate) struct ScanOutcome {
    /// `true` when the whole `[start, end)` was scanned (or a match ended it).
    pub done: bool,
    /// `true` when a match was recorded during this claim (hit flag is set and
    /// every worker must stop).
    pub matched: bool,
    /// `true` when the start key is outside `[1, n-1]` — the caller should park
    /// the chunk back as pending at its original start.
    pub invalid_start: bool,
    /// Boundary key: forward = the next key to scan (resume `current`);
    /// reverse = the key the scan last advanced past (resume `end` = sk + 1).
    pub sk: [u8; 32],
}

/// Scan one claimed sub-range `[start, end)` with the shared hot path: pubkey
/// derive → hash160 compare → point-add advance.  This is the exact loop the
/// local `puzzle_worker` and the remote worker both run; only the persistence
/// (via `on_position`) and the post-scan finalize differ by caller.
pub(crate) fn scan_chunk(opts: ScanChunkOptions<'_>) -> ScanOutcome {
    let ScanChunkOptions {
        target_h160,
        puzzle_number,
        worker_id,
        chunk_id,
        start,
        end,
        dir,
        rotate_keys,
        progress,
        matches,
        stop_flag,
        hit_flag,
        abort_flag,
        start_elapsed,
        on_position,
    } = opts;

    let reverse = dir == ScanDir::Reverse;

    let secp = secp256k1::Secp256k1::new();
    // Scalar(1) so `sk += 1` per iteration.  Also kept to preserve the SK ↔ PK
    // pairing: sk.add_tweak advances the private key, pk.combine(&G) advances
    // the public key, and they stay in sync because (sk+1)*G == sk*G + G.
    let one = secp256k1::Scalar::from_be_bytes({
        let mut b = [0u8; 32];
        b[31] = 1;
        b
    })
    .expect("scalar 1 is always valid");

    // Generator point G as a compressed PublicKey.  Parsed once per claim —
    // cheap (single affine decode) vs. pay-per-key if re-parsed in the loop.
    let point_g = crate::btc::generator_public_key();

    // Reverse-step constants (−1 mod n, −G) — the exact mirror of the forward
    // `+1`/`+G` step at identical cost, so scanning downward is free.
    let (neg_one, neg_g) = reverse_step(&secp);

    // 初始私钥：正向 = start；反向 = end - 1（区间的最后一个 key）。
    let init_key = if reverse {
        crate::gpu::convert::scalar_sub_be_u64(&end, 1)
    } else {
        start
    };
    let mut sk = match secp256k1::SecretKey::from_byte_array(init_key) {
        Ok(k) => k,
        Err(_) => {
            // The start key is outside [1, n-1]; the caller parks the chunk back.
            return ScanOutcome {
                done: false,
                matched: false,
                invalid_start: true,
                sk: start,
            };
        }
    };
    let mut local_count = 0u64;

    // 每步推进量：正向 +1/+G，反向 -1/-G。方向在 claim 级固定，热路径只用
    // 局部绑定（同一套加法的相反数，成本相同），无每-key 分支。
    let (step_scalar, step_point): (&secp256k1::Scalar, &secp256k1::PublicKey) =
        if reverse { (&neg_one, &neg_g) } else { (&one, &point_g) };

    // Scan budget: `Some(n)` when the sub-range width ≤ ROTATION_BUDGET
    // (scan exactly n keys to completion); `None` when wider (use rotation).
    let budget: Option<u64> = scan_budget(&start, &end);

    // 初始一次完整标量乘 `pk = sk * G`。之后循环不再做标量乘，只用点加
    // `pk = pk + G` 推进 —— 比每步 from_secret_key 便宜 10-20×。
    let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);

    let done = 'scan: loop {
        // ── hot path (exactly mirrors lottery worker_loop) ─────────────
        // Pubkey derive + dhash160 compare.  No branches, no IO, no lock
        // on the per-key critical path — this is what keeps the rate at
        // ~500 kkeys/s.  We only fall through to the boundary check on
        // every 2048th iteration.
        //
        // Note: `pk` is already derived — no per-key scalar mult here.
        let pk_c = pk.serialize();
        let pk_u = pk.serialize_uncompressed();

        if h160_eq(&pk_c, target_h160) || h160_eq(&pk_u, target_h160) {
            let ev = MatchEvent {
                private_key: sk.secret_bytes(),
                compressed: pk_c.to_vec(),
                uncompressed: pk_u.to_vec(),
                worker_id,
                chunk_id,
                key_index: local_count,
                elapsed: start_elapsed.elapsed().as_secs_f64(),
                puzzle_number,
            };
            // 锁中毒也照样保存命中（命中是最珍贵的事件，绝不丢弃）。
            let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
            g.push(ev.clone());
            // 命中即停：首个命中的 worker 立即打印（含私钥）并通知所有 worker 停止。
            if hit_flag
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                term_line(&format!(
                    "[HIT] 🎯 puzzle=#{} worker=#{} chunk={} idx={} sk_hex={}",
                    puzzle_number.map_or_else(String::new, |n| n.to_string()),
                    worker_id,
                    chunk_id.map_or_else(String::new, |c| c.to_string()),
                    fmt_comma(local_count),
                    hex::encode(ev.private_key),
                ));
            }
            stop_flag.store(true, Ordering::SeqCst);
            // 命中即结束本次 claim：调用方负责把该 chunk 标记 finished（本地延迟到
            // 最后一次 save；remote 走 done）。
            return ScanOutcome {
                done: true,
                matched: true,
                invalid_start: false,
                sk: sk.secret_bytes(),
            };
        }

        // ── mutate state ───────────────────────────────────────────────
        // 配对推进：sk +1（标量，mod n，用于断点续传和报告），
        // pk + G（点加，底层 gej_add_ge，零次 doubling）。
        // 两者保持 pk == sk * G 不变式。
        sk = match sk.add_tweak(step_scalar) {
            Ok(next) => next,
            Err(_) => break 'scan false, // scalar overflow → stop
        };
        pk = match pk.combine(step_point) {
            Ok(next) => next,
            // 正向：pk == -G (无穷远点) ⟺ sk == n-1；反向：pk == G ⟺ sk == 1。
            // 两种都等价于 scalar overflow → stop。
            Err(_) => break 'scan false,
        };
        local_count += 1;
        if local_count.is_multiple_of(1000) {
            progress.increment(1000);
        }

        // ── NEW: exact-termination for small chunks (every iteration) ─
        // For budgeted chunks (width ≤ ROTATION_BUDGET) this is the *only*
        // termination that matters and it prevents overshoot past end_bytes
        // between the 2048-cadence checks (a sub-2048-key chunk would otherwise
        // run the hot path 2048 times before the first boundary check).
        if let Some(b) = budget {
            if local_count >= b {
                break 'scan true;
            }
        }

        // ── 2048-cadence housekeeping ─────────────────────────────────
        // Every 2048 iterations we do three cheap things:
        //   1. SIGINT flag — break out if the user pressed Ctrl+C.
        //   2. End-of-range — break out if we've scanned the whole chunk.
        //   3. Refresh in-memory resume position, flushed to the DB at
        //      rotation/finalize (claim end).
        // The release backend compiles the `is_multiple_of` modulo to a
        // single `test` instruction; the predictor lock-stamps the
        // not-taken path so the hot loop stays tight.
        if local_count.is_multiple_of(2048) {
            if stop_flag.load(Ordering::Relaxed) {
                break 'scan false;
            }

            // End-of-range and rotation only apply to *unbudgeted* (wide)
            // chunks.  For budgeted chunks the per-iteration check above
            // already guarantees we stop exactly at the end, so these are
            // skipped to keep rotation from firing on a chunk that fits
            // in one go.
            if budget.is_none() {
                // 越界判定方向相关：正向在 sk 达到 end 时完成（[start, end) 扫完）；
                // 反向在 sk 回落到 start 之下时完成（已从 end-1 一路扫到 start）。
                if reverse {
                    if sk.secret_bytes() < start {
                        break 'scan true;
                    }
                } else if sk.secret_bytes() >= end {
                    break 'scan true;
                }
                // ── rotation (random-subrange mode) ───────────────────────
                // Park the chunk after `rotate_keys` scanned *this claim*
                // and let the worker move on to a fresh random pending
                // chunk.  `local_count` resets every claim, so this is
                // "per-claim budget", not cumulative across re-claims of
                // the same chunk.
                if let Some(rot) = rotate_keys {
                    if local_count >= rot {
                        break 'scan false; // caller parks (release) + re-claims
                    }
                }
            }

            // Refresh resume position + persist via the caller's sink.
            // 正向记录 current = sk（下一个待扫 key）；反向记录 end = sk + 1
            // （未扫描区间 [start, sk+1) 的独占上界），current 保持 start 不动。
            let resume = if reverse {
                ResumePosition {
                    current: start,
                    end: crate::gpu::convert::scalar_add_be(&sk.secret_bytes(), 1),
                }
            } else {
                ResumePosition {
                    current: sk.secret_bytes(),
                    end,
                }
            };
            on_position(&resume);

            // Early abort: the persistence sink (remote heartbeat) may have
            // signalled "give up this claim" (lease lost).  Stop within one
            // cadence instead of scanning to the rotation budget — the hub
            // already reverted the chunk and may have re-issued it.
            if let Some(abort) = abort_flag {
                if abort.load(Ordering::Relaxed) {
                    break 'scan false;
                }
            }
        }
    };

    ScanOutcome {
        done,
        matched: false,
        invalid_start: false,
        sk: sk.secret_bytes(),
    }
}

#[allow(clippy::too_many_arguments)]
fn puzzle_worker(
    wid: u32,
    target_h160: [u8; 20],
    ctx: Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    rotate_keys: Option<u64>,
) {
    // Puzzle number is constant for this worker's lifetime — capture once.
    let puzzle_number = ctx.lock().ok().map(|c| c.file.puzzle_number);

    loop {
        // ── pick + claim a pending chunk (with optional split) ──────────────
        // `claim_random_chunk` picks a random pending chunk, splits it in two
        // (when under the cap and it has >1 key), marks the scanned half
        // "running", and persists.  It returns None only when no claimable
        // pending chunk remains.  Empty chunks are finalized-and-retried
        // internally so the worker never sees a spurious "all done".
        let claim = {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            claim_random_chunk(&mut ctx)
        };

        let claimed = match claim {
            Some(c) => c,
            None => return, // all done
        };
        let (idx, chunk_id, start_bytes, end_bytes) = (claimed.idx, claimed.id, claimed.start, claimed.end);

        // 随机扫描方向（每次 claim 掷一次硬币，不持久化）：正向从 start 向上
        // （key += 1），反向从 end-1 向下（key -= 1）。两个方向扫过的 key 集合
        // 完全相同（[start, end)），只改变遍历顺序 —— 覆盖/零重叠/断点续传不受影响。
        let reverse = pick_random(&[true, false]).copied().unwrap_or(false);

        // Append a log line each time a fresh sub-range is claimed (the only
        // full-line output during steady scanning — heartbeat status rewrites
        // its own line in place).
        term_line(&format!(
            "[claim] w={wid} chunk={chunk_id} range={}..{} dir={}",
            abbr_hex(&start_bytes),
            abbr_hex(&end_bytes),
            if reverse { "REV" } else { "FWD" },
        ));

        let dir = if reverse { ScanDir::Reverse } else { ScanDir::Forward };

        // ── scan the claimed sub-range with the shared core ────────────────
        // The hot path (pubkey derive → hash160 → point-add advance) lives in
        // `scan_chunk`; the 2048-cadence resume-position refresh is routed back
        // into this worker's in-memory chunk + dirty flag, exactly as before.
        let outcome = scan_chunk(ScanChunkOptions {
            target_h160,
            puzzle_number,
            worker_id: wid,
            chunk_id: Some(chunk_id),
            start: start_bytes,
            end: end_bytes,
            dir,
            rotate_keys,
            progress,
            matches,
            stop_flag,
            hit_flag,
            // Local mode has no lease to lose — scan always runs to completion
            // or rotation budget; nothing to abort early.
            abort_flag: None,
            start_elapsed: start,
            on_position: &mut |pos: &ResumePosition| {
                let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
                if reverse {
                    ctx.file.chunks[idx].end_hex = hex_encode_key(&pos.end);
                } else {
                    ctx.file.chunks[idx].current_hex = hex_encode_key(&pos.current);
                }
                ctx.mark_dirty(chunk_id);
            },
        });

        // ── finalize the chunk ──────────────────────────────────────────────
        // On disk right away — the chunk's new status + current_hex must be
        // visible to a future run; this is the sole DB write for the claim
        // (rotation/finalize/empty-chunk paths all sync-flush here).
        {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            let chunk = &mut ctx.file.chunks[idx];

            // 命中即停：chunk 在内存标记 finished + dirty，但不 sync —— SQLite 由
            // 最后一次 ctx.save()（aman 落盘之后）统一落库。
            if outcome.matched {
                chunk.status = "finished".to_string();
                chunk.current_hex = chunk.end_hex.clone();
                ctx.file.completed_chunks += 1;
                ctx.mark_dirty(chunk_id);
                return;
            }

            // Start key outside [1, n-1]: park the chunk back as pending at its
            // original start so a future run can re-claim it.
            if outcome.invalid_start {
                chunk.status = "pending".to_string();
                chunk.current_hex = hex_encode_key(&start_bytes);
                ctx.mark_dirty(chunk_id);
                sync_flush_chunk(&mut ctx);
                continue;
            }

            if outcome.done {
                chunk.status = "finished".to_string();
                chunk.current_hex = chunk.end_hex.clone(); // fully consumed
                ctx.file.completed_chunks += 1;
            } else {
                // either SIGINT or scalar overflow — preserve progress & re-enable
                chunk.status = "pending".to_string();
                if reverse {
                    // 反向停车：current 保持 start（区间左端）不动，end 收缩到下一个
                    // 待扫 key 之后 —— [start, sk+1) 即剩余未扫描部分。`sk` 在 advance
                    // 后已是下一个待扫 key，故 +1 即独占上界。
                    chunk.end_hex = hex_encode_key(&crate::gpu::convert::scalar_add_be(
                        &outcome.sk,
                        1,
                    ));
                } else {
                    chunk.current_hex = hex_encode_key(&outcome.sk);
                }
            }
            // Mark dirty: the next save_dirty flushes just this chunk (plus
            // whatever else is dirty) in one transaction.
            ctx.mark_dirty(chunk_id);
            // 命中即停时（hit_flag 已置位），所有 worker 的 chunk 落库推迟到
            // 最后一次 ctx.save()，保证 aman 落盘先于任何 sqlite 写入。SIGINT
            // 路径（hit_flag=false）保持原样立即 sync。
            if !hit_flag.load(Ordering::Relaxed) {
                sync_flush_chunk(&mut ctx);
            }
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ── GPU worker ──────────────────────────────────────────────────────────────

/// A deterministic sub-range claimed by the GPU worker.  Mirrors the per-chunk
/// view the CPU worker scans (start key, exclusive end, resume position).
struct GpuChunk {
    idx: usize,
    chunk_id: u32,
    start: [u8; 32], // inclusive first key to scan (resume position if any)
    end: [u8; 32],   // exclusive upper bound
}

/// Pick + claim a pending chunk for the GPU worker.  Delegates to the shared
/// `claim_random_chunk` helper (so the GPU participates in random splitting) and
/// maps the result onto a `GpuChunk`.  Returns `None` when no pending chunks
/// remain.
fn gpu_claim_chunk(ctx: &mut PuzzleCtx) -> Option<GpuChunk> {
    let claimed = claim_random_chunk(ctx)?;
    Some(GpuChunk {
        idx: claimed.idx,
        chunk_id: claimed.id,
        start: claimed.start,
        end: claimed.end,
    })
}

/// Finalize a GPU-scanned chunk: either finished (whole range done) or parked
/// (SIGINT) with the current scan position preserved for resume.
///
/// `defer_sync = true`（命中即停时）只标记 dirty，把落库推迟到 run() 最后的
/// `ctx.save()` —— 保证 aman 落盘先于任何 sqlite 写入。
fn gpu_finalize_chunk(ctx: &mut PuzzleCtx, idx: usize, done: bool, current: [u8; 32], defer_sync: bool) {
    let id = ctx.file.chunks[idx].id;
    let chunk = &mut ctx.file.chunks[idx];
    if done {
        chunk.status = "finished".to_string();
        chunk.current_hex = chunk.end_hex.clone();
        ctx.file.completed_chunks += 1;
    } else {
        chunk.status = "pending".to_string();
        chunk.current_hex = hex_encode_key(&current);
    }
    ctx.mark_dirty(id);
    if !defer_sync {
        sync_flush_chunk(ctx);
    }
}

/// Convert a GPU match (`scalar` = winning private key as LE limbs) into the
/// shared `MatchEvent`.  Re-derives the compressed pubkey on the CPU so the
/// output mirrors CPU-worker matches.
fn gpu_match_to_event(
    m: &crate::gpu::GpuMatchOutput,
    chunk: &GpuChunk,
) -> MatchEvent {
    let priv_be = crate::gpu::convert::limbs_to_be_bytes(&m.scalar);
    let secp = secp256k1::Secp256k1::new();
    let pk = secp256k1::SecretKey::from_byte_array(priv_be)
        .and_then(|sk| Ok(secp256k1::PublicKey::from_secret_key(&secp, &sk)))
        .unwrap_or(secp256k1::PublicKey::from_slice(
            &crate::btc::GENERATOR_COMPRESSED,
        )
        .unwrap());
    let compressed = pk.serialize().to_vec();
    let uncompressed = pk.serialize_uncompressed().to_vec();
    // Informational only: offset of the key within the chunk.  Compute it
    // defensively — the GPU scalar is reconstructed from LE limbs and may be
    // off by a hair, so a naive subtraction can underflow (panic).  Saturate
    // to 0 rather than crash on what is purely a report field.
    let key_index = if crate::gpu::convert::be_lt(&priv_be, &chunk.start) {
        0
    } else {
        let diff = crate::gpu::convert::scalar_sub_be(&priv_be, &chunk.start);
        if diff > i64::MAX as u64 {
            0
        } else {
            diff
        }
    };
    MatchEvent {
        private_key: priv_be,
        compressed,
        uncompressed,
        worker_id: 0, // GPU worker id — report distinguishes via key origin; kept 0
        chunk_id: Some(chunk.chunk_id),
        key_index,
        elapsed: 0.0, // filled in by the caller after CPU verification
        puzzle_number: None,
    }
}

/// Minimal backend interface the puzzle GPU scan loop needs.  Implemented by
/// both `GpuScanner` (WebGPU) and `CudaScanner` so the chunk-claim / dispatch /
/// finalize loop is shared instead of duplicated per backend.
trait PuzzleScannerBackend {
    fn seed_range(&mut self, start_be: [u8; 32]) -> anyhow::Result<()>;
    fn step(&mut self) -> anyhow::Result<Vec<crate::gpu::GpuMatchOutput>>;
    fn steps_per_call(&mut self) -> &mut u32;
}

impl PuzzleScannerBackend for crate::gpu::GpuScanner {
    fn seed_range(&mut self, start_be: [u8; 32]) -> anyhow::Result<()> {
        crate::gpu::GpuScanner::seed_range(self, start_be)
    }
    fn step(&mut self) -> anyhow::Result<Vec<crate::gpu::GpuMatchOutput>> {
        crate::gpu::GpuScanner::step(self)
    }
    fn steps_per_call(&mut self) -> &mut u32 {
        &mut self.steps_per_call
    }
}

#[cfg(feature = "cuda")]
impl PuzzleScannerBackend for crate::cuda::CudaScanner {
    fn seed_range(&mut self, start_be: [u8; 32]) -> anyhow::Result<()> {
        crate::cuda::CudaScanner::seed_range(self, start_be)
    }
    fn step(&mut self) -> anyhow::Result<Vec<crate::gpu::GpuMatchOutput>> {
        crate::cuda::CudaScanner::step(self)
    }
    fn steps_per_call(&mut self) -> &mut u32 {
        &mut self.steps_per_call
    }
}

/// One WebGPU puzzle worker thread.  Claims pending chunks from the shared
/// worklist and dense-tiles each `[start, end)` with 100k strided walkers.
/// Runs alongside the CPU workers — all of them pull from the same pending
/// queue, so CPU and GPU share the load without double-scanning.
#[allow(clippy::too_many_arguments)]
fn gpu_puzzle_worker(
    target_h160: [u8; 20],
    ctx: Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    rotate_keys: Option<u64>,
) {
    // Set up GPU.  If no GPU device is available (CI, headless) we log and
    // fall back to CPU-only — never block the whole run on a missing GPU.
    let gpu_ctx = match crate::gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            term_line(&format!("[puzzle] GPU unavailable ({e}) — running CPU-only."));
            return;
        }
    };
    term_line(&format!(
        "[puzzle] GPU worker up on {}",
        gpu_ctx.device_name()
    ));
    let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
    let mut scanner = match crate::gpu::GpuScanner::new(gpu_ctx, &candidates) {
        Ok(s) => s,
        Err(e) => {
            term_line(&format!("[puzzle] GpuScanner::new failed ({e}) — running CPU-only."));
            return;
        }
    };
    // Dense-tiling config: stride = N threads, single target candidate.
    scanner.stride = crate::gpu::NUM_GPU_THREADS;
    scanner.num_candidates = 1;

    puzzle_gpu_scan_loop(
        target_h160,
        scanner,
        &ctx,
        progress,
        matches,
        stop_flag,
        hit_flag,
        start,
        rotate_keys,
        "GPU",
    );
}

/// One CUDA puzzle worker thread — same dense-tiling scan as `gpu_puzzle_worker`
/// but on the CUDA backend.  Falls back to CPU-only when CUDA is unavailable
/// (no device, or the kernel was not compiled — see `CudaScanner::probe`).
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn cuda_puzzle_worker(
    target_h160: [u8; 20],
    ctx: Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    rotate_keys: Option<u64>,
) {
    if !crate::cuda::CudaScanner::probe() {
        term_line("[puzzle] CUDA unavailable — running CPU-only.");
        return;
    }
    let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
    let mut scanner = match crate::cuda::CudaScanner::new(&candidates) {
        Ok(s) => s,
        Err(e) => {
            term_line(&format!("[puzzle] CudaScanner::new failed ({e}) — running CPU-only."));
            return;
        }
    };
    term_line(&format!(
        "[puzzle] CUDA worker up on {}",
        scanner.device_name()
    ));
    // Dense-tiling config: stride = N threads, single target candidate.
    scanner.stride = crate::gpu::NUM_GPU_THREADS;
    scanner.num_candidates = 1;

    puzzle_gpu_scan_loop(
        target_h160,
        scanner,
        &ctx,
        progress,
        matches,
        stop_flag,
        hit_flag,
        start,
        rotate_keys,
        "CUDA",
    );
}

/// Shared GPU scan loop (backend-agnostic): claim a pending chunk, dense-tile
/// it with 100k strided walkers, checkpoint per dispatch, park-and-rotate on
/// `rotate_keys`.  `backend_label` names the worker in log/[HIT] lines.
///
/// When `rotate_keys` is `Some(n)`, the chunk is parked after `n` keys scanned
/// this claim and a fresh random pending chunk is claimed, mirroring the CPU
/// worker's per-claim rotation budget.
#[allow(clippy::too_many_arguments)]
fn puzzle_gpu_scan_loop<S: PuzzleScannerBackend>(
    target_h160: [u8; 20],
    mut scanner: S,
    ctx: &Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    rotate_keys: Option<u64>,
    backend_label: &str,
) {
    // Puzzle number is constant for this worker's lifetime — capture once
    // (mirrors the CPU worker) so the [HIT] line can label it.
    let puzzle_number = ctx.lock().ok().map(|c| c.file.puzzle_number);

    // Rotation budget for this worker.  `Some(n)` parks the current chunk after
    // `n` keys scanned *this claim* (mirrors the CPU worker's `local_count`) and
    // moves on to a fresh random pending chunk; `None` scans to completion.
    // The GPU can saturate throughput quickly, so with a 2^31 budget it rotates
    // every ~3 minutes instead of plowing one huge chunk forever.

    // Per-dispatch coverage (keys).  Constant once calibrated.
    let dispatch_keys = crate::gpu::NUM_GPU_THREADS as u64
        * *scanner.steps_per_call() as u64;

    loop {
        // ── claim a pending chunk ────────────────────────────────────────────
        let claim = {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            gpu_claim_chunk(&mut ctx)
        };
        let chunk = match claim {
            Some(c) => c,
            None => return, // all chunks claimed/finished
        };

        // Append a log line each time a fresh sub-range is claimed, mirroring
        // the CPU workers (heartbeat status keeps rewriting its own line).
        term_line(&format!(
            "[claim] w={backend_label} chunk={} range={}..{}",
            chunk.chunk_id,
            abbr_hex(&chunk.start),
            abbr_hex(&chunk.end),
        ));

        // Seed walkers at `start + i` so they tile [start, start+N) with stride N.
        if scanner.seed_range(chunk.start).is_err() {
            term_line(&format!(
                "[puzzle] {backend_label} seed_range failed — parking chunk"
            ));
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            gpu_finalize_chunk(
                &mut ctx,
                chunk.idx,
                false,
                chunk.start,
                hit_flag.load(Ordering::Relaxed),
            );
            continue;
        }

        let mut current = chunk.start; // next key NOT yet covered
        let mut parked = false;
        let mut scanned_keys: u64 = 0; // keys covered this claim (rotation)
        let mut hit = false; // CPU 验证通过的命中 —— 该 chunk 是赢家 chunk

        // ── scan the chunk in N·steps_per_call-key dispatches ───────────────
        loop {
            // Decide this dispatch's step count.  A full dispatch covers
            // `dispatch_keys` keys; the final (partial) dispatch is trimmed so the
            // walkers land on or just past `end`.  The catch: the chunk width can
            // exceed 2^64 (puzzle #76 spans 2^65), so we MUST NOT compute
            // `end - current` directly (it overflows u64).  Instead we compare
            // `end` against `current + dispatch_keys` (adding the small u64 never
            // overflows) and only subtract once we know the remainder fits.
            let steps = if crate::gpu::convert::be_lt(&current, &chunk.end) {
                let reach = crate::gpu::convert::scalar_add_be(&current, dispatch_keys);
                // `reach >= end`  ⟺  `end - current <= dispatch_keys` (no overflow).
                if !crate::gpu::convert::be_lt(&reach, &chunk.end) {
                    let remaining = crate::gpu::convert::scalar_sub_be(&chunk.end, &current);
                    let n = crate::gpu::NUM_GPU_THREADS as u64;
                    std::cmp::max(1, (remaining + n - 1) / n) as u32
                } else {
                    *scanner.steps_per_call()
                }
            } else {
                0
            };
            if steps == 0 {
                break; // reached the exclusive end
            }
            // 命中即停：别的 worker 命中了，在 dispatch 间隙尽快退出（不再多跑一趟）。
            if stop_flag.load(Ordering::Relaxed) {
                parked = true;
                break;
            }
            // Temporarily set steps_per_call for this (possibly final, partial)
            // dispatch, restoring the default afterwards.
            let saved_steps = *scanner.steps_per_call();
            *scanner.steps_per_call() = steps;
            let batch = crate::gpu::NUM_GPU_THREADS as u64 * steps as u64;
            match scanner.step() {
                Ok(batch_matches) => {
                    if !batch_matches.is_empty() {
                        // Collect CPU-verified matches inside the lock, then drop the
                        // lock before printing ([HIT] is never emitted under it).
                        let verified: Vec<MatchEvent> = {
                            let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                            let mut out = Vec::new();
                            for m in &batch_matches {
                                let mut ev = gpu_match_to_event(m, &chunk);
                                // CPU verification — a real puzzle solver never trusts the
                                // GPU candidate flag alone: re-derive the pubkey, hash160
                                // it, and confirm it equals the target.  Spurious GPU
                                // matches (impossible with a 160-bit hash, but defense in
                                // depth) are dropped here silently.
                                let h = btc::hash160(&ev.compressed);
                                if h == target_h160 {
                                    ev.elapsed = start.elapsed().as_secs_f64();
                                    g.push(ev.clone());
                                    out.push(ev);
                                }
                            }
                            out
                        };
                        if let Some(first) = verified.first() {
                            hit = true;
                            // 命中即停：首个命中的 worker 立即打印（含私钥）并通知
                            // 所有 worker 停止。
                            if hit_flag
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                                .is_ok()
                            {
                                term_line(&format!(
                                    "[HIT] 🎯 puzzle=#{} worker={backend_label} chunk={} sk_hex={}",
                                    puzzle_number.map_or_else(String::new, |n| n.to_string()),
                                    chunk.chunk_id,
                                    hex::encode(first.private_key),
                                ));
                            }
                            stop_flag.store(true, Ordering::SeqCst);
                        }
                    }
                    progress.increment(batch);
                }
                Err(e) => {
                    term_line(&format!(
                        "[puzzle] {backend_label} step failed ({e}) — parking chunk"
                    ));
                    parked = true;
                    break;
                }
            }
            *scanner.steps_per_call() = saved_steps;

            // Advance checkpoint by exactly the keys this dispatch covered.
            current = crate::gpu::convert::scalar_add_be(&current, batch);
            scanned_keys += batch;

            // Refresh the in-memory resume position (persisted at finalize).
            {
                let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
                ctx.file.chunks[chunk.idx].current_hex = hex_encode_key(&current);
                ctx.mark_dirty(chunk.chunk_id);
            }

            if stop_flag.load(Ordering::Relaxed) {
                parked = true;
                break;
            }

            // ── rotation ────────────────────────────────────────────────────
            // Park after `rotate_keys` keys scanned *this claim* and let the
            // loop claim a fresh random pending chunk.  `scanned_keys` resets
            // every claim, so this is a per-claim budget exactly like the CPU
            // worker's `local_count`.  Setting `parked` makes finalize mark the
            // chunk back to "pending" (resume position preserved) instead of
            // "finished", so it is picked up again later.
            if let Some(rot) = rotate_keys {
                if scanned_keys >= rot {
                    parked = true;
                    break;
                }
            }
        }

        // ── finalize ────────────────────────────────────────────────────────
        {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            // 命中 chunk 标记 finished；命中即停时（hit_flag 已置位）推迟落库到
            // 最后一次 ctx.save()（aman 落盘之后）。
            let done = hit || (!parked && !crate::gpu::convert::be_lt(&current, &chunk.end));
            gpu_finalize_chunk(
                &mut ctx,
                chunk.idx,
                done,
                current,
                hit_flag.load(Ordering::Relaxed),
            );
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ── hex helpers ─────────────────────────────────────────────────────────────

/// Parse a hex string as a 32-byte big-endian key.  Odd-length inputs get a
/// leading "0" to make them even; the result is left-padded (high bytes) to a
/// full 64 hex-char / 32-byte key.
///
/// Panics on malformed input (>64 chars, non-hex).  Callers feed it either
/// locally-controlled worklist data or the LAN hub's API responses, both of
/// which are trusted — a malformed key is a programming/hub error we surface
/// loudly rather than silently mis-scan.
pub(crate) fn parse_hex_key(hex_str: &str) -> [u8; 32] {
    let s = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let s = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };
    assert!(
        s.len() <= 64,
        "hex key {} exceeds 64 chars ({} chars)",
        s,
        s.len()
    );
    let padded = format!("{:0>64}", s); // left-pad with zeros to 64 chars
                                    // hex::decode requires an even-length string — guaranteed by:
                                    //   (a) odd inputs get a leading zero above,
                                    //   (b) the resulting length (max 64) is even.
    let raw = hex::decode(padded).unwrap_or_else(|e| panic!("invalid hex key {hex_str}: {e}"));
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    out
}

/// Hex-encode a 32-byte key (always 64 hex chars, no padding ambiguity).
pub(crate) fn hex_encode_key(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

/// Hex of a key with leading zero bytes stripped.  Puzzle sub-ranges sit at
/// 2^(N-1)..2^N, so the high bytes are all zero — showing them (0000…0000) is
/// noise.  This prints the significant part, e.g. 2^70 → "4000000000000000".
pub(crate) fn abbr_hex(k: &[u8; 32]) -> String {
    let s = hex::encode(k);
    let t = s.trim_start_matches('0');
    if t.is_empty() {
        "0".to_string()
    } else {
        t.to_string()
    }
}

/// Parse a 40-character hex string as a 20-byte hash160.  Returns `None` if the
/// string isn't exactly 40 hex chars (the canonical RIPEMD-160 length).
pub(crate) fn hash160_from_hex(s: &str) -> Option<[u8; 20]> {
    if s.len() != 40 {
        return None;
    }
    let raw = hex::decode(s).ok()?;
    if raw.len() != 20 {
        return None;
    }
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Some(out)
}

/// Compute the exclusive upper bound (`start + 2^range_bits`) for a chunk as a
/// 32-byte big-endian key.  `range_bits` must be ≤ 255.
pub(crate) fn chunk_end(start: &[u8; 32], range_bits: u32) -> [u8; 32] {
    assert!(range_bits <= 255, "range_bits {range_bits} out of range");
    let mut result = *start;
    // 2^range_bits in a big-endian [u8; 32]:
    //   byte index (from MSB) = 31 - (range_bits / 8)
    //   bit within that byte (counting from the byte's LSB) = `range_bits % 8`
    let byte_idx = (31 - (range_bits / 8)) as usize;
    let bit_in_byte = (range_bits % 8) as u8;
    let mut carry = (1u16) << bit_in_byte;
    let v = result[byte_idx] as u16 + carry;
    result[byte_idx] = (v & 0xFF) as u8;
    carry = v >> 8;
    let mut i = byte_idx;
    while carry > 0 && i > 0 {
        i -= 1;
        let v = result[i] as u16 + carry;
        result[i] = (v & 0xFF) as u8;
        carry = v >> 8;
    }
    result
}

// ── random-split helpers ─────────────────────────────────────────────────────

/// Convert a `BigUint` to a 32-byte big-endian key, left-padding with zeros.
/// Safe because all values here are < 2^76.
fn biguint_to_32be(v: &BigUint) -> [u8; 32] {
    let bytes = v.to_bytes_be();
    assert!(
        bytes.len() <= 32,
        "key {} exceeds 32 bytes ({} bytes)",
        v,
        bytes.len()
    );
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    out
}

/// Uniform random `BigUint` in `[0, limit)`.  Generates `ceil(bits/8) + 8` random
/// bytes (the extra 8 dilute modulo bias to < 2^-64, negligible for scan
/// distribution) and reduces modulo `limit`.  `limit` must be ≥ 1.
fn random_below(limit: &BigUint) -> BigUint {
    assert!(
        &BigUint::one() <= limit,
        "random_below: limit must be >= 1"
    );
    let nbytes = limit.to_bytes_be().len(); // == ceil(bits/8) for limit > 0
    let mut buf = vec![0u8; nbytes + 8]; // extra bytes remove modulo bias
    rand::rngs::SysRng
        .try_fill_bytes(&mut buf)
        .expect("OS entropy source always available");
    let v = BigUint::from_bytes_be(&buf);
    &v % limit
}

/// Big-endian random `d` strictly between `x` and `y` (i.e. `x < d < y`).
/// Caller guarantees `y - x > 1` (check via `can_split`).
fn random_split_point(x: &[u8; 32], y: &[u8; 32]) -> [u8; 32] {
    let xb = BigUint::from_bytes_be(x);
    let yb = BigUint::from_bytes_be(y);
    let range = &yb - &xb; // BigUint, ≥ 2
    let modulus = &range - BigUint::one(); // range - 1 ≥ 1

    // random offset in [0, range-2], then +1 → offset in [1, range-1]
    let mut offset = random_below(&modulus);
    offset += BigUint::one();

    let d = xb + offset;
    biguint_to_32be(&d)
}

/// Big-endian difference `end - start` as a BigUint.  Precondition: `end >= start`.
fn be_diff(end: &[u8; 32], start: &[u8; 32]) -> BigUint {
    BigUint::from_bytes_be(end) - BigUint::from_bytes_be(start)
}

/// True iff the sub-range `[start, end)` contains more than one key, i.e. a split
/// point strictly between `start` and `end` exists.
fn can_split(start: &[u8; 32], end: &[u8; 32]) -> bool {
    let two = BigUint::from(2u32);
    be_diff(end, start) >= two
}

/// Scan budget for a sub-range.  Returns `Some(n)` when the sub-range width is
/// ≤ ROTATION_BUDGET (scan exactly `n` keys to completion); `None` when it is
/// wider (use the rotation mechanism and ignore the exact width).
fn scan_budget(start: &[u8; 32], end: &[u8; 32]) -> Option<u64> {
    let diff = be_diff(end, start);
    let budget = BigUint::from(ROTATION_BUDGET);
    if diff > budget {
        None
    } else {
        Some(diff.to_u64_digits().first().copied().unwrap_or(0))
    }
}

// ── claim / split result ─────────────────────────────────────────────────────

/// A sub-range claimed by a worker (CPU or GPU).  `start` is the inclusive
/// first key to scan; `end` is the exclusive upper bound.
struct ClaimedChunk {
    idx: usize,
    id: u32,
    start: [u8; 32],
    end: [u8; 32],
}

/// Pick a random pending chunk, optionally split it (when under the cap and it
/// has >1 key), mark the scanned half "running", and persist.  Returns `None`
/// only when no claimable pending chunk exists.  Empty chunks (`current >= end`)
/// are finalized as finished and skipped (the loop retries with another chunk).
fn claim_random_chunk(ctx: &mut PuzzleCtx) -> Option<ClaimedChunk> {
    loop {
        let pendings: Vec<usize> = ctx
            .file
            .chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.status == "pending")
            .map(|(i, _)| i)
            .collect();
        if pendings.is_empty() {
            return None;
        }

        let &idx = pick_random(&pendings)?;

        // Read bounds (copy out to end the borrow before mutating).
        let (cur_hex, end_hex, id) = {
            let c = &ctx.file.chunks[idx];
            (c.current_hex.clone(), c.end_hex.clone(), c.id)
        };
        let cur = parse_hex_key(&cur_hex);
        let end = parse_hex_key(&end_hex);

        // Empty chunk — finalize as finished, persist, and try another.
        if !crate::gpu::convert::be_lt(&cur, &end) {
            let c = &mut ctx.file.chunks[idx];
            c.status = "finished".to_string();
            c.current_hex = end_hex.clone();
            ctx.file.completed_chunks += 1;
            ctx.file.total_chunks = ctx.file.chunks.len() as u32;
            ctx.mark_dirty(id);
            sync_flush_chunk(ctx);
            continue;
        }

        let scan_start = if ctx.file.chunks.len() < MAX_CHUNKS && can_split(&cur, &end) {
            // Split [cur, end) at a random d with cur < d < end.
            // Scan [d, end); park [cur, d) as a fresh pending chunk.
            let d = random_split_point(&cur, &end);

            let new_id = ctx.file.next_id;
            ctx.file.chunks.push(Chunk {
                id: new_id,
                current_hex: hex_encode_key(&cur),
                end_hex: hex_encode_key(&d),
                status: "pending".to_string(),
                chunk_index: 0,
                start_hex: String::new(),
                range_bits: 0,
            });
            ctx.file.next_id = ctx.file.next_id.saturating_add(1);
            ctx.file.total_chunks = ctx.file.chunks.len() as u32;

            let c = &mut ctx.file.chunks[idx];
            c.current_hex = hex_encode_key(&d); // new left bound
            c.status = "running".to_string();

            // Mark BOTH chunks dirty: the updated original (now running, new
            // left bound) and the freshly inserted pending half.  The next
            // save_dirty will INSERT OR REPLACE both in one transaction — so
            // the new sub-range is persisted to the DB *before* we start
            // scanning it.
            ctx.mark_dirty(id);
            ctx.mark_dirty(new_id);
            d
        } else {
            // At cap or too narrow to split: scan [cur, end) directly.
            let c = &mut ctx.file.chunks[idx];
            c.status = "running".to_string();
            ctx.mark_dirty(id);
            cur
        };

        sync_flush_chunk(ctx);
        return Some(ClaimedChunk {
            idx,
            id,
            start: scan_start,
            end,
        });
    }
}

/// Compare a serialised compressed/uncompressed pubkey against the target's
/// 20-byte hash160.  Inlinable, branchless-friendly wrapper around what used
/// to be two separate `hash160` calls on the hot path.
#[inline(always)]
fn h160_eq(pubkey: &[u8], target: [u8; 20]) -> bool {
    btc::hash160(pubkey) == target
}

/// Persist the worklist chunk metadata (status + current_hex) to disk.
/// Used at claim boundaries (claim/split/rotation/finalize) — the only points
/// where the worklist is written, so a Ctrl+C or hard-kill cannot lose more
/// than the current claim's in-flight progress.  Failures are logged but never
/// panic — losing a pending write is preferable to tearing down a worker
/// mid-scan.
#[inline]
fn sync_flush_chunk(ctx: &mut PuzzleCtx) {
    if let Err(e) = ctx.save() {
        term_line(&format!("[puzzle] chunk flush failed: {e}"));
    }
}

/// Pick a random element of a non-empty slice using the OS RNG.  Falls back to
/// the first element of the slice if entropy is unavailable.
pub(crate) fn pick_random<T>(slice: &[T]) -> Option<&T> {
    if slice.is_empty() {
        return None;
    }
    let mut buf = [0u8; 4];
    if rand::rngs::SysRng.try_fill_bytes(&mut buf).is_err() {
        return Some(&slice[0]);
    }
    let r = u32::from_le_bytes(buf) as usize % slice.len();
    Some(&slice[r])
}

/// Return the (pending, running, finished) counts of a worklist.
fn chunk_summary(file: &PuzzleFile) -> (usize, usize, usize) {
    let mut pending = 0usize;
    let mut running = 0usize;
    let mut finished = 0usize;
    for c in &file.chunks {
        match c.status.as_str() {
            "pending" => pending += 1,
            "running" => running += 1,
            "finished" => finished += 1,
            _ => {}
        }
    }
    (pending, running, finished)
}

impl PuzzleCtx {
    /// Incremental save: flush only dirty chunks to the DB.  This is the
    /// hot-path save used by claim, rotation, finalize, ticker, and Ctrl+C.
    fn save(&mut self) -> Result<(), String> {
        save_dirty(self)
    }

    /// Full-sync save (rewrites the entire chunks table).  Escape hatch for
    /// maintenance / compaction — not used on the hot path.
    #[allow(dead_code)]
    fn save_all(&self) -> Result<(), String> {
        save_to_db(&self.db_path, &self.file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── biguint_to_32be round-trips with parse_hex_key ─────────────────────────

    #[test]
    fn test_biguint_to_32be_roundtrip() {
        // Zero
        let z = BigUint::from(0u32);
        assert_eq!(biguint_to_32be(&z), [0u8; 32]);

        // 2^75 (puzzle 76 start)
        let v = BigUint::one() << 75;
        let bytes = biguint_to_32be(&v);
        assert_eq!(parse_hex_key(&hex_encode_key(&bytes)), bytes);

        // Small value
        let v = BigUint::from(0x42u32);
        let bytes = biguint_to_32be(&v);
        assert_eq!(bytes[31], 0x42);
        assert_eq!(bytes[..31], [0u8; 31]);
    }

    #[test]
    fn test_biguint_to_32be_puzzle76_range() {
        // start = 2^75, end = 2^76
        let start = BigUint::one() << 75;
        let end = BigUint::one() << 76;
        let s = parse_hex_key(&hex_encode_key(&biguint_to_32be(&start)));
        let e = parse_hex_key(&hex_encode_key(&biguint_to_32be(&end)));
        assert!(crate::gpu::convert::be_lt(&s, &e));
        // Verify against known hex for 2^75 and 2^76:
        //   2^75 = 0x00...0800...00  (byte[22] = 0x08)
        //   2^76 = 0x00...1000...00  (byte[22] = 0x10)
        assert_eq!(
            hex_encode_key(&s),
            "0000000000000000000000000000000000000000000008000000000000000000"
        );
        assert_eq!(
            hex_encode_key(&e),
            "0000000000000000000000000000000000000000000010000000000000000000"
        );
    }

    // ── scan_budget ────────────────────────────────────────────────────────────

    #[test]
    fn test_scan_budget_small() {
        // [0, 2^27) → Some(2^27)
        let start = [0u8; 32];
        let end = parse_hex_key("8000000"); // 2^27
        assert_eq!(scan_budget(&start, &end), Some(1u64 << 27));
    }

    #[test]
    fn test_scan_budget_at_boundary() {
        // [0, 2^27) → Some(2^27); [0, 2^27 + 1) → None (exceeds budget)
        let start = [0u8; 32];
        let end = parse_hex_key("8000000"); // 2^27
        assert_eq!(scan_budget(&start, &end), Some(1u64 << 27));
        let end = parse_hex_key("8000001"); // 2^27 + 1
        assert_eq!(scan_budget(&start, &end), None);
    }

    #[test]
    fn test_scan_budget_tiny() {
        // [0, 1) → Some(1)
        let start = [0u8; 32];
        let end = parse_hex_key("1"); // 1
        assert_eq!(scan_budget(&start, &end), Some(1));
    }

    #[test]
    fn test_scan_budget_wide_high_bytes_differ() {
        // High bytes differ → width ≥ 2^64 > ROTATION_BUDGET → None
        let start = [0u8; 32];
        // 2^248: byte[0] = 0x01, rest zero → far exceeds ROTATION_BUDGET
        let end = parse_hex_key("0100000000000000000000000000000000000000000000000000000000000000");
        assert_eq!(scan_budget(&start, &end), None);
    }

    // ── can_split ──────────────────────────────────────────────────────────────

    #[test]
    fn test_can_split_adjacent() {
        // [5, 6) → width 1 → cannot split
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        a[31] = 0x05;
        b[31] = 0x06;
        assert!(!can_split(&a, &b));
    }

    #[test]
    fn test_can_split_two_keys() {
        // [5, 7) → width 2 → can split (only d=6)
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        a[31] = 0x05;
        b[31] = 0x07;
        assert!(can_split(&a, &b));
    }

    #[test]
    fn test_can_split_wide() {
        // High bytes differ → definitely can split
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        a[0] = 0x00;
        b[0] = 0x01;
        assert!(can_split(&a, &b));
    }

    #[test]
    fn test_can_split_empty() {
        // [5, 5) → width 0 → cannot split
        let mut a = [0u8; 32];
        a[31] = 0x05;
        assert!(!can_split(&a, &a));
    }

    // ── random_split_point ─────────────────────────────────────────────────────

    #[test]
    fn test_random_split_point_strict_between() {
        // Run many samples to ensure d is always strictly between x and y.
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x[31] = 0x10; // 16
        y[31] = 0x20; // 32
        for _ in 0..1000 {
            let d = random_split_point(&x, &y);
            assert!(
                crate::gpu::convert::be_lt(&x, &d),
                "d ({}) must be > x ({})",
                hex_encode_key(&d),
                hex_encode_key(&x)
            );
            assert!(
                crate::gpu::convert::be_lt(&d, &y),
                "d ({}) must be < y ({})",
                hex_encode_key(&d),
                hex_encode_key(&y)
            );
        }
    }

    #[test]
    fn test_random_split_point_wide_range() {
        // Wide range (2^65) like the initial puzzle 76 chunks.
        let x = parse_hex_key("8000000000000000000");
        let y = parse_hex_key("8020000000000000000"); // x + 2^65
        for _ in 0..100 {
            let d = random_split_point(&x, &y);
            assert!(crate::gpu::convert::be_lt(&x, &d));
            assert!(crate::gpu::convert::be_lt(&d, &y));
        }
    }

    #[test]
    fn test_random_split_point_non_degenerate() {
        // Ensure d != x and d != y (no degenerate split into an empty half).
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x[31] = 0x00;
        y[31] = 0x10;
        for _ in 0..1000 {
            let d = random_split_point(&x, &y);
            assert_ne!(d, x, "split point must not equal x");
            assert_ne!(d, y, "split point must not equal y");
        }
    }

    // ── migration ──────────────────────────────────────────────────────────────

    fn old_format_chunk(chunk_index: u32, start_hex: &str, range_bits: u32, current_hex: Option<&str>) -> Chunk {
        Chunk {
            id: 0,
            current_hex: current_hex.unwrap_or("").to_string(),
            end_hex: String::new(),
            status: "pending".to_string(),
            chunk_index,
            start_hex: start_hex.to_string(),
            range_bits,
        }
    }

    #[test]
    fn test_migrate_basic() {
        let mut file = PuzzleFile {
            puzzle_number: 76,
            total_bits: 75,
            chunk_bits_used: 65,
            total_chunks: 2,
            completed_chunks: 0,
            target: "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF".to_string(),
            hash160: None,
            chunks: vec![
                old_format_chunk(0, "8000000000000000000", 65, None),
                old_format_chunk(1, "8020000000000000000", 65, Some("8020000000000000000")),
            ],
            start_hex: String::new(),
            end_hex: String::new(),
            next_id: 0,
        };

        migrate_puzzle_file(&mut file);

        // Fresh-counter ids start above max(existing ids)=0: chunk 0 → 1, chunk 1 → 2.
        assert_eq!(file.chunks[0].id, 1);
        assert_eq!(file.chunks[0].current_hex, "8000000000000000000");
        assert!(!file.chunks[0].end_hex.is_empty());
        assert_eq!(file.chunks[0].range_bits, 0);
        assert_eq!(file.chunks[0].start_hex, "");
        assert_eq!(file.chunks[0].chunk_index, 0);

        // Chunk 1: current_hex preserved
        assert_eq!(file.chunks[1].id, 2);
        assert_eq!(file.chunks[1].current_hex, "8020000000000000000");

        // end_hex = start + 2^range_bits
        let base0 = parse_hex_key("8000000000000000000");
        let expected_end0 = chunk_end(&base0, 65);
        assert_eq!(
            parse_hex_key(&file.chunks[0].end_hex),
            expected_end0,
            "chunk 0 end_hex mismatch"
        );

        // next_id follows the fresh counter
        assert_eq!(file.next_id, 3);
        assert_eq!(file.total_chunks, 2);

        // File-level range: [2^75, 2^76)
        assert_eq!(
            file.start_hex,
            "0000000000000000000000000000000000000000000008000000000000000000"
        );
        assert_eq!(
            file.end_hex,
            "0000000000000000000000000000000000000000000010000000000000000000"
        );
    }

    #[test]
    fn test_migrate_self_heals_out_of_range_current_hex() {
        // current_hex outside [base, end) → self-heal to base
        let mut file = PuzzleFile {
            puzzle_number: 76,
            total_bits: 75,
            chunk_bits_used: 65,
            total_chunks: 1,
            completed_chunks: 0,
            target: "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF".to_string(),
            hash160: None,
            chunks: vec![old_format_chunk(
                0,
                "8000000000000000000",
                65,
                Some("9999999999999999999"), // out of range
            )],
            start_hex: String::new(),
            end_hex: String::new(),
            next_id: 0,
        };

        migrate_puzzle_file(&mut file);
        assert_eq!(file.chunks[0].current_hex, "8000000000000000000");
    }

    #[test]
    fn test_migrate_idempotent_on_new_format() {
        // A file already in new format should not be altered.
        let mut file = PuzzleFile {
            puzzle_number: 76,
            total_bits: 75,
            chunk_bits_used: 0,
            total_chunks: 1,
            completed_chunks: 0,
            target: "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF".to_string(),
            hash160: None,
            chunks: vec![Chunk {
                id: 42,
                current_hex: "8000000000000000000".to_string(),
                end_hex: "8020000000000000000".to_string(),
                status: "pending".to_string(),
                chunk_index: 0,
                start_hex: String::new(),
                range_bits: 0,
            }],
            start_hex: "0000000000000000000000000000000000000000000008000000000000000000".to_string(),
            end_hex: "0000000000000000000000000000000000000000000010000000000000000000".to_string(),
            next_id: 43,
        };

        migrate_puzzle_file(&mut file);

        assert_eq!(file.chunks[0].id, 42);
        assert_eq!(file.chunks[0].current_hex, "8000000000000000000");
        assert_eq!(file.chunks[0].end_hex, "8020000000000000000");
        assert_eq!(file.next_id, 43); // unchanged
    }

    // ── random_below sanity ────────────────────────────────────────────────────

    #[test]
    fn test_random_below_in_range() {
        let limit = BigUint::from(100u32);
        for _ in 0..1000 {
            let v = random_below(&limit);
            assert!(v < limit);
        }
    }

    #[test]
    fn test_random_below_large() {
        let limit = BigUint::one() << 65; // matches initial chunk width
        for _ in 0..100 {
            let v = random_below(&limit);
            assert!(v < limit);
        }
    }

    // ── SQLite round-trip ──────────────────────────────────────────────────────

    #[test]
    fn test_sqlite_roundtrip() {
        // Build a small PuzzleFile in memory.  Note: `current_hex` uses the
        // short form ("8000…000") that JSON worklists commonly carry; the DB
        // round-trip canonicalises it to full 64-char hex via hex_encode_key.
        // The values are equal modulo parse_hex_key, so we compare parsed keys.
        let file = PuzzleFile {
            puzzle_number: 42,
            total_bits: 75,
            chunk_bits_used: 66,
            total_chunks: 3,
            completed_chunks: 0,
            target: "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa".to_string(),
            hash160: None,
            chunks: vec![
                Chunk { id: 0, current_hex: "8000000000000000000".into(), end_hex: "8000000000000000001".into(), status: "pending".into(), chunk_index: 0, start_hex: String::new(), range_bits: 0 },
                Chunk { id: 1, current_hex: "8000000000000000001".into(), end_hex: "8000000000000000002".into(), status: "running".into(), chunk_index: 0, start_hex: String::new(), range_bits: 0 },
                Chunk { id: 2, current_hex: "8000000000000000002".into(), end_hex: "8000000000000000003".into(), status: "finished".into(), chunk_index: 0, start_hex: String::new(), range_bits: 0 },
            ],
            start_hex: "0000000000000000000000000000000000000000000008000000000000000000".into(),
            end_hex: "0000000000000000000000000000000000000000000010000000000000000000".into(),
            next_id: 3,
        };

        let tmp = std::env::temp_dir().join("luckfind_test_roundtrip.db");
        let _ = std::fs::remove_file(&tmp);

        // Create DB from the file.
        create_db_from_file(&tmp, &file).expect("create_db_from_file");

        // Load it back.
        let loaded = load_from_db(&tmp).expect("load_from_db");

        assert_eq!(loaded.puzzle_number, file.puzzle_number);
        assert_eq!(loaded.total_bits, file.total_bits);
        assert_eq!(loaded.target, file.target);
        assert_eq!(loaded.total_chunks, file.total_chunks);
        assert_eq!(loaded.next_id, file.next_id);
        assert_eq!(loaded.chunks.len(), file.chunks.len());
        for (a, b) in loaded.chunks.iter().zip(file.chunks.iter()) {
            assert_eq!(a.id, b.id);
            // Compare parsed keys (short hex == full hex modulo parse_hex_key).
            assert_eq!(parse_hex_key(&a.current_hex), parse_hex_key(&b.current_hex));
            assert_eq!(parse_hex_key(&a.end_hex), parse_hex_key(&b.end_hex));
            assert_eq!(a.status, b.status);
        }

        // Save again (full sync path) and reload — exercises save_to_db.
        save_to_db(&tmp, &loaded).expect("save_to_db");
        let reloaded = load_from_db(&tmp).expect("load_from_db 2");
        assert_eq!(reloaded.chunks.len(), file.chunks.len());
        assert_eq!(reloaded.chunks[1].status, "running");

        let _ = std::fs::remove_file(&tmp);
    }

    // ── abbr_hex strips leading zero bytes (puzzle ranges sit at 2^(N-1)..2^N) ─

    #[test]
    fn test_abbr_hex_strips_leading_zeros() {
        // 2^70 (puzzle #71 start) → low 8 bytes hold 0x4000…; high bytes zero.
        let mut start = [0u8; 32];
        start[24] = 0x40; // 2^70 = 0x4000_0000_0000_0000 in bytes 24..31
        assert_eq!(abbr_hex(&start), "4000000000000000");

        // 2^71 (puzzle #71 end) = 0x8000_0000_0000_0000.
        let mut end = [0u8; 32];
        end[24] = 0x80;
        assert_eq!(abbr_hex(&end), "8000000000000000");

        // A split point partway into the range keeps its significant digits.
        let mut mid = [0u8; 32];
        mid[24] = 0x5a;
        mid[31] = 0x01;
        assert_eq!(abbr_hex(&mid), "5a00000000000001");

        // All-zero key degrades to "0" rather than an empty string.
        assert_eq!(abbr_hex(&[0u8; 32]), "0");
    }

    // ── reverse-step constants ─────────────────────────────────────────────────

    #[test]
    fn reverse_step_matches_scalar_subtraction() {
        // The reverse hot step must satisfy the same invariant as the forward
        // one, mirrored: `sk·G + (-G) == (sk - 1)·G`, stepping via
        // `add_tweak(n-1)` and `combine(-G)`.
        let secp = secp256k1::Secp256k1::new();
        let (neg_one, neg_g) = reverse_step(&secp);

        // A key in the puzzle key space [2^70, 2^160): 2^70 + 0x13.
        let mut sk_bytes = [0u8; 32];
        sk_bytes[23] = 0x40; // 2^70 = bit 70 → byte 23, bit 6
        sk_bytes[31] = 0x13;
        let sk = secp256k1::SecretKey::from_byte_array(sk_bytes).expect("valid key");

        // Path (a): the reverse hot step — sk·G then combine(-G), tweak (n-1).
        let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
        let pk_minus = pk.combine(&neg_g).expect("sk > 1, so pk - G != infinity");
        let sk_minus = sk.add_tweak(&neg_one).expect("sk > 1, so sk-1 != 0");

        // Path (b): direct scalar multiplication of sk - 1.
        let ref_pk = secp256k1::PublicKey::from_secret_key(&secp, &sk_minus);

        assert_eq!(pk_minus.serialize(), ref_pk.serialize());
        assert_eq!(
            pk_minus.serialize_uncompressed(),
            ref_pk.serialize_uncompressed()
        );

        // Sanity: sk - 1 really is 2^70 + 0x12.
        let mut expect = [0u8; 32];
        expect[23] = 0x40;
        expect[31] = 0x12;
        assert_eq!(sk_minus.secret_bytes(), expect);
    }

    #[test]
    fn reverse_step_neg_one_is_curve_order_minus_one() {
        // n-1 is a valid scalar (in [1, n-1]) and its bytes equal CURVE_ORDER - 1.
        let secp = secp256k1::Secp256k1::new();
        let (neg_one, neg_g) = reverse_step(&secp);
        assert!(neg_one > secp256k1::Scalar::ZERO);

        let n_minus_1: [u8; 32] = {
            let mut b = secp256k1::constants::CURVE_ORDER;
            for i in (0..32).rev() {
                if b[i] > 0 {
                    b[i] -= 1;
                    break;
                }
                b[i] = 0xFF;
            }
            b
        };
        assert_eq!(neg_one.to_be_bytes(), n_minus_1);

        // (n-1)·G == -G: must serialize identically to PublicKey::negate(G).
        let expected = crate::btc::generator_public_key().negate(&secp);
        assert_eq!(neg_g.serialize(), expected.serialize());
    }
}
