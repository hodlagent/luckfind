//! Candidate puzzle set for BTC collision scanning.
//!
//! Hardcodes the 78 unsolved BTC puzzles as a JSON string literal.  Each puzzle
//! carries its target hash160 and key range `[2^n, 2^(n+1))` so the lottery can
//! generate keys *only* inside the covered key space — a ~2^96× improvement over
//! scanning the full 256-bit space.

use std::sync::OnceLock;

use rand::TryRng;
use rustc_hash::FxHashMap;

// ── Embedded puzzle data ─────────────────────────────────────────────────────
//
// Hardcoded JSON for the 78 unsolved BTC puzzles.  Each record carries the
// target hash160 and the key range `[2^n, 2^(n+1))`.  Sourced from
// `docs/puzzles.json`; regenerate with `make embed-puzzles` if the file changes.

const PUZZLES_JSON: &str = r#"[{"puzzle_number":71,"hex_bytes_len":9,"start_hex":"0x400000000000000000","end_hex":"0x7fffffffffffffffff","hash160":"f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8"},{"puzzle_number":72,"hex_bytes_len":9,"start_hex":"0x800000000000000000","end_hex":"0xffffffffffffffffff","hash160":"bf7413e8df4e7a34ce9dc13e2f2648783ec54adb"},{"puzzle_number":73,"hex_bytes_len":10,"start_hex":"0x1000000000000000000","end_hex":"0x1ffffffffffffffffff","hash160":"105b7f253f0ebd7843adaebbd805c944bfb863e4"},{"puzzle_number":74,"hex_bytes_len":10,"start_hex":"0x2000000000000000000","end_hex":"0x3ffffffffffffffffff","hash160":"9f1adb20baeacc38b3f49f3df6906a0e48f2df3d"},{"puzzle_number":76,"hex_bytes_len":10,"start_hex":"0x8000000000000000000","end_hex":"0xfffffffffffffffffff","hash160":"86f9fea5cdecf033161dd2f8f8560768ae0a6d14"},{"puzzle_number":77,"hex_bytes_len":10,"start_hex":"0x10000000000000000000","end_hex":"0x1fffffffffffffffffff","hash160":"783c138ac81f6a52398564bb17455576e8525b29"},{"puzzle_number":78,"hex_bytes_len":10,"start_hex":"0x20000000000000000000","end_hex":"0x3fffffffffffffffffff","hash160":"35003c3ef8759c92092f8488fca59a042859018c"},{"puzzle_number":79,"hex_bytes_len":10,"start_hex":"0x40000000000000000000","end_hex":"0x7fffffffffffffffffff","hash160":"67671d5490c272e3ab7ddd34030d587738df33da"},{"puzzle_number":81,"hex_bytes_len":11,"start_hex":"0x100000000000000000000","end_hex":"0x1ffffffffffffffffffff","hash160":"351e605fac813965951ba433b7c2956bf8ad95ce"},{"puzzle_number":82,"hex_bytes_len":11,"start_hex":"0x200000000000000000000","end_hex":"0x3ffffffffffffffffffff","hash160":"20d28d4e87543947c7e4913bcdceaa16e2f8f061"},{"puzzle_number":83,"hex_bytes_len":11,"start_hex":"0x400000000000000000000","end_hex":"0x7ffffffffffffffffffff","hash160":"24cef184714bbd030833904f5265c9c3e12a95a2"},{"puzzle_number":84,"hex_bytes_len":11,"start_hex":"0x800000000000000000000","end_hex":"0xfffffffffffffffffffff","hash160":"7c99ce73e19f9fbfcce4825ae88261e2b0b0b040"},{"puzzle_number":86,"hex_bytes_len":11,"start_hex":"0x2000000000000000000000","end_hex":"0x3fffffffffffffffffffff","hash160":"c60111ed3d63b49665747b0e31eb382da5193535"},{"puzzle_number":87,"hex_bytes_len":11,"start_hex":"0x4000000000000000000000","end_hex":"0x7fffffffffffffffffffff","hash160":"fbc708d671c03e26661b9c08f77598a529858b5e"},{"puzzle_number":88,"hex_bytes_len":11,"start_hex":"0x8000000000000000000000","end_hex":"0xffffffffffffffffffffff","hash160":"38a968fdfb457654c51bcfc4f9174d6ee487bb41"},{"puzzle_number":89,"hex_bytes_len":12,"start_hex":"0x10000000000000000000000","end_hex":"0x1ffffffffffffffffffffff","hash160":"5c3862203d1e44ab3af441503e22db97b1c5097e"},{"puzzle_number":91,"hex_bytes_len":12,"start_hex":"0x40000000000000000000000","end_hex":"0x7ffffffffffffffffffffff","hash160":"9978f61b92d16c5f1a463a0995df70da1f7a7d2a"},{"puzzle_number":92,"hex_bytes_len":12,"start_hex":"0x80000000000000000000000","end_hex":"0xfffffffffffffffffffffff","hash160":"6534b31208fe6e100d29f9c9c75aac8bf06fbb38"},{"puzzle_number":93,"hex_bytes_len":12,"start_hex":"0x100000000000000000000000","end_hex":"0x1fffffffffffffffffffffff","hash160":"463013cd41279f2fd0c31d0a16db3972bfffac8d"},{"puzzle_number":94,"hex_bytes_len":12,"start_hex":"0x200000000000000000000000","end_hex":"0x3fffffffffffffffffffffff","hash160":"c6927a00970d0165327d0a6db7950f05720c295c"},{"puzzle_number":96,"hex_bytes_len":12,"start_hex":"0x800000000000000000000000","end_hex":"0xffffffffffffffffffffffff","hash160":"2da63cbd251d23c7b633cb287c09e6cf888b3fe4"},{"puzzle_number":97,"hex_bytes_len":13,"start_hex":"0x1000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffff","hash160":"578d94dc6f40fff35f91f6fba9b71c46b361dff2"},{"puzzle_number":98,"hex_bytes_len":13,"start_hex":"0x2000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffff","hash160":"7eefddd979a1d6bb6f29757a1f463579770ba566"},{"puzzle_number":99,"hex_bytes_len":13,"start_hex":"0x4000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffff","hash160":"c01bf430a97cbcdaedddba87ef4ea21c456cebdb"},{"puzzle_number":101,"hex_bytes_len":13,"start_hex":"0x10000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffff","hash160":"7c1a77205c03b9909663b2034faa0b544e6bc96b"},{"puzzle_number":102,"hex_bytes_len":13,"start_hex":"0x20000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffff","hash160":"f72b812932f6d7102233971d65cec0a22b89e136"},{"puzzle_number":103,"hex_bytes_len":13,"start_hex":"0x40000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffff","hash160":"695fd6dcf33f47166b25de968b2932b351b0afc4"},{"puzzle_number":104,"hex_bytes_len":13,"start_hex":"0x80000000000000000000000000","end_hex":"0xffffffffffffffffffffffffff","hash160":"93022af9a38f3ebb0c3f15dd1c83f8fadaf64e74"},{"puzzle_number":106,"hex_bytes_len":14,"start_hex":"0x200000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffff","hash160":"505aaa63a5e209dfb90cee683a8e227a8c278e47"},{"puzzle_number":107,"hex_bytes_len":14,"start_hex":"0x400000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffff","hash160":"2e644e46b042ffa86da35c54d7275f1abe6d4911"},{"puzzle_number":108,"hex_bytes_len":14,"start_hex":"0x800000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffff","hash160":"b166c44f12c7fc565f37ff6288ee64e0f0ec9a0b"},{"puzzle_number":109,"hex_bytes_len":14,"start_hex":"0x1000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffff","hash160":"aeb0a0197442d4ade8ef41442d557b0e22b85ac0"},{"puzzle_number":111,"hex_bytes_len":14,"start_hex":"0x4000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffff","hash160":"4cfc43fe12a330c8164251e38c0c0c3c84cf86f6"},{"puzzle_number":112,"hex_bytes_len":14,"start_hex":"0x8000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffff","hash160":"4e81efec43c5195aeca0e3877664330418b8e48e"},{"puzzle_number":113,"hex_bytes_len":15,"start_hex":"0x10000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffff","hash160":"ed673389e4b12925316f9166d56d701829e53cf8"},{"puzzle_number":114,"hex_bytes_len":15,"start_hex":"0x20000000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffffff","hash160":"42773005f9594cd16b10985d428418acb7f352ec"},{"puzzle_number":116,"hex_bytes_len":15,"start_hex":"0x80000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffff","hash160":"e3f381c34a20da049779b44cae0417c7fb2898d0"},{"puzzle_number":117,"hex_bytes_len":15,"start_hex":"0x100000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffffff","hash160":"c97f9591e28687be1c4d972e25be7c372a3221b4"},{"puzzle_number":118,"hex_bytes_len":15,"start_hex":"0x200000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffff","hash160":"f4a4e1c11a5bbbd2fc139d221825407c66e0b8b4"},{"puzzle_number":119,"hex_bytes_len":15,"start_hex":"0x400000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffff","hash160":"ae6804b35c82f47f8b0a42d8c5e514fe5ef0a883"},{"puzzle_number":121,"hex_bytes_len":16,"start_hex":"0x1000000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffffff","hash160":"a6e4818537e42f7b3f021daa810367dad4dda16f"},{"puzzle_number":122,"hex_bytes_len":16,"start_hex":"0x2000000000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffffffff","hash160":"e263b62ea294b9650615a13b926e75944c823990"},{"puzzle_number":123,"hex_bytes_len":16,"start_hex":"0x4000000000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffffffff","hash160":"7fa4515066ba6905f894b2078f9af7b1379169cf"},{"puzzle_number":124,"hex_bytes_len":16,"start_hex":"0x8000000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffffff","hash160":"75f74467ce7214f1767406d5ed12012aa523c48e"},{"puzzle_number":126,"hex_bytes_len":16,"start_hex":"0x20000000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffffff","hash160":"683ea8a1ef06eada90556017d44323b5c04e00f1"},{"puzzle_number":127,"hex_bytes_len":16,"start_hex":"0x40000000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffffff","hash160":"a58708aa98ad35c889bb36d8049bf9e9cacfd02a"},{"puzzle_number":128,"hex_bytes_len":16,"start_hex":"0x80000000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffffffff","hash160":"e170ef514689d7230da362a0c121a07723550512"},{"puzzle_number":129,"hex_bytes_len":17,"start_hex":"0x100000000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffffffff","hash160":"ba4c2748360a6b66263e11d1dc8658463ca5ff18"},{"puzzle_number":131,"hex_bytes_len":17,"start_hex":"0x400000000000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffffffffff","hash160":"41b4b36a6c036568972380177eca2916cacd71de"},{"puzzle_number":132,"hex_bytes_len":17,"start_hex":"0x800000000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffffffff","hash160":"cecd3ca4319651bd3afd1e23ab66e111ed38d16d"},{"puzzle_number":133,"hex_bytes_len":17,"start_hex":"0x1000000000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffffffffff","hash160":"014e15e4ea6da460cc7835e262676baa37988e4f"},{"puzzle_number":134,"hex_bytes_len":17,"start_hex":"0x2000000000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffffffff","hash160":"17a5ebfaf62e73f149e33ba674836801f13a80b9"},{"puzzle_number":135,"hex_bytes_len":17,"start_hex":"0x4000000000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffffffff","hash160":"3b6f58a75a54bfd85d1bc6c51180fdc732992326"},{"puzzle_number":136,"hex_bytes_len":17,"start_hex":"0x8000000000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffffffffff","hash160":"05257be4b57ee43fc09762d5d3a9ad4a6e1a0364"},{"puzzle_number":137,"hex_bytes_len":18,"start_hex":"0x10000000000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffffffffff","hash160":"3482f8986e13c018692053a784481c63a3554c9c"},{"puzzle_number":138,"hex_bytes_len":18,"start_hex":"0x20000000000000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffffffffffff","hash160":"692a8e583866fc9056f5c61a45969fb9d868a08c"},{"puzzle_number":139,"hex_bytes_len":18,"start_hex":"0x40000000000000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffffffffffff","hash160":"a45dae9cd5d3fde21e5aa9a95367d107267b3b8a"},{"puzzle_number":140,"hex_bytes_len":18,"start_hex":"0x80000000000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffffffffff","hash160":"ffbb35a7bb9bbe16c1aa2534f7ff11d59c8e3d1a"},{"puzzle_number":141,"hex_bytes_len":18,"start_hex":"0x100000000000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffffffffffff","hash160":"7af50f73fd580f1713af3a6f9c5de49643ec6fc6"},{"puzzle_number":142,"hex_bytes_len":18,"start_hex":"0x200000000000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffffffffff","hash160":"2fcea55e6d027a2ba7c7ebe95eedf47766730fe2"},{"puzzle_number":143,"hex_bytes_len":18,"start_hex":"0x400000000000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffffffffff","hash160":"19ed3e03d19ddcedd5fa86543be820b3a7951650"},{"puzzle_number":144,"hex_bytes_len":18,"start_hex":"0x800000000000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffffffffffff","hash160":"ed87120066e244ff5331d5f8625873d7a3acc39c"},{"puzzle_number":145,"hex_bytes_len":19,"start_hex":"0x1000000000000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffffffffffff","hash160":"5abf369388deb8072741b4eb43ef10fa9388a729"},{"puzzle_number":146,"hex_bytes_len":19,"start_hex":"0x2000000000000000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffffffffffffff","hash160":"dca7ebfb78ce21884300f133d89244bc4b1b756f"},{"puzzle_number":147,"hex_bytes_len":19,"start_hex":"0x4000000000000000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffffffffffffff","hash160":"5318b9d7fcc93873f768725eb68ba2c924bb07ee"},{"puzzle_number":148,"hex_bytes_len":19,"start_hex":"0x8000000000000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffffffffffff","hash160":"a3e3612e586fd206efb8eee6ccd58318e182829a"},{"puzzle_number":149,"hex_bytes_len":19,"start_hex":"0x10000000000000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffffffffffffff","hash160":"7e827e3b90da24c2a15f7b67e3bbece39955a5d0"},{"puzzle_number":150,"hex_bytes_len":19,"start_hex":"0x20000000000000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffffffffffff","hash160":"e08c4d3bc9cf2b3e2cb88de2bfaa4fe8c7aa3f24"},{"puzzle_number":151,"hex_bytes_len":19,"start_hex":"0x40000000000000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffffffffffff","hash160":"1a4fb632f0de0c53a0a31d57f840a19e56c645ee"},{"puzzle_number":152,"hex_bytes_len":19,"start_hex":"0x80000000000000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffffffffffffff","hash160":"da56cd815fa2f0d6a4ce6d25ed7b1a01d9f9bc6b"},{"puzzle_number":153,"hex_bytes_len":20,"start_hex":"0x100000000000000000000000000000000000000","end_hex":"0x1ffffffffffffffffffffffffffffffffffffff","hash160":"4ccf94a1b0efd63cddeee0ef5eee5ebe720cfcbf"},{"puzzle_number":154,"hex_bytes_len":20,"start_hex":"0x200000000000000000000000000000000000000","end_hex":"0x3ffffffffffffffffffffffffffffffffffffff","hash160":"edd2e206825fa8949d1304cd82c08d64b222f2eb"},{"puzzle_number":155,"hex_bytes_len":20,"start_hex":"0x400000000000000000000000000000000000000","end_hex":"0x7ffffffffffffffffffffffffffffffffffffff","hash160":"6b8b7830f73c5bf9e8beb9f161ad82b3bde992e4"},{"puzzle_number":156,"hex_bytes_len":20,"start_hex":"0x800000000000000000000000000000000000000","end_hex":"0xfffffffffffffffffffffffffffffffffffffff","hash160":"9ea3f29aaedf7da10b1488934c50a39e271b0b64"},{"puzzle_number":157,"hex_bytes_len":20,"start_hex":"0x1000000000000000000000000000000000000000","end_hex":"0x1fffffffffffffffffffffffffffffffffffffff","hash160":"242d790e5a168043c76f0539fd894b73ee67b3b3"},{"puzzle_number":158,"hex_bytes_len":20,"start_hex":"0x2000000000000000000000000000000000000000","end_hex":"0x3fffffffffffffffffffffffffffffffffffffff","hash160":"628dacebb0faa7f81670e174ca4c8a95a7e37029"},{"puzzle_number":159,"hex_bytes_len":20,"start_hex":"0x4000000000000000000000000000000000000000","end_hex":"0x7fffffffffffffffffffffffffffffffffffffff","hash160":"2ac1295b4e54b3f15bb0a99f84018d2082495645"},{"puzzle_number":160,"hex_bytes_len":20,"start_hex":"0x8000000000000000000000000000000000000000","end_hex":"0xffffffffffffffffffffffffffffffffffffffff","hash160":"e84818e1bf7f699aa6e28ef9edfb582099099292"}]"#;

/// One parsed puzzle range with precomputed key-generation constants.
#[derive(Debug, Clone)]
pub struct PuzzleRange {
    pub puzzle_number: u32,
    /// Target RIPEMD-160 hash (compressed-pubkey hash; we also check uncompressed).
    pub hash160: [u8; 20],
    /// Number of significant bytes in the range (9..=20).  Only used during
    /// parsing to compute `top_byte_idx`; not read on the hot path.
    #[allow(dead_code)]
    pub hex_bytes_len: u8,
    /// Byte index in a 32-byte big-endian key where the "top byte" sits.
    /// = 32 - hex_bytes_len.  Bytes above this are always 0.
    pub top_byte_idx: u8,
    /// Top byte lower bound (inclusive).
    pub start_top: u8,
    /// Top byte upper bound (exclusive).  A value of 0 encodes the overflow
    /// case where the range ends at 0xFF…FF, so the real upper bound is 0xFF
    /// (inclusive) and the exclusive boundary lives one byte higher.
    pub end_top: u8,
    /// Inclusive start as 32-byte big-endian.
    pub start: [u8; 32],
    /// Exclusive end as 32-byte big-endian.
    pub end: [u8; 32],
}

/// The full embedded puzzle table plus the index structures the lottery needs.
#[derive(Debug, Clone)]
pub struct PuzzleSet {
    pub ranges: Vec<PuzzleRange>,
    /// hash160 → puzzle_number, for match reporting.
    by_hash160: FxHashMap<[u8; 20], u32>,
    /// Weighted-selection lookup: bit position (0..160) → index into `ranges`.
    /// `None` = gap (no puzzle starts at 2^bit).  Entries 0..70 stay None
    /// (no puzzle below 2^70).
    bit_to_range: [Option<u16>; 160],
}

impl PuzzleSet {
    /// Pick a random puzzle index weighted by range size (P ∝ 2^bit).
    /// Re-draws on gap bits (P ≈ 2^-31; effectively never).
    pub fn pick_random_puzzle(&self) -> usize {
        let mut buf = [0u8; 20];
        loop {
            rand::rngs::SysRng
                .try_fill_bytes(&mut buf)
                .expect("OS entropy source always available");
            if buf == [0u8; 20] {
                continue;
            }
            let bit_pos = highest_set_bit_20(&buf);
            if let Some(idx) = self.bit_to_range[bit_pos] {
                return idx as usize;
            }
            // Gap bit → retry.  Probability ~2^-31 per draw.
        }
    }

    /// Look up which puzzle (if any) a hash160 belongs to.
    pub fn puzzle_number_for_hash160(&self, h: &[u8; 20]) -> Option<u32> {
        self.by_hash160.get(h).copied()
    }

    /// Check whether a hash160 matches any embedded puzzle (faster than
    /// `puzzle_number_for_hash160` when you don't need the number).
    pub fn contains(&self, h: &[u8; 20]) -> bool {
        self.by_hash160.contains_key(h)
    }

    /// Generate a uniform random key within a puzzle range (into `buf`).
    /// High bytes (above top_byte_idx) are zeroed; top byte is drawn from
    /// [start_top, end_top) (or [start_top, 0xFF] on overflow); lower bytes
    /// are random.
    pub fn generate_key_in_range(&self, r: &PuzzleRange, buf: &mut [u8; 32]) {
        buf.fill(0);
        let ti = r.top_byte_idx as usize;

        // Top byte
        buf[ti] = if r.end_top == 0 {
            // overflow: [start_top, 0xFF] inclusive
            rand_inclusive(r.start_top, 0xFF)
        } else {
            rand_range(r.start_top, r.end_top)
        };

        // Lower bytes: random
        rand::rngs::SysRng
            .try_fill_bytes(&mut buf[ti + 1..])
            .expect("OS entropy source always available");
    }

    pub fn ranges(&self) -> &[PuzzleRange] {
        &self.ranges
    }

    /// Look up a puzzle range by the bit position of a key's highest set bit.
    /// Returns the range if one starts at `2^bit`, or None if it's a gap.
    pub fn range_for_bit(&self, bit: usize) -> Option<&PuzzleRange> {
        self.bit_to_range.get(bit)
            .and_then(|opt| opt.map(|idx| &self.ranges[idx as usize]))
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

/// Lazily parse the embedded puzzle table.  Cheap after first call.
pub fn puzzle_set() -> &'static PuzzleSet {
    static ONCE: OnceLock<PuzzleSet> = OnceLock::new();
    ONCE.get_or_init(|| parse_puzzle_set(PUZZLES_JSON))
}

// ── Parsing ──────────────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct RawPuzzle {
    puzzle_number: u32,
    hash160: String,
    hex_bytes_len: u8,
    start_hex: String,
    end_hex: String,
}

fn parse_puzzle_set(json: &str) -> PuzzleSet {
    let raws: Vec<RawPuzzle> = serde_json::from_str(json)
        .expect("embedded puzzles.json is valid JSON");

    let mut ranges = Vec::with_capacity(raws.len());
    let mut by_hash160 = FxHashMap::with_capacity_and_hasher(raws.len(), Default::default());
    let mut bit_to_range: [Option<u16>; 160] = [None; 160];

    for raw in &raws {
        let range = parse_range(raw);
        let idx = ranges.len() as u16;

        // Guard: start must be exactly 2^(puzzle_number - 1).  This is critical
        // for the bit-position weighted selection to work correctly.
        assert!(
            is_power_of_two_32(&range.start),
            "puzzle {}: start is not a power of two (JSON data corrupt?)",
            raw.puzzle_number
        );

        ranges.push(range.clone());
        by_hash160.insert(range.hash160, range.puzzle_number);

        let bit = (raw.puzzle_number - 1) as usize;
        bit_to_range[bit] = Some(idx);
    }

    PuzzleSet { ranges, by_hash160, bit_to_range }
}

fn parse_range(raw: &RawPuzzle) -> PuzzleRange {
    let start = parse_hex_key(&raw.start_hex);
    let end_inc = parse_hex_key(&raw.end_hex);
    let end = big_endian_inc(&end_inc);

    let top_byte_idx = 32 - raw.hex_bytes_len;
    let start_top = start[top_byte_idx as usize];
    let end_top = end[top_byte_idx as usize];

    let hash160 = hash160_from_hex(&raw.hash160);

    PuzzleRange {
        puzzle_number: raw.puzzle_number,
        hash160,
        hex_bytes_len: raw.hex_bytes_len,
        top_byte_idx,
        start_top,
        end_top,
        start,
        end,
    }
}

/// Parse a hex string like "0x400000000000000000" into a 32-byte big-endian key.
/// Handles odd-length hex (e.g. "0x1000…000" has 19 hex chars = 9.5 bytes).
fn parse_hex_key(hex: &str) -> [u8; 32] {
    let hex = hex.strip_prefix("0x").unwrap_or(hex);
    // Pad to even length if necessary.
    let hex = if hex.len() % 2 == 1 {
        format!("0{}", hex)
    } else {
        hex.to_string()
    };
    let bytes = hex::decode(&hex).expect("valid hex in puzzles.json");
    assert!(bytes.len() <= 32, "key too large in puzzles.json");
    let mut out = [0u8; 32];
    out[32 - bytes.len()..].copy_from_slice(&bytes);
    out
}

/// 32-byte big-endian increment.  Returns `key + 1`.
fn big_endian_inc(key: &[u8; 32]) -> [u8; 32] {
    let mut out = *key;
    let mut carry = 1u16;
    for i in (0..32).rev() {
        if carry == 0 {
            break;
        }
        let sum = out[i] as u16 + carry;
        out[i] = (sum & 0xff) as u8;
        carry = sum >> 8;
    }
    // carry == 1 only if key was 0xFF…FF (2^256), which never happens here.
    out
}

/// Decode a 40-char hex string into a 20-byte hash160.
fn hash160_from_hex(hex: &str) -> [u8; 20] {
    let bytes = hex::decode(hex).expect("valid hex in puzzles.json hash160");
    assert_eq!(bytes.len(), 20, "hash160 must be 20 bytes");
    let mut out = [0u8; 20];
    out.copy_from_slice(&bytes);
    out
}

/// Find the position of the highest set bit in a 20-byte big-endian number.
/// Returns 0..160.  Assumes `bytes` is not all zeros.
fn highest_set_bit_20(bytes: &[u8; 20]) -> usize {
    for (i, &b) in bytes.iter().enumerate() {
        if b != 0 {
            let bit_in_byte = 7 - b.leading_zeros() as usize;
            return (19 - i) * 8 + bit_in_byte;
        }
    }
    0 // unreachable if bytes != [0; 20]
}

/// Big-endian < comparison for 32-byte keys.
pub fn be_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in 0..32 {
        if a[i] < b[i] { return true; }
        if a[i] > b[i] { return false; }
    }
    false // equal
}

/// Random u8 in [lo, hi) — rejection sampling to avoid modulo bias.
pub fn rand_range(lo: u8, hi: u8) -> u8 {
    debug_assert!(lo < hi);
    let range = (hi - lo) as u16;
    // Rejection sample: accept only values < (256 / range) * range
    let limit = (256u16 / range) * range;
    loop {
        let mut b = [0u8; 1];
        rand::rngs::SysRng.try_fill_bytes(&mut b).expect("entropy");
        let val = b[0] as u16;
        if val < limit {
            return lo + (val % range) as u8;
        }
        // If limit == 0 (range > 256, impossible here since range ≤ 255),
        // fall through and retry.
    }
}

/// Random u8 in [lo, hi] inclusive.
pub fn rand_inclusive(lo: u8, hi: u8) -> u8 {
    debug_assert!(lo <= hi);
    if lo == 0 && hi == 0xFF {
        let mut b = [0u8; 1];
        rand::rngs::SysRng.try_fill_bytes(&mut b).expect("entropy");
        return b[0];
    }
    if hi == 0xFF {
        // [lo, 0xFF] inclusive = [lo, 256) exclusive
        // Use rejection sampling with 2 bytes for enough range.
        let range = (256 - lo as u16) as u32;
        let limit = (65536u32 / range) * range;
        loop {
            let mut b = [0u8; 2];
            rand::rngs::SysRng.try_fill_bytes(&mut b).expect("entropy");
            let val = ((b[0] as u32) << 8) | b[1] as u32;
            if val < limit {
                return lo + (val % range) as u8;
            }
        }
    }
    // [lo, hi] inclusive where hi < 0xFF → [lo, hi+1) exclusive
    rand_range(lo, hi.wrapping_add(1))
}

/// Check if a 32-byte big-endian key is exactly a power of two (single bit set).
fn is_power_of_two_32(key: &[u8; 32]) -> bool {
    let mut seen_bit = false;
    for &b in key.iter() {
        if b == 0 {
            continue;
        }
        if seen_bit {
            return false;
        }
        if b & (b - 1) != 0 {
            return false; // more than one bit in this byte
        }
        seen_bit = true;
    }
    seen_bit // must have seen exactly one bit
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn puzzle_set_has_78() {
        assert_eq!(puzzle_set().len(), 78);
    }

    #[test]
    fn all_hash160_unique() {
        let ps = puzzle_set();
        let mut seen = rustc_hash::FxHashSet::default();
        for r in ps.ranges() {
            assert!(seen.insert(r.hash160), "duplicate hash160 for puzzle {}", r.puzzle_number);
        }
    }

    #[test]
    fn puzzle_numbers_match_bit_positions() {
        let ps = puzzle_set();
        for r in ps.ranges() {
            let bit = r.puzzle_number - 1;
            assert_eq!(
                ps.bit_to_range[bit as usize],
                Some(ps.ranges().iter().position(|x| x.puzzle_number == r.puzzle_number).unwrap() as u16),
                "bit_to_range mismatch for puzzle {}",
                r.puzzle_number
            );
        }
    }

    #[test]
    fn gap_bits_are_none() {
        let ps = puzzle_set();
        let occupied: std::collections::HashSet<usize> = ps.ranges()
            .iter()
            .map(|r| (r.puzzle_number - 1) as usize)
            .collect();
        // The known gaps (solved puzzles): 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130
        for gap in [75usize, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130] {
            assert_eq!(ps.bit_to_range[gap - 1], None, "gap at puzzle {} should be None", gap);
        }
        // All non-gap bits 70..=159 should be Some
        for bit in 70..=159 {
            if !occupied.contains(&bit) {
                assert_eq!(ps.bit_to_range[bit], None, "unoccupied bit {} should be None", bit);
            } else {
                assert!(ps.bit_to_range[bit].is_some(), "occupied bit {} should be Some", bit);
            }
        }
    }

    #[test]
    fn start_is_power_of_two() {
        let ps = puzzle_set();
        for r in ps.ranges() {
            assert!(is_power_of_two_32(&r.start), "puzzle {} start is not power of two", r.puzzle_number);
        }
    }

    #[test]
    fn end_is_exclusive_next_power() {
        let ps = puzzle_set();
        for r in ps.ranges() {
            // end should be start * 2 (i.e., next power of two)
            let mut expected_end = r.start;
            // shift left by 1 (multiply by 2)
            let mut carry = 0u8;
            for i in (0..32).rev() {
                let v = (expected_end[i] << 1) | carry;
                carry = expected_end[i] >> 7;
                expected_end[i] = v;
            }
            if r.end != expected_end {
                eprintln!("puzzle {}: start={:?}", r.puzzle_number, &r.start);
                eprintln!("puzzle {}: end={:?}", r.puzzle_number, &r.end);
                eprintln!("puzzle {}: expected={:?}", r.puzzle_number, &expected_end);
            }
            assert_eq!(r.end, expected_end, "puzzle {}: end != start * 2", r.puzzle_number);
        }
    }

    #[test]
    fn key_generation_within_range() {
        let ps = puzzle_set();
        let mut rng_buf = [0u8; 32];

        for r in ps.ranges() {
            for _ in 0..100 {
                ps.generate_key_in_range(r, &mut rng_buf);
                // key >= start
                assert!(be_gte(&rng_buf, &r.start), "key < start for puzzle {}", r.puzzle_number);
                // key < end
                assert!(be_lt(&rng_buf, &r.end), "key >= end for puzzle {}", r.puzzle_number);
            }
        }
    }

    #[test]
    fn top_byte_respects_bounds() {
        let ps = puzzle_set();
        for r in ps.ranges() {
            let ti = r.top_byte_idx as usize;
            // Generate many keys and check top byte distribution
            let mut rng_buf = [0u8; 32];
            for _ in 0..1000 {
                ps.generate_key_in_range(r, &mut rng_buf);
                let top = rng_buf[ti];
                if r.end_top == 0 {
                    // overflow case: top in [start_top, 0xFF]
                    assert!(top >= r.start_top, "top {} < start_top {} for puzzle {}", top, r.start_top, r.puzzle_number);
                } else {
                    // normal case: top in [start_top, end_top)
                    assert!(top >= r.start_top && top < r.end_top,
                        "top {} not in [{}, {}) for puzzle {}", top, r.start_top, r.end_top, r.puzzle_number);
                }
                // All bytes above top_byte_idx must be 0
                for i in 0..ti {
                    assert_eq!(rng_buf[i], 0, "byte {} not zero for puzzle {}", i, r.puzzle_number);
                }
            }
        }
    }

    #[test]
    fn pick_distribution_chi_squared() {
        let ps = puzzle_set();
        let n_samples = 200_000usize;
        let mut counts = vec![0u64; ps.len()];

        for _ in 0..n_samples {
            let idx = ps.pick_random_puzzle();
            counts[idx] += 1;
        }

        // Expected frequency ∝ 2^(puzzle_number - 1).  Use f64 to avoid overflow
        // for large exponents (puzzle 160 → 2^159).
        let weights: Vec<f64> = ps.ranges().iter()
            .map(|r| 2.0f64.powi((r.puzzle_number - 1) as i32))
            .collect();
        let total_weight: f64 = weights.iter().sum();
        let mut chi_sq = 0.0;
        for (i, w) in weights.iter().enumerate() {
            let expected = n_samples as f64 * w / total_weight;
            if expected > 5.0 {
                let diff = counts[i] as f64 - expected;
                chi_sq += diff * diff / expected;
            }
        }
        // Degrees of freedom ≈ 77, critical value at p=0.001 is ~120.
        // With 200k samples we have good power; allow some slack.
        assert!(chi_sq < 200.0, "chi_sq = {} (too high, distribution biased)", chi_sq);
    }

    // ── Helpers for tests ─────────────────────────────────────────────────────

    fn be_gte(a: &[u8; 32], b: &[u8; 32]) -> bool {
        for i in 0..32 {
            if a[i] > b[i] { return true; }
            if a[i] < b[i] { return false; }
        }
        true // equal
    }
}
