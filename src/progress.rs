//! Progress counters (atomic) shared between workers and heartbeat ticker.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct Progress {
    pub checked:       AtomicU64,
    pub workers_alive: AtomicU64,
    // ── pow 会话计数（仅 remote 轨会增量；本地模式恒 0）────────────────────────
    /// 领到 pow 任务的 claim 数。**含重领**（同一窗口弃窗后再领算两次）——它是
    /// "尝试"数，不是窗口数；收敛率看 `pow_verified / pow_attempts`。
    pub pow_attempts: AtomicU64,
    /// 成功收敛（`POST /api/pow` 2xx）的窗口数。
    pub pow_verified: AtomicU64,
    /// 本进程累计 verified keys（各窗口 `keys_scanned` = y−x 之和）。
    pub verified_keys: AtomicU64,
    /// 命中的 proof key 数（同下标重复命中只记一次）。
    pub proofs_found: AtomicU64,
}

impl Progress {
    pub fn new(n_workers: u64) -> Self {
        Self {
            checked:       AtomicU64::new(0),
            workers_alive: AtomicU64::new(n_workers),
            pow_attempts:  AtomicU64::new(0),
            pow_verified:  AtomicU64::new(0),
            verified_keys: AtomicU64::new(0),
            proofs_found:  AtomicU64::new(0),
        }
    }

    pub fn increment(&self, n: u64) {
        self.checked.fetch_add(n, Ordering::Relaxed);
    }

    /// 刚领到一个 pow 任务（claim 的 grant 项带 `task`）。
    pub fn note_pow_claim(&self) {
        self.pow_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// 一个 pow 窗口收敛成功：verified 工作量 = 窗口宽度 `keys`。
    pub fn note_pow_verified(&self, keys: u64) {
        self.pow_verified.fetch_add(1, Ordering::Relaxed);
        self.verified_keys.fetch_add(keys, Ordering::Relaxed);
    }

    /// 扫描命中一个 proof key（调用方保证同一 proof 下标只记一次）。
    pub fn note_proof(&self) {
        self.proofs_found.fetch_add(1, Ordering::Relaxed);
    }
}
