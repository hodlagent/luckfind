//! Progress counters (atomic) shared between workers and heartbeat ticker.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct Progress {
    pub checked:       AtomicU64,
    pub workers_alive: AtomicU64,
}

impl Progress {
    pub fn new(n_workers: u64) -> Self {
        Self {
            checked:       AtomicU64::new(0),
            workers_alive: AtomicU64::new(n_workers),
        }
    }

    pub fn increment(&self, n: u64) {
        self.checked.fetch_add(n, Ordering::Relaxed);
    }
}
