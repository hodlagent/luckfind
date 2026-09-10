//! Remote (LAN-hub) worker mode.
//!
//! `luckfind --remote http://{hub_ip}:42069` turns this machine into a worker
//! for a puzzle worklist owned by a lan-hub (`/Users/jerin/Dev/lan-hub`).  The
//! hub holds the SQLite `.db` and is the single writer; workers never open a
//! local database.  Each worker thread:
//!
//!   1. claims one chunk over HTTP (`POST /api/chunks/claim`),
//!   2. scans it with the shared CPU core (`crate::puzzle::scan_chunk`),
//!   3. keeps the lease alive with throttled heartbeats (30s — comfortably under
//!      the hub's `[reclaim] timeout_seconds`, read from its config.toml) that
//!      also carry the scan position,
//!   4. reports back via `win` (match) / `done` (finished) or `release`
//!      (parked, with the resume position — forward `current`, reverse `end`).
//!
//! Solved signaling: when a worker finds the key it posts `/api/win`; the hub
//! persists the win record and marks the puzzle solved.  That state is
//! broadcast on claim / heartbeat / status responses, and every worker stops
//! promptly when it sees it (mid-claim workers abort on the next heartbeat;
//! idle workers exit on the next claim).
//!
//! Crash recovery is the hub's job: a worker that dies (or loses the network)
//! stops heartbeating, and the hub reclaims the lease after its `[reclaim]
//! timeout_seconds` (read from the hub's config.toml), reverting the
//! chunk to `pending` at the last reported position.  A 404/409 from the hub
//! means our lease is already gone — we abandon the chunk and re-claim, never
//! crashing.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

use crate::btc;
use crate::clientauth;
use crate::config::BtcCheck;
use crate::progress::Progress;
use crate::puzzle::{
    self, abbr_hex, hash160_from_hex, hex_encode_key, parse_hex_key, parse_hex_key_checked,
    scan_chunk,
    term_line, term_status, ProofHit, PuzzleScannerBackend, ResumePosition, ScanChunkOptions,
    ScanDir,
};
use crate::workers::{fmt_comma, MatchEvent};

// Per-claim rotation (reclaim) budgets are resolved config/CLI values threaded
// in from main() — `rotate_keys` (CPU) and `gpu_rotate_keys` (GPU), defaulting
// to 2^27 / 2^31 exactly like local puzzle mode.  Each claim declares the same
// number to the hub as its `capability`, so the hub hands out chunks sized to
// what the worker actually scans before reclaiming.  `None` (a config `0`)
// disables rotation: the chunk is scanned to completion and the claim declares
// capability 0, which the hub treats as its 2^41 default.

/// Lease-refresh cadence.  The hub reclaims leases after `[reclaim]
/// timeout_seconds` (read from the hub's config.toml, not a fixed value), so
/// 30s gives a comfortable margin while keeping LAN traffic negligible.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Sleep between claim attempts when the hub has nothing to hand out (every
/// pending chunk is already running on another worker).
const CLAIM_IDLE: Duration = Duration::from_secs(2);

// ── hub API response shapes (see lan-hub backend/app/routes.py) ──────────────

#[derive(Debug, Deserialize)]
struct HubStatus {
    meta: HubMeta,
    summary: HubSummary,
    #[serde(default)]
    workers: Vec<HubWorker>,
}

#[derive(Debug, Deserialize)]
struct HubMeta {
    puzzle_number: u32,
    target: String,
    hash160: Option<String>,
    /// puzzle 已被某个 worker 命中（hub 落盘 win 记录）。旧 hub 不返回 → false。
    #[serde(default)]
    solved: bool,
}

#[derive(Debug, Deserialize)]
struct HubSummary {
    pending: u64,
    running: u64,
    #[serde(default)]
    finished: u64,
}

#[derive(Debug, Deserialize)]
struct HubWorker {
    #[serde(default)]
    worker_id: String,
    #[serde(default)]
    chunk_count: usize,
    #[serde(default)]
    chunks: Vec<HubWorkerChunk>,
}

#[derive(Debug, Deserialize)]
struct HubWorkerChunk {
    #[serde(default)]
    progress_pct: f64,
}

#[derive(Debug, Deserialize)]
struct ClaimResponse {
    granted: usize,
    #[serde(default)]
    chunks: Vec<ClaimedChunk>,
    /// hub 已 solved：别的 worker 已命中 → 本 worker 应停止。旧 hub 不返回 → false。
    #[serde(default)]
    solved: bool,
}

/// `POST /api/chunks/{id}/heartbeat` 的响应体：`{ok, solved?}`。solved 用于让正在
/// 扫 chunk 的 worker 尽快停止；旧 hub 不返回 solved → 默认 false。
#[derive(Debug, Default, Deserialize)]
struct HeartbeatResp {
    #[serde(default)]
    solved: bool,
}

#[derive(Debug, Deserialize)]
struct ClaimedChunk {
    id: u32,
    current_hex: String,
    end_hex: String,
    /// Phase 2 工作量证明任务（hub `[pow] enabled=true` 且本 client 已登记时随
    /// grant 项下发）。**缺省 = 无 pow 语义**——下面每条分支都跳过，行为与今日
    /// 逐字节一致（旧 hub / 匿名轨 / 窗口过窄都不带此键）。
    ///
    /// 刻意收成 `Value` 而不是 `ChunkTaskRaw`：`task` 在这里是**外层结构的一个
    /// 字段**，直接放 `ChunkTaskRaw` 会让它形状不符时把整个 `ClaimResponse` 的
    /// 反序列化拖垮——`claim()` 返 Err，worker 只能无限 `claim failed — retrying`，
    /// 一块也领不到（hub 侧改个键名即触发）。收成 `Value` 后形状校验挪进
    /// `pow_task()`，最坏后果退回"这一块没 pow"。
    #[serde(default)]
    task: Option<serde_json::Value>,
}

/// claim 响应 grant 项上的原始 `task` 对象（hub `puzzle.py` 的 task_info）。
/// 出现时各字段皆必填；用 `pow_task()` 反序列化 + 校验后才可当 `PowTask` 用。
#[derive(Debug, Deserialize)]
struct ChunkTaskRaw {
    task_id: u64,
    /// 任务窗口底/顶，恒等于该项的 `current_hex`/`end_hex`（hub 同源生成）。
    x_hex: String,
    y_hex: String,
    /// proof 个数（hub `[pow] proof_count`，默认 6——**不得硬编码**）。
    proof_count: usize,
    /// 每个 proof 的压缩公钥 hash160（40-hex）。
    proof_hash160s: Vec<String>,
}

/// 校验过的 pow 任务。由 `ClaimedChunk::pow_task` 构造——畸形任务返回 `None`
/// （调用方按无 pow 语义走并弃窗，绝不 `fatal()`：hub 侧 bug 不该杀掉整机队）。
#[derive(Debug, Clone)]
pub(crate) struct PowTask {
    pub task_id: u64,
    pub x: [u8; 32],
    pub y: [u8; 32],
    pub proof_hash160s: Vec<[u8; 20]>,
}

impl ClaimedChunk {
    /// 校验并解析本项的 pow 任务；`None` = 无任务或任务畸形。
    ///
    /// 校验链：`proof_count >= 1` 且与 `proof_hash160s.len()` 相符 → 每个 hash160
    /// 可解析 → `x < y` → **`x_hex`/`y_hex` == `current_hex`/`end_hex`**（hub 由同一
    /// 对 scan_start/scan_end 生成，不符即协议错位，宁可不挂任务）。
    pub(crate) fn pow_task(&self) -> Option<PowTask> {
        let value = self.task.as_ref()?;
        // 形状校验在这一层做（`task` 是 `Value`）：hub 侧键名/类型漂移只让本块
        // 退回"无 pow"，绝不拖垮整个 claim 响应。
        let raw: ChunkTaskRaw = match serde_json::from_value(value.clone()) {
            Ok(r) => r,
            Err(e) => {
                term_line(&format!(
                    "[pow] chunk {} 任务形状不符（{e}）——按无 pow 处理",
                    self.id
                ));
                return None;
            }
        };
        if raw.proof_count == 0 || raw.proof_hash160s.len() != raw.proof_count {
            term_line(&format!(
                "[pow] chunk {} 任务畸形：proof_count={} 与 proof_hash160s.len()={} 不符——按无 pow 处理",
                self.id,
                raw.proof_count,
                raw.proof_hash160s.len()
            ));
            return None;
        }
        if raw.proof_hash160s.len() + 1 > 78 {
            term_line(&format!(
                "[pow] chunk {} 任务畸形：proof_count={} 超过候选缓冲 78 槽——按无 pow 处理",
                self.id,
                raw.proof_hash160s.len()
            ));
            return None;
        }
        let mut proof_hash160s = Vec::with_capacity(raw.proof_hash160s.len());
        for h in &raw.proof_hash160s {
            match hash160_from_hex(h) {
                Some(v) => proof_hash160s.push(v),
                None => {
                    term_line(&format!(
                        "[pow] chunk {} 任务畸形：proof hash160 {h:?} 非 40-hex——按无 pow 处理",
                        self.id
                    ));
                    return None;
                }
            }
        }
        // 全函数解析：`parse_hex_key` 对畸形输入是 assert + panic（puzzle.rs），
        // 而这里处理的正是**不可信的 hub 响应**——`pow_task()` 的契约是畸形即
        // `None`，不能反过来把进程打死。x/y 恒为 32B 大端 hex（hub
        // `powproof.build_task` 的 `to_bytes(32,"big").hex()`，恒 64 位）。
        let (x, y) = match (
            parse_hex_key_checked(&raw.x_hex),
            parse_hex_key_checked(&raw.y_hex),
        ) {
            (Some(x), Some(y)) => (x, y),
            _ => {
                term_line(&format!(
                    "[pow] chunk {} 任务畸形：任务窗 hex 无法解析（x={:?} y={:?}，需 ≤64 位 hex）\
                     ——按无 pow 处理",
                    self.id, raw.x_hex, raw.y_hex
                ));
                return None;
            }
        };
        if Some(x) != parse_hex_key_checked(&self.current_hex)
            || Some(y) != parse_hex_key_checked(&self.end_hex)
        {
            term_line(&format!(
                "[pow] chunk {} 任务畸形：任务窗 [{}, {}) 与该 grant 项的 [{}, {}) 不符——按无 pow 处理",
                self.id, raw.x_hex, raw.y_hex, self.current_hex, self.end_hex
            ));
            return None;
        }
        if !crate::gpu::convert::be_lt(&x, &y) {
            term_line(&format!(
                "[pow] chunk {} 任务畸形：窗口退化 x >= y——按无 pow 处理",
                self.id
            ));
            return None;
        }
        Some(PowTask {
            task_id: raw.task_id,
            x,
            y,
            proof_hash160s,
        })
    }
}

/// True when the hub says the lease is gone: 404 (no lease on that chunk) or
/// 409 (the chunk is now owned by someone else).  The worker then abandons the
/// chunk and re-claims instead of fighting over it.
fn is_lease_lost(e: &ureq::Error) -> bool {
    matches!(e, ureq::Error::StatusCode(404 | 409))
}

/// 内部错误出口：本地身份 / hub 公钥不自洽、RNG 失败、对端响应信封解不开等。auth 已把
/// 两把 key 校验过（合法曲线点 + TOFU 一致），真走到这里 = 不该裸跑明文继续扫的内部状态
/// 或「对端不是当初 auth 的那个 hub」——带原因 exit 2（与 auth 门禁同级，绝不静默降级）。
fn fatal(msg: impl std::fmt::Display) -> ! {
    eprintln!("[remote] {msg}");
    std::process::exit(2);
}

/// `/api/auth` 失败分类：
/// - `Transport` — 网络层错误（超时/连接断），启动时可像 `connect()` 一样有界重试；
/// - `Denied`    — hub 以 4xx 拒绝（whitelist/blacklist 命中 403、pubkey 非法 422…），
///   直接带 hub 的 `detail` 文案报错；
/// - `Envelope`  — 响应/信封本身坏（非 JSON、缺 auth 键、版本不支持、hex 非法、AES-GCM
///   tag 校验失败）——解不开信封 = 验证不通过，一律 exit 2，绝不降级为无身份扫描。
enum AuthErr {
    Transport(ureq::Error),
    Denied { code: u16, detail: String },
    Envelope(String),
}

/// 4xx 响应体 → 可读文案：优先 FastAPI 的 `{detail: "…"}`，否则截断原始 body。
fn body_detail(resp: ureq::http::Response<ureq::Body>) -> String {
    let mut body = resp.into_body();
    let text = match body.read_to_string() {
        Ok(t) => t,
        Err(_) => return "<响应体不可读>".to_string(),
    };
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) => v
            .get("detail")
            .and_then(|d| d.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| clip(&text)),
        Err(_) => clip(&text),
    }
}

/// 控制台用：超过 ~200 字符截断。
fn clip(s: &str) -> String {
    let t = s.trim();
    if t.chars().count() > 200 {
        format!("{}…", t.chars().take(200).collect::<String>())
    } else {
        t.to_string()
    }
}

// ── blocking HTTP wrapper ────────────────────────────────────────────────────

/// Thin wrapper over `ureq::Agent` for the lan-hub API.  Timeouts are set so a
/// dead hub never hangs a worker thread forever (the hub reclaims the lease
/// after its `[reclaim] timeout_seconds`, read from its config.toml, regardless
/// of whether the worker noticed).
struct HubClient {
    agent: ureq::Agent,
    /// auth 专用 agent：`http_status_as_error(false)`，让 4xx（403 whitelist 拒绝、
    /// 422 pubkey 非法）落到正常 `Response`，从而能读到 hub 返回的 `{detail}` 文案。
    /// 业务请求（claim/heartbeat/…）仍走 `agent`，保留「非 2xx = Err(StatusCode)」语义
    /// （`is_lease_lost` 依赖 404/409）。
    auth_agent: ureq::Agent,
    base: String,
    /// 已 auth 的本地身份（`identity.json` 存在、握手通过后才有）。None = 明文匿名昵称运行
    /// （worker_id = 昵称，业务请求不加密，旧行为）。
    identity: Option<clientauth::ClientIdentity>,
    /// hub 的 Nostr 公钥（解开 auth ECIES 信封拿到、TOFU 校验一致后收下）。与 `identity`
    /// 恒同 Some/None；Some 时业务请求体用它对端派生 pair-key 做 v2 密封、2xx 响应解封。
    hub_pubkey: Option<String>,
}

impl HubClient {
    fn new(base: &str) -> Self {
        let base = base.trim_end_matches('/').to_string();
        // Split the chain so the concrete `ConfigBuilder<AgentScope>` type is
        // pinned before `new_agent` (the timeout builders are generic over the
        // scope and would otherwise leave it ambiguous).
        let builder = || {
            ureq::config::Config::builder()
                .timeout_connect(Some(Duration::from_secs(5)))
                .timeout_per_call(Some(Duration::from_secs(15)))
        };
        let agent = builder().build().new_agent();
        let auth_agent = builder().http_status_as_error(false).build().new_agent();
        Self {
            agent,
            auth_agent,
            base,
            identity: None,
            hub_pubkey: None,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn status(&self) -> Result<HubStatus, ureq::Error> {
        let resp = self.agent.get(&self.url("/api/status")).call()?;
        resp.into_body().read_json()
    }

    /// Phase 1 client-auth：明文 POST 登记本机身份（`identity.json` 的 pubkey），
    /// hub 把 `{hub_pubkey, server_time}` 加密成信封（ECIES v1）返回；这里用本地私钥
    /// + 信封的 `ephemeral_pubkey` 解开（`clientauth::decrypt_auth`）→ 拿到 hub 公钥。
    /// 解得开 = 声明的 pubkey 与本地私钥匹配（验证通过）；hub 拒绝（名单/非法）与信封
    /// 校验失败各自带原因返回，不做重试。
    fn auth(
        &self,
        identity: &clientauth::ClientIdentity,
        name: &str,
    ) -> Result<clientauth::AuthReply, AuthErr> {
        let resp = self
            .auth_agent
            .post(&self.url("/api/auth"))
            .send_json(json!({
                "pubkey": identity.pubkey,
                "name": name,
                "version": 1,
            }))
            .map_err(AuthErr::Transport)?;
        let status = resp.status().as_u16();
        if status != 200 {
            return Err(AuthErr::Denied {
                code: status,
                detail: body_detail(resp),
            });
        }
        let body: serde_json::Value = resp
            .into_body()
            .read_json()
            .map_err(|e| AuthErr::Envelope(format!("auth 响应不是合法 JSON：{e}")))?;
        let env = body
            .get("auth")
            .ok_or_else(|| AuthErr::Envelope("auth 响应缺少 auth 信封".into()))?;
        clientauth::decrypt_auth(identity, env).map_err(AuthErr::Envelope)
    }

    /// 业务请求体密封（client-auth 后所有工作端点共用）：已 auth（`identity`+`hub_pubkey`
    /// 在）→ 把内层明文 body 加密成 v2 信封并包外层 `{"pubkey": <己方>, "enc": …}`；明文
    /// 匿名昵称轨 → 原样返回（旧行为）。hub 只对带 `enc` 的请求回加密响应，所以密封与否
    /// 是这一个开关（wire 中间件按它判定）。
    fn sealed_body(&self, plain: &impl serde::Serialize) -> serde_json::Value {
        let Some(id) = &self.identity else {
            return serde_json::to_value(plain)
                .unwrap_or_else(|e| fatal(format!("序列化业务请求体失败：{e}")));
        };
        let hub_pub = self
            .hub_pubkey
            .as_deref()
            .unwrap_or_else(|| fatal("auth 状态不自洽：有身份文件但缺 hub 公钥".to_string()));
        let bytes = serde_json::to_vec(plain)
            .unwrap_or_else(|e| fatal(format!("序列化业务请求体失败：{e}")));
        let enc = clientauth::encrypt_to_peer(id, hub_pub, &bytes)
            .unwrap_or_else(|e| fatal(format!("密封业务请求失败：{e}")));
        json!({ "pubkey": id.pubkey, "enc": enc })
    }

    /// 2xx 业务响应解封为 JSON Value：auth 模式且 hub 把 body 加密成 `{"enc": v2}` →
    /// 用本地私钥 + hub 公钥解回明文 JSON（**解不开 = 对端不是当初 auth 的那个 hub**，
    /// TOFU 违背 → exit 2，绝不把进度/命中状态当明文裸读）；明文旧 hub / 匿名模式 → 原样。
    fn unseal_response(&self, resp: ureq::http::Response<ureq::Body>) -> Result<serde_json::Value, ureq::Error> {
        let body: serde_json::Value = resp.into_body().read_json()?;
        Ok(match (&self.identity, &self.hub_pubkey) {
            (Some(id), Some(hub_pub)) => match body.get("enc") {
                // 信封在就必须解得开（hub 只会给已登记 client 回密文）
                Some(enc) => {
                    let plain = clientauth::decrypt_from_peer(id, hub_pub, enc)
                        .unwrap_or_else(|e| fatal(format!("hub 业务响应信封解密失败：{e}")));
                    serde_json::from_slice(&plain)
                        .unwrap_or_else(|e| fatal(format!("响应解密后不是合法 JSON：{e}")))
                }
                None => body, // 明文旧 hub（无 wire 中间件）→ 直通
            },
            _ => body,
        })
    }

    /// Claim `count` pending chunks, declaring how many keys this worker scans
    /// before it reclaims (`capability`).  The hub uses the capability to prefer
    /// handing out chunks it won't have to split mid-scan; 0 = hub default 2^41.
    fn claim(
        &self,
        worker_id: &str,
        count: usize,
        capability: u64,
    ) -> Result<ClaimResponse, ureq::Error> {
        let body = self.sealed_body(&json!({
            "worker_id": worker_id,
            "count": count,
            "capability": capability
        }));
        let resp = self
            .agent
            .post(&self.url("/api/chunks/claim"))
            .send_json(body)?;
        let value = self.unseal_response(resp)?;
        serde_json::from_value(value).map_err(ureq::Error::Json)
    }

    /// Heartbeat。Ok(true) 表示 hub 已 solved（别的 worker 命中）——调用方应尽快
    /// 停止本 claim。信封解不开会 exit 2（见 `unseal_response`）；**明文**旧 hub 只回
    /// `{ok:true}`，其响应体解析失败按未 solved 处理（旧行为，`read_json` 容忍路径）。
    fn heartbeat(
        &self,
        chunk_id: u32,
        worker_id: &str,
        current_hex: Option<String>,
        end_hex: Option<String>,
        keys: Option<u64>,
        rate: Option<f64>,
    ) -> Result<bool, ureq::Error> {
        let body = ChunkUpdateBody {
            worker_id: worker_id.to_string(),
            current_hex,
            end_hex,
            keys,
            rate,
        };
        let sealed = self.sealed_body(&body);
        let resp = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/heartbeat")))
            .send_json(sealed)?;
        // read_json 失败（明文旧 hub 的非对象体）只影响 solved 读取 → 按未 solved 处理；
        // auth 模式的信封已在 unseal_response 内强制解封，不会走到这里的默认值。
        let value = self.unseal_response(resp).unwrap_or(serde_json::Value::Null);
        let parsed: HeartbeatResp = serde_json::from_value(value).unwrap_or_default();
        Ok(parsed.solved)
    }

    fn done(&self, chunk_id: u32, worker_id: &str) -> Result<(), ureq::Error> {
        let body = self.sealed_body(&json!({ "worker_id": worker_id }));
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/done")))
            .send_json(body)?;
        Ok(())
    }

    /// 命中上报（取代 done）：hub 落 win 记录 + 置 puzzle solved。409（已被别的
    /// worker 先标记 solved）按 lease 丢失处理即可——本 worker 放弃本 chunk。
    fn win(&self, chunk_id: u32, worker_id: &str) -> Result<(), ureq::Error> {
        let body = self.sealed_body(&json!({ "worker_id": worker_id, "chunk_id": chunk_id }));
        let _ = self
            .agent
            .post(&self.url("/api/win"))
            .send_json(body)?;
        Ok(())
    }

    fn release(
        &self,
        chunk_id: u32,
        worker_id: &str,
        current_hex: Option<String>,
        end_hex: Option<String>,
    ) -> Result<(), ureq::Error> {
        let body = ChunkUpdateBody {
            worker_id: worker_id.to_string(),
            current_hex,
            end_hex,
            keys: None,
            rate: None,
        };
        let sealed = self.sealed_body(&body);
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/release")))
            .send_json(sealed)?;
        Ok(())
    }

    /// `POST /api/pow` —— pow 窗口的收敛点（取代该窗口上的 done）。hub 校验全窗
    /// digest 后按任务窗口终态化 chunk 并记 verified 工作量。
    ///
    /// 经 `sealed_body` → auth 轨自动包 v2 信封（与 claim/heartbeat/done 同款）。
    /// 用 `self.agent`（非 auth_agent）以便非 2xx 保持 `Err(StatusCode)`——
    /// `is_lease_lost` 照常把 404（task 没了）/ 409（task 非 active / 非本人 /
    /// lease 已丢）判成"放弃重领"；400 = digest 不符（task 仍 active，可重扫重试，
    /// 但真实 hash160 比对不可能对错，故调用方按客户端/hub bug 处理）。
    /// 400 的 `{detail}` 在响应信封里，这里只能拿到状态码——日志记状态码即可。
    fn pow(
        &self,
        worker_id: &str,
        task_id: u64,
        chunk_id: u32,
        hashed_proof_key: &str,
    ) -> Result<PowResp, ureq::Error> {
        let body = self.sealed_body(&json!({
            "worker_id": worker_id,
            "task_id": task_id,
            "chunk_id": chunk_id,
            "hashed_proof_key": hashed_proof_key,
        }));
        let resp = self
            .agent
            .post(&self.url("/api/pow"))
            .send_json(body)?;
        Ok(resp.into_body().read_json::<PowResp>().unwrap_or_default())
    }
}

/// `POST /api/pow` 的 200 响应体：`{ok, task_id, chunk_id, keys_scanned, solved}`。
/// 全字段 `#[serde(default)]`——只有 `keys_scanned`/`solved` 会被用到，旧/新 hub
/// 增删字段都不该让收敛失败。`ok`/`task_id`/`chunk_id` 是线上协议形状的一部分
/// （单测读它们做解析回归），生产代码不据此分支。
#[derive(Debug, Default, Deserialize)]
#[allow(dead_code)]
struct PowResp {
    #[serde(default)]
    ok: bool,
    #[serde(default)]
    task_id: u64,
    #[serde(default)]
    chunk_id: u32,
    #[serde(default)]
    keys_scanned: u64,
    #[serde(default)]
    solved: bool,
}

/// `HashedProofKey = SHA256(升序 32B-BE concat)` 小写 hex —— 镜像 hub
/// `backend/app/powproof.py::expected_digest_for`。
///
/// **升序是关键**：proof key 由扫描命中顺序决定（正扫/反扫、命中先后都不同），
/// digest 必须与顺序无关，故先按无符号整数升序排好再 concat。⚠️ 姊妹实现
/// `vanity_worker_bpi.py::compute_proof_hash` 是**按池端数组顺序拼 64-hex ASCII**
/// ——那是另一套契约，照抄必然 400（`scripts/test_pow.py` 的 `[1,2,3]` 向量是护栏）。
pub(crate) fn hashed_proof_key(keys: &[[u8; 32]]) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<&[u8; 32]> = keys.iter().collect();
    sorted.sort_unstable_by(|a, b| {
        if crate::gpu::convert::be_lt(a, b) {
            std::cmp::Ordering::Less
        } else if crate::gpu::convert::be_lt(b, a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let mut h = Sha256::new();
    for k in sorted {
        h.update(k);
    }
    hex::encode(h.finalize())
}

/// pow 窗口的**唯一收敛点**（CPU / GPU 报告链共用，避免两条路径漂移）。
///
/// `found` 是扫描期间攒下的 proof 命中（`ProofHit{index, key}`）。先按 proof 下标
/// 归位（同一下标重复命中只算一次），集齐 N 个即算全窗 digest 提交 `/api/pow`：
///
/// - 提交成功 → 打印 verified 工作量；`resp.solved`（puzzle 已被别人解出）⇒ 返回
///   `false`（调用方停机），否则 `true`（继续 claim）。
/// - 404/409（`is_lease_lost`：task 不存在 / 非 active / 非本人 / lease 已丢）⇒
///   `true`——任务已作废，直接重领。
/// - 400（digest 不符）或其它传输错 ⇒ 大声警告 + `release` 弃窗 + `true`。
///   **刻意不重试**：hub 侧 400 保持 task active、租约内可重扫后重传（`puzzle.py`
///   `verify_pow`），但客户端在真实 hash160 比对下不该凑不出 digest——真凑不出
///   就说明本地扫描口径与 hub 下发的 proof 集不符，重传多少次都一样。弃窗重扫
///   （拿一份新任务）> 抱着一个永远收敛不了的窗口空转。
/// - **未集齐**（扫描被中断，或 hub 的 proof 集在本地根本不可能命中）⇒ 限流警告 +
///   `release` 弃窗 + `true`：无法收敛，立刻还回 pending 胜过等 hub reclaim 超时。
fn converge_pow(
    client: &HubClient,
    worker_id: &str,
    chunk_id: u32,
    task: &PowTask,
    found: &[ProofHit],
    backend: &str,
) -> bool {
    let need = task.proof_hash160s.len();
    // 按 proof 下标归位：proof 是**集合**语义（digest 升序后与命中顺序无关），
    // 同一下标重复命中不重复计数。
    let mut slots: Vec<Option<[u8; 32]>> = vec![None; need];
    for hit in found {
        if let Some(slot) = slots.get_mut(hit.index) {
            if slot.is_none() {
                *slot = Some(hit.key);
            }
        }
    }
    let collected = slots.iter().filter(|s| s.is_some()).count();

    if collected < need {
        term_line(&format!(
            "[pow] {backend} w={worker_id} chunk={chunk_id} 只集齐 {collected}/{need} 个 \
             proof key —— 弃窗（release，无部分 credit）"
        ));
        if let Err(e) = client.release(chunk_id, worker_id, None, None) {
            if !is_lease_lost(&e) {
                term_line(&format!("[pow] release failed ({e}) — hub will reclaim the chunk"));
            }
        }
        return true;
    }

    let keys: Vec<[u8; 32]> = slots.into_iter().flatten().collect();
    let hashed = hashed_proof_key(&keys);
    match client.pow(worker_id, task.task_id, chunk_id, &hashed) {
        Ok(resp) => {
            term_line(&format!(
                "[pow] {backend} w={worker_id} chunk={chunk_id} task={} 收敛 ✅ verified {} keys",
                task.task_id,
                fmt_comma(resp.keys_scanned),
            ));
            !resp.solved
        }
        Err(e) if is_lease_lost(&e) => {
            term_line(&format!(
                "[pow] {backend} chunk={chunk_id} task={} 提交被拒（{e}）—— 任务已作废，重新 claim",
                task.task_id
            ));
            true
        }
        Err(e) => {
            term_line(&format!(
                "[pow] {backend} chunk={chunk_id} task={} digest 提交失败（{e}）—— \
                 客户端/hub 口径不符？弃窗重扫",
                task.task_id
            ));
            if let Err(e2) = client.release(chunk_id, worker_id, None, None) {
                if !is_lease_lost(&e2) {
                    term_line(&format!(
                        "[pow] release failed ({e2}) — hub will reclaim the chunk"
                    ));
                }
            }
            true
        }
    }
}

/// `{worker_id, current_hex?, end_hex?, keys?, rate?}` — only present fields are
/// sent (the hub's Pydantic model accepts either, `end_hex` being the new
/// reverse-park field; `keys`/`rate` are transient metrics the hub caches in
/// memory, and older hubs simply ignore the unknown keys).
#[derive(serde::Serialize)]
struct ChunkUpdateBody {
    worker_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keys: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate: Option<f64>,
}

// ── entry point ──────────────────────────────────────────────────────────────

pub fn run(
    remote_url: &str,
    nickname: String,
    identity_path: &Path,
    n_workers: usize,
    heartbeat_secs: f64,
    output_dir: Option<&Path>,
    framework: crate::framework::GpuFramework,
    gpu_available: bool,
    rotate_keys: Option<u64>,
    gpu_rotate_keys: Option<u64>,
    check: BtcCheck,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    let mut client = HubClient::new(remote_url);

    // ── 1. connect to the hub and read the puzzle meta ─────────────────────
    let (target_h160, puzzle_number, summary) = connect(&client);

    // ── 1b. client-auth gate（identity.json 在才走；缺 → 明文匿名昵称运行）────
    // claim 之前先声明身份：明文 POST 登记 + 解开 hub 的加密信封拿 hub_pubkey
    // （`maybe_authenticate`），并把 hub 公钥 TOFU 进 hub.json、身份收进 client——
    // 此后 claim/heartbeat/done/win/release 的请求体自动 v2 密封、响应解封。验证不过
    // 直接 exit 2；**文件在 = 期望 auth，绝不静默降级为明文扫描**；文件缺失才明文匿名
    // 昵称运行（用户 2026-09 拍板：opt-in by omission）。
    //
    // worker_id 双语义（hub docs/remote-api.md §1）：auth 轨把 `nickname`（旧自由文本
    // worker_id，--worker-id / 主机名）只作为 auth 的 `name` 上报供 hub UI 显示，干活端点
    // （claim/heartbeat/done/win/release）的 worker_id 一律用本机身份 pubkey（auth 轨记账）；
    // 无 identity.json 则 worker_id = 昵称，走匿名轨（零账本，旧行为不变）。
    let worker_id: String = maybe_authenticate(&mut client, identity_path, &nickname)
        .unwrap_or_else(|| nickname);
    let client = Arc::new(client);

    if summary.pending + summary.running == 0 {
        println!("[remote] hub reports no pending or running chunks — nothing to do.");
        return (Arc::new(Progress::new(0)), Vec::new());
    }

    // ── 2. shared state + SIGINT handler (mirrors puzzle.rs) ───────────────
    // GPU worker count for the heartbeat's alive counter: WebGPU = 1 worker,
    // CUDA = one thread per physical device (mainboard + discrete cards).
    let gpu_worker_count = if gpu_available {
        match framework {
            crate::framework::GpuFramework::WebGpu => 1,
            crate::framework::GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaScanner::device_count() as usize
                }
                #[cfg(not(feature = "cuda"))]
                {
                    0
                }
            }
            crate::framework::GpuFramework::Auto => 0, // resolved before remote::run
        }
    } else {
        0
    };
    let progress = Arc::new(Progress::new((n_workers + gpu_worker_count) as u64));
    let matches = Arc::new(Mutex::new(Vec::<MatchEvent>::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    // 命中即停：hit_flag 记录首个命中者（赢家负责打印 [HIT]）；stop_flag 由
    // 命中或首次 Ctrl+C 置位，只负责让 worker/ticker 停下。
    let hit_flag = Arc::new(AtomicBool::new(false));
    // sigint_flag 区分「第一次 Ctrl+C」（优雅退出）与第二次（立即 abort）。
    let sigint_flag = Arc::new(AtomicBool::new(false));

    let sigint_handler = sigint_flag.clone();
    let stop_handler = stop_flag.clone();
    ctrlc::set_handler(move || {
        if sigint_handler
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            stop_handler.store(true, Ordering::SeqCst);
            term_line("[remote] Ctrl+C — releasing chunks and stopping workers …");
        } else {
            term_line("[remote] second Ctrl+C — aborting immediately");
            std::process::exit(130);
        }
    })
    .expect("[remote] failed to install SIGINT handler");

    // ── 3. worker threads + status ticker ──────────────────────────────────
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_workers);
    for wid in 0..n_workers {
        let client = client.clone();
        let worker_id = worker_id.clone();
        let progress = progress.clone();
        let matches = matches.clone();
        let stop_flag = stop_flag.clone();
        let hit_flag = hit_flag.clone();
        handles.push(std::thread::spawn(move || {
            remote_worker(
                wid as u32,
                &client,
                &worker_id,
                target_h160,
                puzzle_number,
                &progress,
                &matches,
                &stop_flag,
                &hit_flag,
                rotate_keys,
                start,
                check,
            )
        }));
    }

    let ticker_client = client.clone();
    let ticker_id = worker_id.clone();
    let ticker_stop = stop_flag.clone();
    let ticker_progress = progress.clone();
    let ticker_handle = std::thread::spawn(move || {
        ticker(
            &ticker_client,
            &ticker_id,
            &ticker_progress,
            &ticker_stop,
            heartbeat_secs,
        )
    });

    // ── 3b. GPU worker thread(s) ─────────────────────────────────────────────
    // One worker per physical CUDA device (or a single WebGPU worker) that
    // claims pending chunks from the hub and scans them with the GPU (100k
    // strided walkers, dense zero-overlap tiling, heartbeats + release-at-
    // rotation for resume).  Runs alongside the CPU workers, all pulling from
    // the same hub claim pool.  Skipped when main() resolved no usable GPU
    // device (framework stays as resolved, never Auto).
    let mut gpu_handles: Vec<_> = Vec::new();
    if gpu_available {
        match framework {
            crate::framework::GpuFramework::WebGpu => {
                let client = client.clone();
                let worker_id = worker_id.clone();
                let progress = progress.clone();
                let matches = matches.clone();
                let stop_flag = stop_flag.clone();
                let hit_flag = hit_flag.clone();
                gpu_handles.push(std::thread::spawn(move || {
                    remote_gpu_worker_entry(
                        &client,
                        &worker_id,
                        target_h160,
                        puzzle_number,
                        &progress,
                        &matches,
                        &stop_flag,
                        &hit_flag,
                        gpu_rotate_keys,
                        start,
                        framework,
                        0, // WebGPU: single device
                        check,
                    )
                }));
            }
            crate::framework::GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let n_devices = crate::cuda::CudaScanner::device_count() as u32;
                    for dev_idx in 0..n_devices {
                        // One worker thread per device; each clones its own Arc.
                        let client = client.clone();
                        let worker_id = worker_id.clone();
                        let progress = progress.clone();
                        let matches = matches.clone();
                        let stop_flag = stop_flag.clone();
                        let hit_flag = hit_flag.clone();
                        gpu_handles.push(std::thread::spawn(move || {
                            remote_gpu_worker_entry(
                                &client,
                                &worker_id,
                                target_h160,
                                puzzle_number,
                                &progress,
                                &matches,
                                &stop_flag,
                                &hit_flag,
                                gpu_rotate_keys,
                                start,
                                framework,
                                dev_idx,
                                check,
                            )
                        }));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    term_line("[remote] CUDA feature not compiled — running CPU-only.");
                }
            }
            crate::framework::GpuFramework::Auto => {
                // Resolved to a concrete backend in main() before we get here.
                unreachable!("framework Auto must be resolved before remote::run")
            }
        }
    }

    for h in handles {
        drop(h.join());
    }
    // The GPU workers finish their current claim on their own (or stop promptly
    // once the hub runs out of chunks / a hit fires), so join them before the
    // stop_flag below — otherwise the flag would cut them off mid-chunk and leak
    // the lease.
    for h in gpu_handles {
        drop(h.join());
    }
    // Every worker has returned.  Stop the ticker and wait for it to exit so
    // a stale status line can't race the final report — the "all done" return
    // path never sets stop_flag, so set it explicitly now that nothing else is
    // running.
    stop_flag.store(true, Ordering::SeqCst);
    let _ = ticker_handle.join();

    let final_matches = {
        match Arc::try_unwrap(matches) {
            Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    };

    // 命中即停收尾顺序：终端已打印 [HIT] → 先落盘 aman_<TS>.txt（worker PC 本地）。
    crate::report::flush_match_files(&final_matches, output_dir);

    // ── 4. final report ────────────────────────────────────────────────────
    term_line("");
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("  REMOTE SCAN COMPLETE  (hub={})", client.base);
    eprintln!("──────────────────────────────────────────────────");
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

/// Fetch `/api/status` once at startup (retrying until the hub is reachable,
/// up to ~90s), then decode + verify the target hash160 — the same startup
/// check local puzzle mode performs against its worklist.
fn connect(client: &HubClient) -> ([u8; 20], Option<u32>, HubSummary) {
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        match client.status() {
            Ok(s) => break s,
            Err(e) => {
                if Instant::now() >= deadline {
                    eprintln!("[remote] hub unreachable after 90s ({e}) — giving up.");
                    std::process::exit(2);
                }
                term_line(&format!("[remote] hub unreachable ({e}) — retrying …"));
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    };

    let meta = &status.meta;
    if meta.solved {
        term_line(&format!(
            "[remote] hub 已 solved（有 worker 命中 puzzle #{}）— 无需工作。",
            meta.puzzle_number,
        ));
        std::process::exit(0);
    }
    let target_h160 = btc::legacy_address_hash160(&meta.target).unwrap_or_else(|| {
        eprintln!("[remote] hub target {} is not a valid P2PKH address", meta.target);
        std::process::exit(2);
    });
    if let Some(ref h160) = meta.hash160 {
        match hash160_from_hex(h160) {
            Some(expected) if expected == target_h160 => {
                term_line(&format!("[remote] hash160 OK ({h160})"));
            }
            Some(_) => {
                eprintln!(
                    "[remote] hash160 MISMATCH: hub says {h160}, target {} decodes to {}",
                    meta.target,
                    hex::encode(target_h160)
                );
                std::process::exit(2);
            }
            None => {
                eprintln!("[remote] hub hash160 is not valid hex: {h160}");
                std::process::exit(2);
            }
        }
    }

    term_line(&format!(
        "[remote] hub={} puzzle=#{} target={}  pend={} running={} done={}",
        client.base,
        meta.puzzle_number,
        meta.target,
        status.summary.pending,
        status.summary.running,
        status.summary.finished,
    ));
    (target_h160, Some(meta.puzzle_number), status.summary)
}

/// 启动 client-auth（`POST /api/auth`，docs/remote-protocol.md 时序 ①a）——**只在
/// `identity_path` 存在时走**：加载本地 `identity.json` 身份 → 明文登记（`name` = 昵称，
/// 供 hub UI 显示）→ 解开 hub 返回的加密信封拿 `hub_pubkey` → 校验它是合法曲线点（业务
/// v2 密封要对它做 ECDH）→ TOFU 写入 `hub.json` → 把 `identity` + `hub_pubkey` 收进
/// client（此后 claim/heartbeat/… 请求体自动 v2 密封）。打印 `[remote] auth ok · hub=<前缀…>`。
///
/// **返回 `Some(己方 pubkey, 小写 64-hex)`**——run() 把它作为干活端点（claim/heartbeat/
/// done/win/release）的 worker_id，hub 才据此把该 client 计入 auth 轨（worker_id 双语义，
/// docs/remote-api.md §1）。
///
/// `identity_path` **缺失** → 明文匿名昵称运行（worker_id = 昵称，业务请求不加密，旧行为）：
/// 打一行日志并返回 `None`，不 exit。文件**在**但损坏/不自洽、auth 被拒（403 whitelist /
/// 422 非法 pubkey 等）、信封解密失败、hub 公钥非法、TOFU 身份变化 → 一律带原因 exit 2
/// （放身份文件 = 期望 auth，绝不静默降级为明文扫描）。网络层错误同 `connect()` 有界重试
/// （≤90s、2s 步进）后 exit 2。
fn maybe_authenticate(
    client: &mut HubClient,
    identity_path: &Path,
    name: &str,
) -> Option<String> {
    if !identity_path.exists() {
        term_line("[remote] 未发现 identity.json — 明文匿名昵称运行（worker_id = 昵称）");
        return None;
    }
    let identity = clientauth::load_identity(identity_path).unwrap_or_else(|e| {
        eprintln!("[remote] {e}");
        std::process::exit(2);
    });
    term_line(&format!(
        "[remote] 身份: pubkey={}…",
        &identity.pubkey[..16]
    ));

    let deadline = Instant::now() + Duration::from_secs(90);
    let reply = loop {
        match client.auth(&identity, name) {
            Ok(reply) => break reply,
            Err(AuthErr::Transport(e)) => {
                if Instant::now() >= deadline {
                    eprintln!("[remote] auth: hub unreachable after 90s ({e}) — giving up.");
                    std::process::exit(2);
                }
                term_line(&format!("[remote] auth: hub unreachable ({e}) — retrying …"));
                std::thread::sleep(Duration::from_secs(2));
            }
            Err(AuthErr::Denied { code, detail }) => {
                eprintln!("[remote] hub 拒绝 auth（HTTP {code}）：{detail}");
                std::process::exit(2);
            }
            Err(AuthErr::Envelope(msg)) => {
                eprintln!("[remote] auth 信封解密失败：{msg}");
                std::process::exit(2);
            }
        }
    };
    // hub 公钥必须是合法曲线点（业务 v2 密封对它做 ECDH 派生 pair-key）——派生一次作证明；
    // 非法（被篡改/坏 hub）→ exit 2，不带着坏 key 开始加密扫描。
    clientauth::static_key(&identity, &reply.hub_pubkey).unwrap_or_else(|e| {
        eprintln!("[remote] hub 公钥非法，无法建立业务加密信道：{e}");
        std::process::exit(2);
    });
    term_line(&format!("[remote] auth ok · hub={}…", &reply.hub_pubkey[..16]));

    // TOFU：hub_pubkey 持久化，跨启动校验 hub 身份未变（变化 → remember_hub Err → exit 2）。
    let hub_path = clientauth::hub_json_path(identity_path);
    clientauth::remember_hub(&hub_path, &reply, &client.base).unwrap_or_else(|e| {
        eprintln!("[remote] {e}");
        std::process::exit(2);
    });

    // 身份 + hub 公钥收进 client：此后所有业务请求自动 v2 密封（sealed_body / unseal_response）。
    let own_pubkey = identity.pubkey.clone();
    client.identity = Some(identity);
    client.hub_pubkey = Some(reply.hub_pubkey);
    Some(own_pubkey)
}

/// Background status line: every `heartbeat_secs`, query the hub and rewrite
/// the in-place line with the global view plus this worker's chunk progress.
fn ticker(
    client: &HubClient,
    worker_id: &str,
    progress: &Progress,
    stop_flag: &AtomicBool,
    heartbeat_secs: f64,
) {
    let mut prev_total = 0u64;
    let mut prev_time = Instant::now();
    loop {
        // Sleep in short slices so a stop (Ctrl+C / hit / all-done) is noticed
        // within ~0.2s instead of on the next full heartbeat_secs tick — this
        // keeps run()'s final `join` prompt and prevents a stale status line
        // after the workers have all returned.
        let slice = Duration::from_millis(200);
        let mut slept = Duration::ZERO;
        while slept < Duration::from_secs_f64(heartbeat_secs) {
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(slice);
            slept += slice;
        }
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let now = Instant::now();
        let total = progress.checked.load(Ordering::Relaxed);
        let dt = now.duration_since(prev_time).as_secs_f64();
        let rate = if dt > 0.1 {
            (total - prev_total) as f64 / dt
        } else {
            0.0
        };
        prev_total = total;
        prev_time = now;

        let line = match client.status() {
            Ok(s) => {
                let sum = &s.summary;
                let total_chunks = sum.pending + sum.running + sum.finished;
                let done_pct = if total_chunks > 0 {
                    sum.finished as f64 / total_chunks as f64 * 100.0
                } else {
                    100.0
                };
                let mine = s
                    .workers
                    .iter()
                    .find(|w| w.worker_id == worker_id)
                    .map(|w| {
                        if w.chunks.is_empty() {
                            format!("{} chunks", w.chunk_count)
                        } else {
                            let avg: f64 =
                                w.chunks.iter().map(|c| c.progress_pct).sum::<f64>()
                                    / w.chunks.len() as f64;
                            format!("{} chunks avg {avg:.1}%", w.chunk_count)
                        }
                    })
                    .unwrap_or_else(|| "no lease".to_string());
                format!(
                    "[remote] pend={} running={} done={} ({:.1}%) {}  keys={} rate={}/s",
                    sum.pending,
                    sum.running,
                    sum.finished,
                    done_pct,
                    mine,
                    fmt_comma(total),
                    fmt_comma(rate.round() as u64),
                )
            }
            Err(e) => format!("[remote] hub status unavailable ({e})"),
        };
        term_status(&format!("  {line}"));
    }
}

// ── one worker thread ────────────────────────────────────────────────────────

/// Claim → scan → report loop.  Each thread holds up to one chunk at a time
/// (claim count = 1), so `n_workers` threads keep `n_workers` chunks in flight
/// on the hub.  All threads share the same `worker_id` — the hub leases chunks
/// by chunk id, not by worker, so this is fine.
#[allow(clippy::too_many_arguments)]
fn remote_worker(
    wid: u32,
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    rotate_keys: Option<u64>,
    start: Instant,
    check: BtcCheck,
) {
    // Throttle for claim-failure logging so a hub outage prints once / 30s,
    // not once / 2s.
    let mut last_fail_log = Instant::now();

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // ── claim one chunk ───────────────────────────────────────────────
        // Claim capability = how many keys this worker scans before reclaiming
        // (the rotation budget, below).  0 (= rotation disabled) maps to the
        // hub's default 2^41, the largest chunk it will hand out.
        let claimed = match client.claim(worker_id, 1, rotate_keys.unwrap_or(0)) {
            Ok(c) => c,
            Err(e) => {
                // Transport error (hub down / slow).  Back off and retry; if
                // we held a lease, the hub reclaims it on its own once its
                // `[reclaim] timeout_seconds` elapses.
                if last_fail_log.elapsed() >= Duration::from_secs(30) {
                    term_line(&format!("[remote] claim failed ({e}) — retrying …"));
                    last_fail_log = Instant::now();
                }
                std::thread::sleep(CLAIM_IDLE);
                continue;
            }
        };

        if claimed.solved {
            // 别的 worker 已命中：hub 不再发任务，本 worker 停止。
            term_line("[remote] hub 已 solved（其他 worker 找到私钥）— 退出。");
            return;
        }

        if claimed.granted == 0 || claimed.chunks.is_empty() {
            // Nothing pending: every chunk is running elsewhere, or the puzzle
            // is complete.  Check the hub before spinning.
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            match client.status() {
                Ok(s) if s.meta.solved || s.summary.pending + s.summary.running == 0 => return, // solved / all done
                _ => {}
            }
            std::thread::sleep(CLAIM_IDLE);
            continue;
        }

        let chunk = &claimed.chunks[0];
        let chunk_id = chunk.id;
        let start_bytes = parse_hex_key(&chunk.current_hex);
        let end_bytes = parse_hex_key(&chunk.end_hex);

        // ── pow 任务（Phase 2）────────────────────────────────────────────
        // hub 在 `[pow] enabled=true` 且本 client 已登记（auth 轨）时随 grant
        // 项下发。缺省 `None` ⇒ 下面的每条分支都退回今日的老路（旧 hub / 匿名
        // 昵称轨 / 窗口过窄都不带 `task`），行为逐字节一致（回归锚点）。
        let task = chunk.pow_task();
        let proofs: Vec<[u8; 20]> = task
            .as_ref()
            .map(|t| t.proof_hash160s.clone())
            .unwrap_or_default();
        // proof 命中走独立 sink——**绝不**进 `matches`/`hit_flag`/`stop_flag`
        // （那是"找到私钥"的通道，proof 只是"扫过这个位置"的凭据）。
        let mut found_proofs: Vec<ProofHit> = Vec::new();
        // pow 窗口**禁止 rotate**：中途 park = 弃整窗（hub 无部分 credit、task
        // cancelled），收敛只能靠扫完整窗后 `POST /api/pow`。声明给 hub 的
        // capability 仍是 `rotate_keys`（窗口宽度本就按它派发），只是本租约内
        // 不许提前交还。
        let eff_rotate = if task.is_some() { None } else { rotate_keys };
        match (&task, rotate_keys) {
            // capability = 0 → hub 按 `reclaim.client_capability` 默认（2^41）派窗：
            // CPU ~500 kkeys/s 下是上千小时的租约，中断即弃整窗（R2 陷阱）。
            (Some(_), None) => term_line(
                "[pow] 警告：task 窗口已挂载但 rotate 预算为 0（声明 capability=0）——\
                 hub 会按默认 2^41 派窗，本租约必须一口气扫完，中断即弃整窗。\
                 建议给 pow 会话开一个 rotate 预算。",
            ),
            // 声明了 capability 仍可能拿到更宽的窗（hub 侧配置/边界情形）——
            // 冻结窗口下无法 park，只能一口气扫完，值得先出声。
            (Some(t), Some(cap)) => {
                let limit = crate::gpu::convert::scalar_add_be(&t.x, cap);
                if crate::gpu::convert::be_lt(&limit, &t.y) {
                    term_line(&format!(
                        "[pow] 警告：task 窗口比声明的 capability（{cap}）宽——冻结窗口不可 \
                         park，本租约必须一口气扫完（否则弃整窗）"
                    ));
                }
            }
            (None, _) => {}
        }

        // 随机扫描方向（每次 claim 掷一次硬币），与本地模式一致：两个方向覆盖
        // 相同的 key 集合 [start, end)，只改变遍历顺序。
        let reverse = puzzle::pick_random(&[true, false]).copied().unwrap_or(false);
        let dir = if reverse {
            ScanDir::Reverse
        } else {
            ScanDir::Forward
        };
        term_line(&format!(
            "[claim] w={wid} chunk={chunk_id} range={}..{} dir={}",
            abbr_hex(&start_bytes),
            abbr_hex(&end_bytes),
            if reverse { "REV" } else { "FWD" },
        ));

        // ── scan with the shared core ─────────────────────────────────────
        // The on_position closure sends a heartbeat every HEARTBEAT_INTERVAL
        // (30s — well under the hub's configurable `[reclaim] timeout_seconds`).
        // Forward sends `current`, reverse
        // sends `end` — the two fields carry the full resume position.  If the
        // hub ever reports the lease lost (404/409), we stop reporting, scan
        // to the end of the claim, and re-claim; the hub already reverted the
        // chunk to pending at the last successful heartbeat.
        let mut lease_lost = false;
        // Set by the heartbeat closure when the hub reports solved (someone
        // else found the key); we then abort the scan and skip the release.
        let mut solved_flag = false;
        let mut last_hb = Instant::now();
        // Worker-wide cumulative keys at the last heartbeat — the delta over
        // the heartbeat window is the rate broadcast to the hub (same source
        // as the status line's `rate=/s`).
        let mut last_keys = progress.checked.load(Ordering::Relaxed);
        // Set by the heartbeat closure when the hub reports the lease lost
        // (404/409); scan_chunk checks it on its next 2048-cadence stop and
        // abandons the rest of this claim instead of scanning to the rotation
        // budget — the hub already reverted the chunk to pending.
        let abort_flag = AtomicBool::new(false);
        let outcome = scan_chunk(ScanChunkOptions {
            target_h160,
            puzzle_number,
            worker_id: wid,
            chunk_id: Some(chunk_id),
            start: start_bytes,
            end: end_bytes,
            dir,
            check,
            rotate_keys: eff_rotate,
            progress,
            matches,
            stop_flag,
            hit_flag,
            abort_flag: Some(&abort_flag),
            proofs: &proofs,
            found_proofs: &mut found_proofs,
            start_elapsed: start,
            on_position: &mut |pos: &ResumePosition| {
                if lease_lost || last_hb.elapsed() < HEARTBEAT_INTERVAL {
                    return;
                }
                let (cur, end) = match dir {
                    ScanDir::Forward => (Some(hex_encode_key(&pos.current)), None),
                    ScanDir::Reverse => (None, Some(hex_encode_key(&pos.end))),
                };
                // Broadcast the worker-wide cumulative keys and the rate over
                // this heartbeat window; the hub keeps them in memory only.
                let keys = progress.checked.load(Ordering::Relaxed);
                let dt = last_hb.elapsed().as_secs_f64();
                let rate = if dt > 0.1 {
                    (keys - last_keys) as f64 / dt
                } else {
                    0.0
                };
                match client.heartbeat(chunk_id, worker_id, cur, end, Some(keys), Some(rate)) {
                    Ok(solved) => {
                        last_hb = Instant::now();
                        last_keys = keys;
                        if solved {
                            // 别的 worker 已命中：尽快放弃本 claim（下轮 claim 退出）。
                            solved_flag = true;
                            abort_flag.store(true, Ordering::Relaxed);
                        }
                    }
                    Err(e) => {
                        if is_lease_lost(&e) {
                            lease_lost = true;
                            abort_flag.store(true, Ordering::Relaxed);
                        } else {
                            // Transport error: re-arm the throttle so we retry
                            // in HEARTBEAT_INTERVAL instead of every 2048 keys.
                            last_hb = Instant::now();
                        }
                    }
                }
            },
        });

        // ── report the outcome ───────────────────────────────────────────
        if lease_lost {
            continue; // hub already parked our chunk; nothing to finalize
        }
        if solved_flag {
            // 别的 worker 已命中：hub 已 solved，无需 release/park——下轮 claim
            // 读到 solved 即退出。
            continue;
        }

        if outcome.matched {
            // Win: report the hit to the hub (`/api/win` — hub 落 win 记录 + 置
            // solved), then stop (scan_chunk already set stop_flag + hit_flag).
            if let Err(e) = client.win(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] win failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            return;
        }

        if outcome.invalid_start {
            // Start key outside [1, n-1]: park it back at its original start so
            // a future claim can re-pick it (mirrors the local finalize).
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&start_bytes)),
                None,
            ) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        } else if outcome.done {
            match &task {
                // pow 窗口扫完：**不调 done**（hub 对冻结窗口恒 409），收敛只能
                // 走全窗 digest 提交；返回 false ⇒ 整个 worker 停机（puzzle 已
                // solved），true ⇒ 继续 claim 下一块。
                Some(t) => {
                    if !converge_pow(client, worker_id, chunk_id, t, &found_proofs, "cpu") {
                        return;
                    }
                }
                // Whole range scanned — finished.
                None => {
                    if let Err(e) = client.done(chunk_id, worker_id) {
                        if !is_lease_lost(&e) {
                            term_line(&format!(
                                "[remote] done failed ({e}) — hub will reclaim the chunk"
                            ));
                        }
                    }
                }
            }
        } else if task.is_some() {
            // pow 窗口被提前中断（SIGINT / stop_flag，rotate 已禁用故不会是预算）：
            // 冻结窗口下 hub 忽略 park 进度、整窗回 pending（无部分 credit），
            // 所以主动 release 弃窗、立刻还回去，胜过等 hub reclaim 超时。
            term_line(&format!(
                "[pow] w={wid} chunk={chunk_id} 扫描中断——弃整窗（无部分 credit）"
            ));
            if let Err(e) = client.release(chunk_id, worker_id, None, None) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[pow] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        } else {
            // Parked (rotation budget hit or SIGINT): send the resume position.
            // Forward → current = next key to scan.  Reverse → end = sk + 1
            // (current stays at start).  Identical semantics to local mode.
            let (cur, end) = match dir {
                ScanDir::Forward => (Some(hex_encode_key(&outcome.sk)), None),
                ScanDir::Reverse => (
                    None,
                    Some(hex_encode_key(&crate::gpu::convert::scalar_add_be(
                        &outcome.sk,
                        1,
                    ))),
                ),
            };
            if let Err(e) = client.release(chunk_id, worker_id, cur, end) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ── GPU remote worker ────────────────────────────────────────────────────────

/// Entry point for the remote GPU worker: resolve the backend, set up the
/// scanner, and hand off to the shared `remote_gpu_worker` loop.  Returns
/// (CPU-only) when the selected backend has no usable device — the run simply
/// continues with the CPU workers, exactly like local puzzle mode's GPU worker.
#[allow(clippy::too_many_arguments)]
fn remote_gpu_worker_entry(
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    rotate_keys: Option<u64>,
    start: Instant,
    framework: crate::framework::GpuFramework,
    device_index: u32,
    check: BtcCheck,
) {
    // Only the CUDA arm consumes `device_index` (WebGPU always uses device 0);
    // silence the unused-param warning in a non-CUDA build.
    let _ = device_index;
    match framework {
        crate::framework::GpuFramework::WebGpu => {
            let gpu_ctx = match crate::gpu::GpuContext::new_blocking(0) {
                Ok(c) => c,
                Err(e) => {
                    term_line(&format!(
                        "[remote] GPU unavailable ({e}) — running CPU-only."
                    ));
                    return;
                }
            };
            term_line(&format!(
                "[remote] GPU worker up on {}",
                gpu_ctx.device_name()
            ));
            let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
            let mut scanner = match crate::gpu::GpuScanner::new(gpu_ctx, &candidates) {
                Ok(s) => s,
                Err(e) => {
                    term_line(&format!(
                        "[remote] GpuScanner::new failed ({e}) — running CPU-only."
                    ));
                    return;
                }
            };
            // Dense-tiling config: stride = N threads, single target candidate.
            scanner.stride = crate::gpu::NUM_GPU_THREADS;
            scanner.num_candidates = 1;
            scanner.check_compressed_pk = check.compressed as u32;
            scanner.check_uncompressed_pk = check.uncompressed as u32;
            remote_gpu_worker(
                client,
                worker_id,
                target_h160,
                puzzle_number,
                scanner,
                progress,
                matches,
                stop_flag,
                hit_flag,
                rotate_keys,
                start,
                "GPU",
                check,
            );
        }
        crate::framework::GpuFramework::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
                let mut scanner =
                    match crate::cuda::CudaScanner::new_on_device(&candidates, device_index) {
                        Ok(s) => s,
                        Err(e) => {
                            term_line(&format!(
                                "[remote] CUDA device #{device_index} init failed ({e}) — running CPU-only on this card."
                            ));
                            return;
                        }
                    };
                term_line(&format!(
                    "[remote] CUDA worker #{} up on {}",
                    scanner.device_index(),
                    scanner.device_name()
                ));
                scanner.stride = crate::gpu::NUM_GPU_THREADS;
                scanner.num_candidates = 1;
                scanner.check_compressed_pk = check.compressed as u32;
                scanner.check_uncompressed_pk = check.uncompressed as u32;
                // Per-card label so [claim] / [HIT] lines identify the GPU.
                // Compute before `scanner` is moved into the worker.
                let backend_label = format!("CUDA[{}]", scanner.device_index());
                remote_gpu_worker(
                    client,
                    worker_id,
                    target_h160,
                    puzzle_number,
                    scanner,
                    progress,
                    matches,
                    stop_flag,
                    hit_flag,
                    rotate_keys,
                    start,
                    &backend_label,
                    check,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                term_line("[remote] CUDA feature not compiled — running CPU-only.");
            }
        }
        crate::framework::GpuFramework::Auto => {
            // Resolved to a concrete backend in main() before we get here.
            unreachable!("framework Auto must be resolved before remote::run")
        }
    }
}

/// Claim → GPU dense-tile scan → report loop for one remote GPU worker thread.
/// Mirrors the local puzzle GPU loop (`puzzle_gpu_scan_loop`) but pulls chunks
/// from the hub over HTTP and persists progress via throttled heartbeats +
/// release-at-rotation instead of a local SQLite worklist.
///
/// The GPU scans *forward only* (dense tiling seeds walker `i` at `start + i`
/// with stride N) — unlike the CPU workers' per-claim direction coin-flip.  That
/// is fine: the hub hands out exclusive chunks, so any traversal order covers
/// the same key set with no overlap; only the in-order walk differs.
///
/// Lease handling mirrors the CPU worker: a 404/409 heartbeat/release means the
/// hub already reverted the chunk to pending at the last reported position, so
/// we abandon the claim and re-claim rather than fight over it.
#[allow(clippy::too_many_arguments)]
fn remote_gpu_worker<S: PuzzleScannerBackend>(
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    mut scanner: S,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    rotate_keys: Option<u64>,
    start: Instant,
    backend_label: &str,
    check: BtcCheck,
) {
    // Per-dispatch coverage (keys) once the scanner is configured.  Constant
    // once calibrated: `steps_per_call` defaults to 1 at construction.
    let dispatch_keys = crate::gpu::NUM_GPU_THREADS as u64
        * *scanner.steps_per_call() as u64;

    // Throttle for claim-failure logging so a hub outage prints once / 30s.
    let mut last_fail_log = Instant::now();

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // ── claim one chunk ───────────────────────────────────────────────
        // Claim capability = how many keys this worker scans before reclaiming
        // (the rotation budget, below).  0 (= rotation disabled) maps to the
        // hub's default 2^41, the largest chunk it will hand out.
        let claimed = match client.claim(worker_id, 1, rotate_keys.unwrap_or(0)) {
            Ok(c) => c,
            Err(e) => {
                if last_fail_log.elapsed() >= Duration::from_secs(30) {
                    term_line(&format!("[remote] claim failed ({e}) — retrying …"));
                    last_fail_log = Instant::now();
                }
                std::thread::sleep(CLAIM_IDLE);
                continue;
            }
        };
        if claimed.solved {
            // 别的 worker 已命中：hub 不再发任务，本 worker 停止。
            term_line("[remote] hub 已 solved（其他 worker 找到私钥）— 退出。");
            return;
        }
        if claimed.granted == 0 || claimed.chunks.is_empty() {
            // Nothing pending: every chunk is running elsewhere, or the puzzle
            // is complete.  Check the hub before spinning.
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            match client.status() {
                Ok(s) if s.meta.solved || s.summary.pending + s.summary.running == 0 => return, // solved / all done
                _ => {}
            }
            std::thread::sleep(CLAIM_IDLE);
            continue;
        }

        let chunk = &claimed.chunks[0];
        let chunk_id = chunk.id;
        let start_bytes = parse_hex_key(&chunk.current_hex);
        let end_bytes = parse_hex_key(&chunk.end_hex);

        // ── pow 任务（Phase 2）：与 CPU 路径同款装配 ────────────────────────
        let task = chunk.pow_task();
        let proofs: Vec<[u8; 20]> = task
            .as_ref()
            .map(|t| t.proof_hash160s.clone())
            .unwrap_or_default();
        let mut found_proofs: Vec<ProofHit> = Vec::new();
        let eff_rotate = if task.is_some() { None } else { rotate_keys };

        term_line(&format!(
            "[claim] w={backend_label} chunk={chunk_id} range={}..{}",
            abbr_hex(&start_bytes),
            abbr_hex(&end_bytes),
        ));

        // ── per-chunk candidate table + check switches ─────────────────────
        // 槽 0 = target，槽 1..=N = proof hash160。**proofs 非空时强制开压缩
        // 比对**：proof 定义在压缩公钥上，压缩哈希不算出来就永远比不到 proof；
        // 真实 target 的 `[btc]` 语义由下面的 CPU 重验块按原开关把关（那里用的
        // 是配置里的 `check`，不是这里上传给内核的强制值）。
        let candidates = crate::gpu::convert::chunk_candidates(&target_h160, &proofs);
        let cc = if proofs.is_empty() {
            check.compressed as u32
        } else {
            1
        };
        if let Err(e) = scanner.configure_chunk(
            &candidates,
            1 + proofs.len() as u32,
            cc,
            check.uncompressed as u32,
        ) {
            term_line(&format!(
                "[remote] {backend_label} configure_chunk failed ({e}) — releasing chunk"
            ));
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&start_bytes)),
                None,
            ) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            continue;
        }

        // ── seed the 100k strided walkers at start + i ─────────────────────
        if scanner.seed_range(start_bytes).is_err() {
            term_line(&format!(
                "[remote] {backend_label} seed_range failed — releasing chunk"
            ));
            // Park back at the original start so a future claim re-picks it.
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&start_bytes)),
                None,
            ) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            continue;
        }

        let mut current = start_bytes; // next key NOT yet covered
        let mut scanned_keys: u64 = 0; // keys covered this claim (rotation)
        let mut hit = false; // CPU-verified match → this chunk is the winner
        let mut lease_lost = false;
        // 心跳报告 hub 已 solved（别的 worker 命中）→ 提前退出，不 park。
        let mut solved = false;
        let mut last_hb = Instant::now();
        // Worker-wide cumulative keys at the last heartbeat — the delta over
        // the heartbeat window is the rate broadcast to the hub.
        let mut last_keys = progress.checked.load(Ordering::Relaxed);

        // ── scan the chunk in N·steps_per_call-key dispatches ──────────────
        loop {
            // Decide this dispatch's step count.  A full dispatch covers
            // `dispatch_keys` keys; the final (partial) dispatch is trimmed so
            // the walkers land on or just past `end`.  The chunk width can
            // exceed 2^64, so never compute `end - current` directly — compare
            // `end` against `current + dispatch_keys` (the small add never
            // overflows) and only subtract once the remainder is known to fit.
            let steps = if crate::gpu::convert::be_lt(&current, &end_bytes) {
                let reach = crate::gpu::convert::scalar_add_be(&current, dispatch_keys);
                // `reach >= end`  ⟺  `end - current <= dispatch_keys` (no overflow).
                if !crate::gpu::convert::be_lt(&reach, &end_bytes) {
                    let remaining = crate::gpu::convert::scalar_sub_be(&end_bytes, &current);
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
                break;
            }

            // ── heartbeat (throttled to HEARTBEAT_INTERVAL) ────────────────
            // Broadcast the scan position so the hub can resume the chunk if
            // this worker dies, plus the worker-wide keys + rate.  Forward
            // scanning always reports `current` (the next key to scan).
            if last_hb.elapsed() >= HEARTBEAT_INTERVAL {
                let keys = progress.checked.load(Ordering::Relaxed);
                let dt = last_hb.elapsed().as_secs_f64();
                let rate = if dt > 0.1 {
                    (keys - last_keys) as f64 / dt
                } else {
                    0.0
                };
                match client.heartbeat(
                    chunk_id,
                    worker_id,
                    Some(hex_encode_key(&current)),
                    None,
                    Some(keys),
                    Some(rate),
                ) {
                    Ok(hub_solved) => {
                        last_hb = Instant::now();
                        last_keys = keys;
                        if hub_solved {
                            // 别的 worker 已命中：尽快退出（下轮 claim 停止）。
                            solved = true;
                            break;
                        }
                    }
                    Err(e) => {
                        if is_lease_lost(&e) {
                            // Hub reverted the chunk — abandon this claim.
                            lease_lost = true;
                            break;
                        }
                        // Transport error: re-arm the throttle so we retry in
                        // HEARTBEAT_INTERVAL instead of every dispatch.
                        last_hb = Instant::now();
                    }
                }
            }

            // ── one dispatch ──────────────────────────────────────────────
            // Temporarily set steps_per_call for this (possibly final, partial)
            // dispatch, restoring the default afterwards.
            let saved_steps = *scanner.steps_per_call();
            *scanner.steps_per_call() = steps;
            let batch = crate::gpu::NUM_GPU_THREADS as u64 * steps as u64;
            match scanner.step() {
                Ok(batch_matches) => {
                    if !batch_matches.is_empty() {
                        // Collect CPU-verified matches inside the lock, then
                        // drop the lock before printing ([HIT] is never emitted
                        // under it).
                        let verified: Vec<MatchEvent> = {
                            let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                            let mut out = Vec::new();
                            for m in &batch_matches {
                                let mut ev = crate::puzzle::gpu_match_to_event(
                                    m,
                                    chunk_id,
                                    start_bytes,
                                    puzzle_number,
                                );
                                // CPU verification — never trust the GPU
                                // candidate flag alone; spurious matches are
                                // dropped silently.  Gated by the same `[btc]`
                                // switches as the shader, so a serialisation
                                // disabled for checking is never accepted here
                                // either.
                                let h = btc::hash160(&ev.compressed);
                                let h_u = btc::hash160(&ev.uncompressed);
                                if (check.compressed && h == target_h160)
                                    || (check.uncompressed && h_u == target_h160)
                                {
                                    ev.elapsed = start.elapsed().as_secs_f64();
                                    g.push(ev.clone());
                                    out.push(ev);
                                } else if let Some(idx) =
                                    crate::puzzle::classify_proof(&h, &proofs)
                                {
                                    // pow proof 命中：**不是命中**——不进
                                    // matches、不设 hit_flag/stop_flag，只记
                                    // proof 下标 + 裸 key（收敛时算全窗 digest）。
                                    // 按重算的压缩 hash160 归类、不信任 GPU 的
                                    // candidate_index（候选槽索引根本没回传）——
                                    // 与上面"target 命中也要 CPU 重验"同一条原则。
                                    found_proofs.push(ProofHit {
                                        index: idx,
                                        key: ev.private_key,
                                    });
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
                                    chunk_id,
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
                        "[remote] {backend_label} step failed ({e}) — releasing chunk"
                    ));
                    break;
                }
            }
            *scanner.steps_per_call() = saved_steps;

            // Advance checkpoint by exactly the keys this dispatch covered.
            current = crate::gpu::convert::scalar_add_be(&current, batch);
            scanned_keys += batch;

            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            // ── rotation ──────────────────────────────────────────────────
            // Park after `rotate_keys` keys scanned *this claim* and let the
            // loop claim a fresh pending chunk.  `scanned_keys` resets every
            // claim, so this is a per-claim budget exactly like the CPU
            // worker's.  Releasing also keeps the hub's resume position fresher
            // than the 30s heartbeat would on its own.  `None` (= config 0)
            // disables rotation: the chunk is scanned to completion.
            // pow 窗口上 `eff_rotate` 恒为 `None`（中途 park = 弃整窗）——见上。
            if let Some(rot) = eff_rotate {
                if scanned_keys >= rot {
                    break;
                }
            }
        }

        // ── report the outcome ────────────────────────────────────────────
        if lease_lost {
            continue; // hub already parked our chunk; nothing to finalize
        }
        if solved {
            // 别的 worker 已命中：hub 已 solved，无需 release/park——下轮 claim
            // 读到 solved 即退出。
            continue;
        }

        if hit {
            // Win: report the hit to the hub (`/api/win` — hub 落 win 记录 + 置
            // solved), then stop (the scan already set stop_flag + hit_flag).
            if let Err(e) = client.win(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] win failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            return;
        }

        // Fully scanned (current >= end) is `done` regardless of why the loop
        // stopped (stop flag / GPU error / rotation): parking an empty
        // `[end, end)` chunk back at the hub would leave it pending forever
        // instead of finishing it.  A chunk that ends exactly on the rotation
        // budget boundary lands here via this path.
        let done = !crate::gpu::convert::be_lt(&current, &end_bytes);
        if done {
            match &task {
                // pow 窗口扫完：不调 done（冻结窗口恒 409），收敛只能 /api/pow。
                Some(t) => {
                    if !converge_pow(
                        client,
                        worker_id,
                        chunk_id,
                        t,
                        &found_proofs,
                        backend_label,
                    ) {
                        return;
                    }
                }
                // Whole range scanned — finished.
                None => {
                    if let Err(e) = client.done(chunk_id, worker_id) {
                        if !is_lease_lost(&e) {
                            term_line(&format!(
                                "[remote] done failed ({e}) — hub will reclaim the chunk"
                            ));
                        }
                    }
                }
            }
        } else if task.is_some() {
            // pow 窗口被提前中断（SIGINT / stop_flag / GPU step 失败）：冻结
            // 窗口下 hub 忽略 park 进度、整窗回 pending（无部分 credit），主动
            // release 弃窗胜过等 hub reclaim 超时。
            term_line(&format!(
                "[pow] w={backend_label} chunk={chunk_id} 扫描中断——弃整窗（无部分 credit）"
            ));
            if let Err(e) = client.release(chunk_id, worker_id, None, None) {
                if !is_lease_lost(&e) {
                    term_line(&format!("[pow] release failed ({e}) — hub will reclaim the chunk"));
                }
            }
        } else {
            // Parked (rotation budget, SIGINT, or GPU error): send the resume
            // position.  Forward scanning → `current` = next key to scan.
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&current)),
                None,
            ) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn be(n: u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[24..32].copy_from_slice(&n.to_be_bytes());
        out
    }

    fn h160(seed: u8) -> [u8; 20] {
        let mut out = [0u8; 20];
        out[19] = seed;
        out
    }

    fn h160_hex(seed: u8) -> String {
        hex::encode(h160(seed))
    }

    // ── digest 契约（镜像 hub `powproof.expected_digest_for`）───────────────

    #[test]
    fn pow_digest_vector_matches_hub() {
        // 向量由 hub 侧实测（backend/app/powproof.expected_digest_for([1,2,3])）：
        // SHA256(升序 32B-BE concat)。⚠️ 姊妹实现 vanity_worker_bpi.py 的
        // compute_proof_hash 是按池端数组顺序拼 64-hex ASCII——照抄必然 400，
        // 这个向量就是那道护栏。
        assert_eq!(
            hashed_proof_key(&[be(1), be(2), be(3)]),
            "9701f34c80e1ef7f8125e5d4d2d7e19b509e25d26e462d5308b5abb95b64783e"
        );
    }

    #[test]
    fn pow_digest_is_order_independent() {
        // proof key 的收集顺序由扫描方向与命中先后决定（正扫/反扫不同），
        // digest 必须与顺序无关 → 先升序再 concat。
        let v = hashed_proof_key(&[be(1), be(2), be(3)]);
        assert_eq!(v, hashed_proof_key(&[be(3), be(1), be(2)]));
        assert_eq!(v, hashed_proof_key(&[be(2), be(3), be(1)]));
        assert_eq!(v, hashed_proof_key(&[be(3), be(2), be(1)]));
    }

    #[test]
    fn pow_digest_is_sensitive_to_key_set() {
        assert_ne!(
            hashed_proof_key(&[be(1), be(2), be(3)]),
            hashed_proof_key(&[be(1), be(2), be(4)])
        );
    }

    #[test]
    fn pow_digest_orders_by_unsigned_value_not_bytes_reversed() {
        // 升序是**无符号整数**序（大端字节序即字典序）——0x100 应排在 0x0FF 之后。
        assert_eq!(
            hashed_proof_key(&[be(0x100), be(0x0FF)]),
            hashed_proof_key(&[be(0x0FF), be(0x100)])
        );
        let mut concat = Vec::new();
        concat.extend_from_slice(&be(0x0FF));
        concat.extend_from_slice(&be(0x100));
        use sha2::{Digest, Sha256};
        assert_eq!(hashed_proof_key(&[be(0x100), be(0x0FF)]), hex::encode(Sha256::digest(&concat)));
    }

    // ── claim 响应反序列化 ─────────────────────────────────────────────────

    /// 一个 grant 项的 JSON（`task` 可选）——`x_hex`/`y_hex` 恒等于
    /// `current_hex`/`end_hex`，与 hub `puzzle.py` 同源生成一致。
    fn claim_json(task: Option<String>) -> String {
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(1000));
        let task = task.map(|t| format!(r#","task":{t}"#)).unwrap_or_default();
        format!(
            r#"{{"granted":1,"solved":false,"chunks":[{{"id":7,"current_hex":"{x}","end_hex":"{y}"{task}}}]}}"#
        )
    }

    fn task_json(x: &str, y: &str, count: usize, hashes: &[String]) -> String {
        let hashes: Vec<String> = hashes.iter().map(|h| format!("\"{h}\"")).collect();
        format!(
            r#"{{"task_id":42,"x_hex":"{x}","y_hex":"{y}","proof_count":{count},"proof_hash160s":[{}]}}"#,
            hashes.join(",")
        )
    }

    #[test]
    fn claim_without_task_parses_as_none() {
        // 回归锚点：旧 hub / 匿名昵称轨 / 窗口过窄（powproof 返回 None）都不带
        // `task` 键 → None → 报告链全部走老路，行为与今日逐字节一致。
        let resp: ClaimResponse = serde_json::from_str(&claim_json(None)).unwrap();
        assert_eq!(resp.granted, 1);
        assert!(!resp.solved);
        assert!(resp.chunks[0].pow_task().is_none());
    }

    #[test]
    fn claim_with_well_formed_task_parses() {
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(1000));
        let json = claim_json(Some(task_json(
            &x,
            &y,
            2,
            &[h160_hex(0xAA), h160_hex(0xBB)],
        )));
        let resp: ClaimResponse = serde_json::from_str(&json).unwrap();
        let t = resp.chunks[0].pow_task().expect("well-formed task");
        assert_eq!(t.task_id, 42);
        assert_eq!(t.x, be(10));
        assert_eq!(t.y, be(1000));
        assert_eq!(t.proof_hash160s, vec![h160(0xAA), h160(0xBB)]);
    }

    #[test]
    fn malformed_tasks_fall_back_to_none() {
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(1000));
        let cases = [
            // proof_count 与数组长度不符
            (
                "count mismatch",
                task_json(&x, &y, 3, &[h160_hex(1)]),
            ),
            // proof_count = 0（不得硬编码 6，但 0 个 proof 无意义）
            ("zero proofs", task_json(&x, &y, 0, &[])),
            // 非 40-hex
            ("bad hex", task_json(&x, &y, 1, &["zz".to_string()])),
            // 长度不足 40-hex
            ("short hex", task_json(&x, &y, 1, &["ab".repeat(19)])),
            // 任务窗与 grant 项的 current/end 不符（协议错位）
            (
                "window mismatch",
                task_json(&hex_encode_key(&be(11)), &y, 1, &[h160_hex(1)]),
            ),
            // 任务窗 hex 非 hex / 超 64 位：**必须返回 None 而不是 panic**
            // （`parse_hex_key` 对这两种输入是 assert + panic）。
            ("x not hex", task_json("zz", &y, 1, &[h160_hex(1)])),
            (
                "x too long",
                task_json(&"a".repeat(70), &y, 1, &[h160_hex(1)]),
            ),
            ("y not hex", task_json(&x, "nothex!", 1, &[h160_hex(1)])),
            ("x empty", task_json("", &y, 1, &[h160_hex(1)])),
        ];
        for (label, task) in cases {
            let json = claim_json(Some(task));
            let resp: ClaimResponse = serde_json::from_str(&json).unwrap();
            assert!(
                resp.chunks[0].pow_task().is_none(),
                "应拒绝畸形任务（{label}）"
            );
        }
    }

    /// **回归锚点（P1）**：`task` 存在但**形状不符**（键改名 / 某字段类型不符）
    /// 时，整个 `ClaimResponse` 仍必须解析成功——否则 `claim()` 返 Err，worker
    /// 只会无限 `claim failed — retrying`，一块也领不到。
    ///
    /// `task` 因此收成 `serde_json::Value`：形状校验下沉到 `pow_task()`，最坏
    /// 后果是"这一块没 pow"，而不是"全队没活干"。hub 侧改个键名即触发。
    #[test]
    fn shape_drifted_task_does_not_break_the_whole_claim() {
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(1000));
        let h = h160_hex(1);
        let cases = [
            // 键改名：x_hex → x
            (
                "renamed key",
                format!(
                    r#"{{"task_id":42,"x":"{x}","y_hex":"{y}","proof_count":1,"proof_hash160s":["{h}"]}}"#
                ),
            ),
            // 类型不符：proof_hash160s 是 null 而非数组
            (
                "null array",
                format!(
                    r#"{{"task_id":42,"x_hex":"{x}","y_hex":"{y}","proof_count":1,"proof_hash160s":null}}"#
                ),
            ),
            // task_id 类型不符（字符串而非数字）
            (
                "task_id as string",
                format!(
                    r#"{{"task_id":"42","x_hex":"{x}","y_hex":"{y}","proof_count":1,"proof_hash160s":["{h}"]}}"#
                ),
            ),
            // 缺 proof_count
            (
                "missing proof_count",
                format!(r#"{{"task_id":42,"x_hex":"{x}","y_hex":"{y}","proof_hash160s":["{h}"]}}"#),
            ),
            // 根本不是对象
            ("not an object", format!("\"whatever\"")),
        ];
        for (label, task) in cases {
            let json = claim_json(Some(task));
            let resp: ClaimResponse = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("claim 响应不该因 task 形状漂移而解析失败（{label}）：{e}"));
            assert_eq!(resp.granted, 1, "granted 仍应可用（{label}）");
            assert_eq!(resp.chunks[0].current_hex, x, "chunk 字段仍应可用（{label}）");
            assert!(
                resp.chunks[0].pow_task().is_none(),
                "形状不符的 task 应按无 pow 处理（{label}）"
            );
        }
    }

    /// 形状对但**多一个未知键**仍是合法任务（向前兼容：hub 加字段不该废掉 pow）。
    #[test]
    fn task_with_extra_unknown_key_is_still_accepted() {
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(1000));
        let task = format!(
            r#"{{"task_id":42,"x_hex":"{x}","y_hex":"{y}","proof_count":1,"proof_hash160s":["{}"],"future_field":123}}"#,
            h160_hex(0xAA)
        );
        let resp: ClaimResponse = serde_json::from_str(&claim_json(Some(task))).unwrap();
        let t = resp.chunks[0].pow_task().expect("未知键不该使任务失效");
        assert_eq!(t.task_id, 42);
        assert_eq!(t.proof_hash160s, vec![h160(0xAA)]);
    }

    #[test]
    fn degenerate_task_window_is_rejected() {
        // x == y：与 grant 项一致但窗口退化 → 无 proof 可放，按无 pow 处理。
        let same = hex_encode_key(&be(1000));
        let task = task_json(&same, &same, 1, &[h160_hex(1)]);
        let json = format!(
            r#"{{"granted":1,"chunks":[{{"id":7,"current_hex":"{same}","end_hex":"{same}","task":{task}}}]}}"#
        );
        let resp: ClaimResponse = serde_json::from_str(&json).unwrap();
        assert!(resp.chunks[0].pow_task().is_none());
    }

    #[test]
    fn task_exceeding_candidate_buffer_is_rejected() {
        // 1 + N ≤ 78：超过候选缓冲的 proof 数不可能被 GPU 比较 → 拒绝。
        let x = hex_encode_key(&be(10));
        let y = hex_encode_key(&be(100000));
        let hashes: Vec<String> = (0..78u32).map(|i| h160_hex(i as u8)).collect();
        let resp: ClaimResponse =
            serde_json::from_str(&claim_json(Some(task_json(&x, &y, 78, &hashes)))).unwrap();
        assert!(resp.chunks[0].pow_task().is_none());
    }

    // ── PowResp ────────────────────────────────────────────────────────────

    #[test]
    fn pow_resp_parses_and_defaults() {
        let r: PowResp = serde_json::from_str(
            r#"{"ok":true,"task_id":42,"chunk_id":7,"keys_scanned":990,"solved":false}"#,
        )
        .unwrap();
        assert!(r.ok);
        assert_eq!(r.task_id, 42);
        assert_eq!(r.chunk_id, 7);
        assert_eq!(r.keys_scanned, 990);
        assert!(!r.solved);

        // 旧/新 hub 增删字段都不该让收敛失败（全字段 serde default）。
        let r: PowResp = serde_json::from_str(r#"{"ok":true}"#).unwrap();
        assert_eq!(r.keys_scanned, 0);
        assert!(!r.solved);
    }
}
