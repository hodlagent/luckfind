# luckfind remote 模式通信协议

> 本文件记录 `luckfind --remote <hub-url>`（LAN-hub worker 模式）下 worker 与远程
> hub 之间的完整通信流程：从领取 chunk 到心跳、旋转释放、上报完成，以及两端各自的
> 超时/回收机制。Rust 端参考 `src/remote.rs`；hub 端参考
> `/Users/jerin/Dev/lan-hub/backend/app/`（`routes.py` / `puzzle.py` / `main.py` /
> `config.py`）。
>
> 各 HTTP 端点的请求/响应/状态码逐字段定义见 lan-hub 仓库的
> `docs/remote-api.md`（`/Users/jerin/Dev/lan-hub/docs/remote-api.md`）。

---

## 1. 架构总览

```
                    ┌─────────────────────────────────────────────┐
                    │           hub (lan-hub, FastAPI)             │
                    │  SQLite worklist（唯一写者）                  │
                    │  worker_leases 表 · reclaim_loop 周期回收     │
                    └───────▲──────────────────────────▲──────────┘
                            │ HTTP /api/*                │ HTTP /api/*
                    claim / heartbeat /                  status (ticker)
                    release / done
                    ┌───────┴──────────────────────────┴──────────┐
                    │         worker (luckfind --remote)           │
                    │  N 个 CPU 线程 + 每 GPU 设备 1 线程           │
                    │  每线程同一时刻持有 ≤1 个 chunk（claim=1）    │
                    │  无本地 DB · 断点全部交还 hub                │
                    └─────────────────────────────────────────────┘
```

关键设计（`src/remote.rs` 模块头注释）：

- **hub 是唯一写者**。worker 从不打开本地 `.db`，断点位置全部通过 HTTP 上报。
- **lease 以 chunk id 区分**，不以 worker 区分——所有线程共享同一个 `worker_id`
  （`remote.rs::remote_worker` 的文档注释）。
- **崩溃恢复是 hub 的职责**：worker 死了/断网 → 心跳停止 → hub 超时回收 lease，
  chunk 回退 pending 并**保留最后上报的位置**。
- **Phase 2 pow**：已登记 client 的 claim 可能带 `task`（N 个 proof hash160），
  扫完整窗后走 `/api/pow` 交全窗 digest 收敛——见 §8。

---

## 2. 参数一览

### 2.1 Worker 端（`src/remote.rs`）

| 常量/配置 | 值 | 作用 | 位置 |
|---|---|---|---|
| `HEARTBEAT_INTERVAL` | 30s | 扫描期间的心跳节流 | `remote.rs::HEARTBEAT_INTERVAL` |
| `CLAIM_IDLE` | 2s | 无 chunk 可领时的重试间隔 | `remote.rs::CLAIM_IDLE` |
| `rotate_keys` (CPU) | 默认 2²⁷ = 134,217,728 keys | CPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `cpu_rotate_keys`；`0` 禁用） |
| `gpu_rotate_keys` (GPU) | 默认 2³¹ = 2,147,483,648 keys | GPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `gpu_rotate_keys`；`0` 禁用） |
| `check_compressed_pk` / `check_uncompressed_pk` | 默认均 true | 决定 worker 把压缩（33B）/非压缩（65B）公钥的 hash160 与目标比较；被禁用的序列化在 CPU/GPU 热路径上不再计算。**不改变 hub 协议**——chunk 照常领取扫描，只是命中判定只看启用的序列化。**例外**：pow 窗口上压缩哈希必须算（proof 定义在压缩公钥上），CPU 侧只在有 proof 时多算一次、GPU 侧经 `configure_chunk` 强制开，真实 target 的开关语义由 CPU 重验块把关 | `[btc]` 配置段 → `BtcCheck`，贯穿 CPU `scan_chunk`/`worker_loop` 与 GPU shader/kernel |
| `identity`（`[remote]` 段） | 缺省 `<cwd>/identity.json` | 本地 Nostr 身份文件（`identify` / `nostr.py create` 产出的 JSON，0600）。**存在 = auth 轨开关**：启动 auth + 业务全走 v2 静态信道信封（§3）；**缺失 → 明文匿名昵称运行**（worker_id = 昵称，用户 2026-09 拍板）；**在但损坏/不自洽 → exit 2**（不静默降级） | `src/clientauth.rs::load_identity`（自检 secret→pubkey 派生一致） |
| `npub_sha256`（`[remote]` 段） | 无 | **hub 身份钉扎摘要** = `sha256(hub 的 npub 文本)`，小写 hex；取自 hub 机 `backend/data/identity.json` 的 `npub_sha256` 字段。auth 拿到 hub 公钥后转 npub 算摘要比对，不符 → exit 2。**有 identity 文件而缺此键 → exit 2**（fail closed，不回落 TOFU）；无 identity 文件（匿名轨）不需要它 | `src/clientauth.rs::verify_hub_pinned` / `npub_sha256` |
| HTTP 连接超时 | 5s | 防 hub 挂死卡线程 | `HubClient::new` |
| HTTP 单请求超时 | 15s | 同上 | `HubClient::new` |
| auth 网络重连 | 每 2s 一次，上限 90s | `POST /api/auth` 传输层错误的有界重试（同 `connect()`） | `remote.rs::maybe_authenticate` |
| 启动重连 | 每 2s 一次，上限 90s | `connect()` 拿 /api/status | `remote.rs::connect` |
| claim 失败日志节流 | 30s | hub 故障时每 30s 才打印一次 | `remote_worker` |

> 旋转预算不是 `remote.rs` 内的常量——由 CLI 或配置文件 `cpu_rotate_keys` /
> `gpu_rotate_keys` 解析后传入（`main.rs` `resolve_rotate`，默认 2²⁷ / 2³¹，`0`
> 禁用 = 扫到整段完成）。每个 claim 会把该值作为 `capability` 声明给 hub（`0`
> 由 hub 按默认 2⁴¹ 处理），hub 据此优先分配宽度匹配的 chunk。

### 2.2 Hub 端（`lan-hub/backend/app/config.py`）

| 常量 | 默认值 | 作用 |
|---|---|---|
| `RECLAIM_TIMEOUT` | 120s | `heartbeat_at` 超过 120s → 回退 pending（保留 current） |
| `RECLAIM_INTERVAL` | 15s | `reclaim_loop` 每 15s 跑一次回收扫描 |
| `MAX_CLAIM_COUNT` | 1000 | 单次 claim 上限（worker 只用 count=1） |

> 两者均可被环境变量 `LANHUB_RECLAIM_TIMEOUT` / `LANHUB_RECLAIM_INTERVAL` 或
> hub 的 `.toml` 配置覆盖；本文件按默认值描述。

---

## 3. API 端点（hub `routes.py`）

| 端点 | 方法 | 请求体（节选） | 语义 |
|---|---|---|---|
| `/api/auth` | POST | `{pubkey, name?, version?}` | **client-auth 登记**（Phase 1，明文握手）：UPSERT 该 client，返回加密信封 `auth`（见下）；hub 自身 Nostr ID 只在信封里、不进明文响应 |
| `/api/status` | GET | — | puzzle meta + `pending/running/finished` 统计 + 各 worker 概况；`meta.solved/win` 表明是否已有人命中 |
| `/api/chunks/claim` | POST | `{worker_id, count, capability}` | 领取 `count` 个 pending chunk；`capability` 声明"扫多少 keys 后 reclaim"，hub 据此优先分配宽度匹配的 chunk |
| `/api/chunks/{id}/heartbeat` | POST | `{worker_id, current_hex?, end_hex?, keys?, rate?}` | 刷新 lease + 可选保存进度 + 上报速率指标 |
| `/api/chunks/{id}/done` | POST | `{worker_id}` | 整段扫完：置 finished、删 lease |
| `/api/win` | POST | `{worker_id, chunk_id, private_key?}` | **命中上报**（取代 done）：最终化 chunk + hub 落 win 记录、置 puzzle solved。`private_key`（64 位小写 hex）**只在 auth 轨带**——hub 据此另存 key 文件并自动扫币（见 §7.4） |
| `/api/chunks/{id}/release` | POST | `{worker_id, current_hex?, end_hex?}` | 旋转/放弃：保存进度、回退 pending、删 lease；**pow 窗口上 = 弃整窗**（见 §8） |
| `/api/pow` | POST | `{worker_id, task_id, chunk_id, hashed_proof_key}` | **pow 全窗 digest 提交**（Phase 2，取代 pow 窗口上的 done）：hub 校验 `expected_digest` 相等 → 按任务窗口终态化 chunk + verified 记账，返回 `{ok, task_id, chunk_id, keys_scanned, solved}`（见 §8） |

**worker_id 双语义（hub docs/remote-api.md §1，2026-09）**：干活端点
（claim/heartbeat/done/win/release）的 `worker_id` **若是该 client 已登记的 pubkey**，
hub 把活跃/keys 回流进其 `clients` 行（auth 轨记账）；自由文本昵称 → 匿名昵称轨
（不积累、绝不自动建行）。本实现已对齐：identity.json 在时 `maybe_authenticate` 返回
己方 pubkey，run() 把它作为干活端点的 `worker_id`；`nickname`（--worker-id / 主机名）
只作 auth 的 `name` 上报供 hub UI 显示，不再出现在干活端点。identity.json **缺失** →
返回 None，干活端点 worker_id = 昵称（匿名昵称轨，全程明文）。

**业务信封（auth 轨，2026-09 §1.1 收尾，hub docs/remote-api.md §20）**：auth 成功后
`HubClient.identity`/`hub_pubkey` 置位 → 每个业务 POST 的 body 包成
`{"pubkey": <己方>, "enc": {v:2, nonce, ciphertext}}` 发送（pair-key
`K = SHA-256(x(ECDH(己方秘密, even-lift(hub 公钥))))`，AES-256-GCM，aad = 发送方 =
己方 pubkey）；响应体若是 `{"enc": {...}}`（hub 加密回，aad = hub 公钥）→ 先解出明文
再按原 typed struct 解析。明文旧 hub / 匿名轨不包不解。**非 2xx 语义不变**（404/409 照旧
判 lease 丢）；信封在但解密失败 → fatal exit 2（对端不是 auth 时钉住的那台 hub = 钉扎违背）。
实现：`clientauth.rs::static_key/encrypt_to_peer/decrypt_from_peer` +
`remote.rs::sealed_body/unseal_response`。

> **注：hub 身份为什么钉摘要而不是 TOFU（2026-09 拍板）**
> v1 auth 信封是 ECIES——发送方（hub）用**一次性**密钥，所以「解得开」只证明**收件人**
> 能解、**不证明发送方持有 hub 私钥**：任何人都能对着本 client 的公钥封一个自称 hub 的
> 信封。于是「首次握手就学会 hub 公钥」（TOFU）不是信任根——LAN 中间人只要抢在真 hub
> 之前应答第一次握手即可，而 win/done 会带着私钥走那条信道。改为在配置里钉住
> `[remote] npub_sha256`：没有可被冒充的首次接触，也不落任何状态文件（旧 `hub.json`
> 不再读写，磁盘上的残留文件无副作用）。钉摘要与直接钉 npub 强度等价（SHA-256 抗原像），
> 好处是钉扎值本身不泄露 hub 身份；缺失即 exit 2（fail closed，不回落 TOFU）。
> 同一套模型见 `vanity_worker.py::HUB_NPUB_SHA256`。

响应/错误约定：

- **auth 成功**返回 `{ok, pubkey, auth, client}`，其中 `auth` 是 hub 用一次性 ECDH
  密钥（ECIES v1，`envelope.py`）加密的**信封**：`{v: 1, ephemeral_pubkey, nonce,
  ciphertext}`，AAD = 该 client 的 pubkey。明文载荷 = `{"hub_pubkey", "server_time"}`
  ——worker 用本地私钥 + 信封的 `ephemeral_pubkey` 解开才拿到 hub 公钥；解不开 =
  pubkey 声明与私钥不符/被篡改。**403**（whitelist/blacklist 命中）与 **422**（pubkey
  非法）带 `{detail}`，worker 打印后 exit 2。
- **claim 成功**返回 `{granted, chunks: [{id, current_hex, end_hex, task?}], solved}`；
  `granted=0` 表示 hub 当前没有可领的 pending chunk；`solved=true` 表示别的 worker
  已命中 → worker 应停止。**`task` 可选**（Phase 2 pow，只在 hub 开 `[pow]` 且本
  client 已登记时出现；缺省 = 无 pow 语义，一切照旧），形状见 §8。
- **heartbeat 成功**返回 `{ok, solved}`——`solved=true` 让正在扫 chunk 的 worker
  **提前放弃本 claim**（下轮 claim 退出），不必等一轮扫完。
- **404**：chunk 无 lease（`puzzle.py::PuzzleService._require_owner`）→ lease 已丢。
- **409**：chunk 被别的 worker 持有（同上，`LeaseError(409, …)` 分支）→ lease 已丢；
  `/api/win` 的 409 表示 puzzle 已被其他 worker 先标记 solved。
- 心跳 `keys`/`rate` 是瞬态指标，hub 只存内存、不落库（`metrics.py`）。
- solved 落盘：`POST /api/win` 后 hub 在 `backend/data/` 写
  `{puzzle_number}_{timestamp}.txt`（worker_id/chunk_id，**不含私钥**）；文件存在即
  solved，重启不丢。auth 轨带了 `private_key` 时另写一份
  `{puzzle_number}_{timestamp}_key.txt`（`0600`）——key 文件独立于记录文件、不参与
  solved 判定，且被 `win.py::_win_files()` 过滤掉，**绝不进 `/api/status` 回显**。

---

## 4. 完整时序

```
worker                                   hub (lan-hub)
──────                                   ──────────────
① 启动 connect()
   └─ GET /api/status        ────────→   失败每 2s 重试，上限 90s（remote.rs::connect）
   ←──────────────────────────────────── 校验 hash160 一致（不匹配直接退出，exit 2）

①a client-auth / 匿名门禁（claim 前的条件开关，src/remote.rs::maybe_authenticate）
   ├─ 读 identity.json（`[remote] identity` 覆盖，缺省 <cwd>/identity.json）
   │     **不存在** → 明文匿名昵称运行：worker_id = 昵称，跳过全部 auth，业务不加密
   │     （打印 `[remote] 未发现 identity.json — 明文匿名昵称运行（worker_id = 昵称）`）
   └─ 存在：要求 `[remote] npub_sha256` 已配置
      │     **缺失 → exit 2**（fail closed：不知对端是谁就不出示身份、不建加密信道）
      └─ 自检由 secret_key 派生 pubkey 与文件一致（不自洽 → exit 2）
      └─ POST /api/auth        ──────────→ {pubkey: <64hex>, name: <昵称>, version: 1}
           · name = 昵称（旧自由文本 worker_id，--worker-id / 主机名），只供 hub UI 显示
           · 此后**干活端点的 worker_id 一律用 <pubkey>**（worker_id 双语义，见上表下方注）
         · 非 2xx：403（名单）/ 422（pubkey 非法）→ 打印 hub detail 后 exit 2
         · 传输层错误每 2s 重试，上限 90s
      ←──────────────────────────────────── {ok, pubkey, auth: {v, ephemeral_pubkey,
                                             nonce, ciphertext}, client}
      └─ 用本地私钥 + ephemeral_pubkey 解 AES-256-GCM 信封（ECIES v1，src/clientauth.rs）
         · tag 校验失败（非发给这把私钥/被篡改）→ exit 2
         · 解开 → 学到 hub_pubkey → 校验是合法曲线点（对端加密信道要对它做 ECDH）
         · hub_pubkey → npub → sha256(npub 文本) 与配置 npub_sha256 比对
           · 不符 → exit 2（打印配置值/实得值与实得 npub，便于核对或换 hub 后更新配置）
           · 相符 → 打印 `[remote] auth ok · hub_sha256=<摘要前缀…>`（日志不含 hub 身份）
         · hub_pubkey 存内存（client.hub_pubkey）→ 此后业务 POST 全走 v2 信封（见 §3）
      └─ 不落任何状态文件：钉扎值来自配置，没有可被冒充的「首次接触」（§3 注）
   └─ pending+running == 0  → 直接退出「nothing to do」

② 领取 chunk（每个线程 ≤1 个）
   └─ POST /api/chunks/claim ─────────→ {worker_id, count: 1, capability: rotate_keys|gpu_rotate_keys}
   ←──────────────────────────────────── {granted, chunks[0], solved}
   └─ solved=true → 别的 worker 已命中，直接退出
   └─ granted=0：
      ├─ 查一次 /api/status，solved 或 pending+running==0 → 退出
      └─ 否则 sleep 2s 重试（CLAIM_IDLE）

③ 扫描阶段（方向随机 FWD/REV；GPU 固定 forward 密集铺片）
   └─ 每 30s 心跳：
      POST /api/chunks/{id}/heartbeat → {worker_id, current|end, keys, rate}
         · forward 发 current_hex，reverse 发 end_hex —— 两字段各承载完整续扫位置
         · keys/rate = 整机累计 keys + 该窗口速率（hub 内存缓存）
   └─ 响应 solved=true → 别的 worker 已命中，**提前放弃本 claim**（下轮 claim 退出）
   └─ 心跳 404/409 → lease 已丢，放弃本 chunk 直接重新 claim（绝不崩溃）

④a 旋转预算打满 / 首次 Ctrl+C：
   └─ POST /api/chunks/{id}/release ──→ {worker_id, current|end}
        （断点交还 hub，下一轮再 claim 新 chunk）

④b 扫完 / 命中：
   └─ 扫完（无 pow 任务）：POST /api/chunks/{id}/done ─→ {worker_id}
   └─ 扫完（**有 pow 任务**）：POST /api/pow ───────────→ {worker_id, task_id,
        chunk_id, hashed_proof_key}
        （hub 校验全窗 digest → 终态化 + verified 记账；**此窗口不得调 done**，见 §8）
   └─ 命中：POST /api/win ──────────────→ {worker_id, chunk_id, private_key?}
        （打印 [HIT]；hub 落 win 记录、置 solved；整个 run 停止，
          其它 worker 在下一次 claim/心跳读到 solved 也停止）

⑤ 背景 status ticker（独立线程，默认每 10s）
   └─ GET /api/status        ────────→   仅重绘状态行，不参与 lease 维护
```

### 4.1 续扫位置语义

| 方向 | 心跳/release 携带字段 | 含义 |
|---|---|---|
| Forward | `current_hex` | 下一个待扫 key（`remote.rs::ChunkUpdateBody::current_hex`） |
| Reverse | `end_hex` | 收缩后的独占上界 = `sk + 1`，`current` 保持 start 不动（`remote.rs::ChunkUpdateBody::end_hex`） |

hub 侧 `puzzle.py::PuzzleService.heartbeat`：`current_hex`/`end_hex` 任一非空即落库对应列，然后
`heartbeat_at` 刷新为当前时间。旧客户端不传 `end_hex` 行为完全不变（向后兼容）。

---

## 5. 心跳 vs 回收：时间线

```
claim     心跳     心跳     心跳        hub 判过期       下次回收检查
 │         │        │        │            │                │
 ▼         ▼        ▼        ▼            ▼                ▼
 ├──── 30s ──── 30s ──── 30s ──…─── [≤120s] ─── 15s ────►
 └─► heartbeat 每次重置 30s 定时器 ─► 最后心跳 +120s 处判定过期，最多再 15s 内回收
```

- **心跳 30s ≪ 回收 120s = 4 倍裕量**（`remote.rs` 常量区注释同样强调）。
- Hub 的 `reclaim_loop`（`main.py::reclaim_loop`）每 **15s** 扫一次 `worker_leases`，按
  `heartbeat_at < now - 120s` 判过期（`puzzle.py::PuzzleService.reclaim`）。因此实际回收发生在
  「最后一次心跳后 120s ~ 135s」之间的某一刻，而非精确 120s。
- 回收动作（同一个 `PuzzleService.reclaim`）：running → pending（保留 `current`）+ 删除 lease。
  下一次 claim 会从该 `current` 继续。

### 5.1 崩溃恢复（双向兜底）

| 场景 | 机制 |
|---|---|
| worker 死了 / 断网 | 心跳停止 → hub 120s 后回收，chunk 回退 pending，保留最后上报位置 |
| worker 活着但 hub 不可达 | 心跳请求超时（15s）→ 只重排节流，不丢 lease；hub 恢复后继续心跳 |
| hub 重启 | `lifespan` 启动回收：`PuzzleService.reclaim(conn, config.RECLAIM_TIMEOUT)`——只回收**真正过期**（`> RECLAIM_TIMEOUT` 没心跳）的 lease、孤儿与空 pending（`main.py::lifespan`）。**刻意不用 `reclaim(conn, 0)` 全清**：那会把快速重启期间还活着的 worker lease 误杀，慢板子下次心跳直接 409；真崩了的 worker 也会在 timeout 内自然过期，等价崩溃恢复，只是从"立即"变成"≤ timeout" |
| worker 心跳/release 收到 404/409 | 视为 lease 已丢，放弃本 chunk 重新 claim，绝不崩溃 |

---

## 6. 旋转释放（rotation）与心跳的关系

`release` 不只是「扫完了」，它还让 hub 的断点位置**比心跳更频繁地刷新**：

- **CPU**：默认每 `cpu_rotate_keys`（2²⁷）keys release 一次。1.4 Mkeys/s 下单线程
  约 **96s** 一次，实际短于 120s 回收线，与心跳共同维持 lease 存活。
- **GPU**：默认每 `gpu_rotate_keys`（2³¹）keys release 一次。约 100 Mkeys/s 下约
  **20s** 一次，比 30s 心跳更频繁——有意为之：release-at-rotation 比 30s 心跳更
  频繁地刷新 hub 的续扫位置。
- 配置 `0` 禁用旋转：worker 把 chunk 扫到整段完成（`done`）才释放，claim 的
  `capability` 声明为 0（hub 按默认 2⁴¹ 处理）。

> 因此「lease 存活」其实由两条路径共同保证：30s 心跳是保底，旋转 release 是加分。
>
> **pow 窗口例外**：带 `task` 的窗口强制禁用旋转（`eff_rotate = None`）——中途 park
> 等于弃掉整窗（hub 无部分 credit），收敛只能扫完整窗走 `/api/pow`（§8.4）。

---

## 7. 易混淆点

1. **CLI 的 `--heartbeat` ≠ 心跳间隔**。`-H/--heartbeat`（默认 10.0）只控制
   终端状态行的刷新频率（`args.rs` 中该选项的帮助文本明确说明）；真正的 hub lease 心跳
   是硬编码的 30s，不受该参数影响。
2. **status ticker 不参与 lease**。`remote.rs::ticker` 线程每
   `heartbeat_secs` 查一次 `/api/status` 重绘进度行，与 lease 维护完全无关。
3. **`claim` 失败 ≠ lease 丢失**。transport 错误（hub 慢/挂）只 sleep 2s 重试；
   只有 404/409 才代表该 chunk 的 lease 已不归我们（`remote.rs::is_lease_lost`）。
4. **`win` ≠ `done`**。`/api/win` 只在命中时调用：除了 done 的最终化，还会让 hub 落
   win 记录文件并置 puzzle solved，从而广播停止其它 worker。扫完一整个区间（没命中）
   仍走 `done`。
5. **私钥只在 auth 轨随 `win` 上报**（`remote.rs` 的 `HubClient::win_body`）。判据就是
   `sealed_body` 的同一个开关——`identity`（`identity.json`）在不在：**在** → body 会被
   v2 信封密封，`private_key`（64 位小写 hex）一并上行，hub 另存
   `{puzzle_number}_{ts}_key.txt`（`0600`）并**立刻自动扫币**（派生地址校验 `== target`
   后把余额扫到 `[btc] vault`，hub `remote-api.md` §22）；**不在**（匿名昵称轨）→ body
   是明文 HTTP，**一律不发私钥**，hub 记 `no-key`，私钥只留本地 `aman_*.txt`，事后由
   运维走 `python3 -m app.sweep --key` 手工补扫。
   自动扫币的时限意义：私钥一解出全网谁都能扫走那笔奖金，"命中 → 广播"的窗口期直接
   决定钱是否到手——所以 auth 轨上这一发是**有意**的取舍，不是漏发。
6. **proof 命中 ≠ 命中**。pow 窗口里扫到某个 proof 的 hash160 只是"确实扫过这个位置"
   的证据，**不停机、不进 `matches`、不打印 `[HIT]`、不设 `hit_flag`/`stop_flag`**
   （那两个标志全队共享，误设会让整支机队停下），只记进独立的 `found_proofs` sink，
   然后接着扫。命中（真实 target）始终优先：同一个 key 又中 target 又是 proof 时走
   `matched` → `/api/win`。
7. **pow 窗口的 `done` 恒被 409 挡**。冻结窗口只能靠 `/api/pow` 收敛；`rotate` 在
   pow 窗口上被强制关闭（`eff_rotate = None`），因为中途 park 等于弃掉整窗、无部分
   credit。

---

## 8. 工作量证明（pow 窗口，Phase 2）

> hub 侧契约见 lan-hub `docs/remote-api.md` §21（任务模型 / 状态码 / verified 记账）。
> 本节只写 worker 侧怎么接。

### 8.1 任务形状

hub 开 `[pow]` 且本 client **已登记**（identity.json 在 → auth 轨）时，claim 的 grant
项多一个 `task` 对象（`puzzle.py::PuzzleService.claim` 内组装，值来自
`powproof.build_task`）：

```json
{"id": 12, "current_hex": "…", "end_hex": "…",
 "task": {"task_id": 7, "x_hex": "…", "y_hex": "…",
          "proof_hash160s": ["<40hex>", …]}}
```

- `x_hex`/`y_hex` **恒等于**该 grant 项的 `current_hex`/`end_hex`（hub 由同一对
  `scan_start`/`scan_end` 生成）——`pow_task()` 本地交叉校验，不符即按无任务处理。
  两者都是 32B 大端 → **恒 64 位小写 hex**（hub `powproof.build_task` 的
  `to_bytes(32,"big").hex()`），没有短写形式。
- **`task` 在客户端是 `serde_json::Value`，不是结构体**：它是 `ClaimResponse` 的
  嵌套字段，直接放强类型会让"形状漂移"（hub 改键名 / 某字段类型不符）把**整个
  claim 响应**的反序列化拖垮 → `claim()` 返 Err → worker 无限 `claim failed —
  retrying`，一块也领不到。形状校验因此下沉到 `pow_task()`（`from_value` 失败 →
  警告 + 按无任务处理），最坏后果是"这一块没 pow"。多出的**未知键**照常接受
  （向前兼容），只有类型/必填项不符才算漂移。
- 解析 `x_hex`/`y_hex` 用 `parse_hex_key_checked`（全函数，畸形返回 `None`）而非
  `parse_hex_key`（对 >64 字符 assert、非 hex panic）——这里处理的正是不可信的
  hub 响应，"畸形即 `None`"的契约不能反过来打死进程。
- **任务里没有 count 字段**：proof 个数 = `proof_hash160s` 数组长度，客户端一律按
  数组长度走（hub 侧个数是常量 `powproof.PROOF_COUNT` = 6，不是配置项——见
  lan-hub `docs/remote-api.md` §21.1 末）。不硬编码 6，但也不读第二个字段：
  没有第二个字段就是没有不一致的余地。`1 + proof_hash160s.len() ≤ 78`（GPU 候选
  缓冲槽位数），越界按畸形任务处理。
- **`expected_digest` 不随任务下发**（只在 hub 库里），所以 worker 只能真扫出
  proof 才提交得出正确 digest。裸 proof keys 同样不出 hub。

### 8.2 扫描：proof 并进比较集合，命中后继续扫

- CPU（`puzzle.rs::scan_chunk`）：热路径的压缩公钥 hash160 只算一次，**target 与
  proof 共用**。`check.compressed` 关着但有 proof 时也要算压缩哈希（proof 定义在
  压缩公钥上，与 `[btc]` 开关无关）。命中 proof → `found_proofs.push(ProofHit{index,
  key})`，然后**照常推进**（见 §7.5）。
- GPU：候选表槽 0 = target、槽 1..=N = proof hash160（`gpu/convert.rs::chunk_candidates`），
  每 chunk 在 `seed_range` 前经 `configure_chunk` 重传（内核按运行时 `num_candidates`
  循环，无需改 shader/kernel）；**proofs 非空时强制开压缩比对**，真实 target 的
  `[btc]` 语义由 CPU 重验块按配置开关把关。命中候选一律 CPU 侧重算 hash160 后归类
  （不信任 GPU 标志，与 target 命中同一条原则）。

### 8.3 收敛：`POST /api/pow`

扫完整窗后（`done`）→ `remote.rs::converge_pow`（CPU/GPU 共用）：

```
集齐 N 个 proof key → 升序 32B-BE concat → SHA256 → 小写 hex → POST /api/pow
```

**digest 契约**：`HashedProofKey = SHA256(升序 32B-BE concat)`，**与收集顺序无关**
（正扫/反扫、命中先后都不同，故必须先排序）。已实测向量（`scripts/test_pow.py` 与
`remote.rs` 单测共用）：

```
keys [1, 2, 3] → 9701f34c80e1ef7f8125e5d4d2d7e19b509e25d26e462d5308b5abb95b64783e
```

> ⚠️ 姊妹实现 `vanity_worker_bpi.py::compute_proof_hash` 是**按池端数组顺序拼 64-hex
> ASCII**——另一套契约，照抄必然 400。上述向量就是那道护栏。

分支处理：

| 情形 | 动作 |
|---|---|
| 提交成功 | 打印 verified 工作量；响应 `solved=true`（puzzle 被别人解出）→ 停机，否则继续 claim |
| 404 / 409 | `is_lease_lost` → 任务已作废（task 没了 / 非 active / 非本人 / lease 已丢），直接重领 |
| 400 | digest 不符 → 大声警告 + `release` 弃窗 + 重领。**刻意不重试**：hub 侧 400 保持 task active、租约内可重传（`verify_pow`），但真凑不出 digest 就说明本地扫描口径与 hub 下发的 proof 集不符，重传多少次都一样——弃窗拿新任务 > 抱着一个永远收敛不了的窗口 |
| **集齐不足** | 限流警告 + `release` 弃窗 + 重领（无法收敛，立刻还回 pending 胜过等 hub 回收超时） |

### 8.4 冻结窗口的强制语义

有 active 任务的窗口被 hub **冻结**在认领快照 `[x, y)`：

| 动作 | worker 行为 |
|---|---|
| heartbeat | 照常发（hub 忽略进度字段、只续租） |
| `done` | **不调用**（hub 恒 409）——收敛只走 `/api/pow` |
| rotate（park） | **禁用**：`eff_rotate = None`。中途 park = 弃整窗（hub 无部分 credit、task cancelled） |
| release | 只在放弃时调（扫不动 / 集齐不足 / 提交失败），参数传 `None, None`——整窗回 pending |
| Ctrl+C / step 失败 | 同上，弃整窗 |

> **配置陷阱（R2）**：pow 窗口不可 park ⇒ 一个租约必须扫完整窗。若 `cpu_rotate_keys`
> / `gpu_rotate_keys` 设为 `0`（禁用旋转），claim 声明的 `capability` 就是 0，hub 会按
> 默认 2⁴¹ 派窗（CPU ~500 kkeys/s 下上千小时）。worker 在挂到 task 时会打警告；**开
> `[pow]` 就应同时开一个 rotate 预算**（窗口宽度 = 声明 capability，一个租约正好扫完）。

### 8.5 回归锚点

`task` 缺失（旧 hub / 匿名昵称轨 / 窗口过窄 `powproof.build_task` 返回 None）⇒
`pow_task() == None` ⇒ 上述所有分支短路，worker 行为与未接 pow 时逐字节一致。

---

## 9. 源码索引

| 逻辑 | Rust | hub |
|---|---|---|
| 心跳常量 / 旋转预算（config 解析传入） | `remote.rs::HEARTBEAT_INTERVAL`/`CLAIM_IDLE`；`main.rs::resolve_rotate` | `config.py::RECLAIM_TIMEOUT`/`RECLAIM_INTERVAL` |
| HTTP 封装与超时 | `remote.rs::HubClient`（impl 块） | `routes.py` |
| 启动 client-auth（条件门禁）+ 解信封 + 钉摘要 | `remote.rs::maybe_authenticate`；`clientauth.rs` | `routes.py::auth`、`envelope.py` |
| 业务 v2 静态信道信封（包/解） | `HubClient::sealed_body`/`unseal_response`；`clientauth.rs::static_key/encrypt_to_peer/decrypt_from_peer` | `routes.py` 前 `transport.py`/`wire.py`（透明中间件） |
| identity.json 自检 | `clientauth.rs::load_identity` | —（hub 侧由 `nostr.py` 生成） |
| hub 身份钉扎（npub 摘要 vs `[remote] npub_sha256`） | `clientauth.rs::npub_of`/`npub_sha256`/`verify_hub_pinned` | `nostr.py::npub_sha256`（生成 `identity.json` 的该字段） |
| 启动 connect + hash160 校验 | `remote.rs::connect` | `routes.py::status` |
| **pow 任务解析/校验** | `ClaimedChunk::pow_task` | `PuzzleService.claim`（组装 `task`）+ `powproof.build_task` |
| **pow digest（升序 32B-BE → SHA256）** | `remote.rs::hashed_proof_key` | `powproof.py::expected_digest_for` |
| **pow 收敛（CPU/GPU 共用）** | `remote.rs::converge_pow`；`HubClient::pow` | `routes.py::pow_submit`、`PuzzleService.verify_pow` |
| **proof 归类（压缩公钥 hash160）** | `puzzle.rs::classify_proof`；`proofs`/`found_proofs` 见 `ScanChunkOptions`、`ProofHit` | `powproof.py::compressed_pubkey/hash160` |
| **GPU 候选表（槽 0 = target，1..=N = proof）** | `gpu/convert.rs::chunk_candidates`；`configure_chunk`（`puzzle.rs` trait 方法 + 两 scanner 的 `set_candidates`） | — |
| CPU worker 主循环 | `remote.rs::remote_worker` | — |
| GPU worker 主循环 | `remote.rs::remote_gpu_worker` | — |
| status ticker | `remote.rs::ticker` | — |
| lease 归属校验（404/409） | `remote.rs::is_lease_lost` | `PuzzleService._require_owner` |
| 心跳落库 + lease 刷新 | — | `PuzzleService.heartbeat`（有 active task → 只续租） |
| done / win / release / pow | `HubClient::done`/`win`/`release`/`pow` | `PuzzleService.release`（弃整窗）、`win.py::record`、`PuzzleService.verify_pow` |
| solved 广播（claim/heartbeat/status 响应） | `remote.rs` 三处停止点 | `routes.py`、`PuzzleService.solved` |
| 回收循环与孤儿清理 | — | `main.py::reclaim_loop`、`PuzzleService.reclaim` |
