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

关键设计（`remote.rs:1-27`）：

- **hub 是唯一写者**。worker 从不打开本地 `.db`，断点位置全部通过 HTTP 上报。
- **lease 以 chunk id 区分**，不以 worker 区分——所有线程共享同一个 `worker_id`
  （`remote.rs:1188-1192` 的 `remote_worker` 文档注释）。
- **崩溃恢复是 hub 的职责**：worker 死了/断网 → 心跳停止 → hub 超时回收 lease，
  chunk 回退 pending 并**保留最后上报的位置**。
- **Phase 2 pow**：已登记 client 的 claim 可能带 `task`（N 个 proof hash160），
  扫完整窗后走 `/api/pow` 交全窗 digest 收敛——见 §8。

---

## 2. 参数一览

### 2.1 Worker 端（`src/remote.rs`）

| 常量/配置 | 值 | 作用 | 位置 |
|---|---|---|---|
| `HEARTBEAT_INTERVAL` | 30s | 扫描期间的心跳节流 | `remote.rs:49` |
| `CLAIM_IDLE` | 2s | 无 chunk 可领时的重试间隔 | `remote.rs:53` |
| `rotate_keys` (CPU) | 默认 2²⁷ = 134,217,728 keys | CPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `cpu_rotate_keys`；`0` 禁用） |
| `gpu_rotate_keys` (GPU) | 默认 2³¹ = 2,147,483,648 keys | GPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `gpu_rotate_keys`；`0` 禁用） |
| `check_compressed_pk` / `check_uncompressed_pk` | 默认均 true | 决定 worker 把压缩（33B）/非压缩（65B）公钥的 hash160 与目标比较；被禁用的序列化在 CPU/GPU 热路径上不再计算。**不改变 hub 协议**——chunk 照常领取扫描，只是命中判定只看启用的序列化。**例外**：pow 窗口上压缩哈希必须算（proof 定义在压缩公钥上），CPU 侧只在有 proof 时多算一次、GPU 侧经 `configure_chunk` 强制开，真实 target 的开关语义由 CPU 重验块把关 | `[btc]` 配置段 → `BtcCheck`，贯穿 CPU `scan_chunk`/`worker_loop` 与 GPU shader/kernel |
| `identity`（`[remote]` 段） | 缺省 `<cwd>/identity.json` | 本地 Nostr 身份文件（`identify` / `nostr.py create` 产出的 5 字段 JSON，0600）。**存在 = auth 轨开关**：启动 auth + 业务全走 v2 静态信道信封（§3）；**缺失 → 明文匿名昵称运行**（worker_id = 昵称，用户 2026-09 拍板）；**在但损坏/不自洽 → exit 2**（不静默降级） | `src/clientauth.rs::load_identity`（自检 secret→pubkey 派生一致） |
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
| `/api/win` | POST | `{worker_id, chunk_id}` | **命中上报**（取代 done）：最终化 chunk + hub 落 win 记录、置 puzzle solved |
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
判 lease 丢）；信封在但解密失败 → fatal exit 2（对端不是当初 auth 的那台 hub = TOFU 违背）。
实现：`clientauth.rs::static_key/encrypt_to_peer/decrypt_from_peer` +
`remote.rs::sealed_body/unseal_response`。

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
- **404**：chunk 无 lease（`_require_owner`，`puzzle.py:166`）→ lease 已丢。
- **409**：chunk 被别的 worker 持有（`puzzle.py:168`）→ lease 已丢；`/api/win` 的
  409 表示 puzzle 已被其他 worker 先标记 solved。
- 心跳 `keys`/`rate` 是瞬态指标，hub 只存内存、不落库（`metrics.py`）。
- solved 落盘：`POST /api/win` 后 hub 在 `backend/data/` 写
  `{puzzle_number}_{timestamp}.txt`（worker_id/chunk_id，**不含私钥**）；文件存在即
  solved，重启不丢。

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
   └─ 存在：自检由 secret_key 派生 pubkey 与文件一致（不自洽 → exit 2）
      └─ POST /api/auth        ──────────→ {pubkey: <64hex>, name: <昵称>, version: 1}
           · name = 昵称（旧自由文本 worker_id，--worker-id / 主机名），只供 hub UI 显示
           · 此后**干活端点的 worker_id 一律用 <pubkey>**（worker_id 双语义，见上表下方注）
         · 非 2xx：403（名单）/ 422（pubkey 非法）→ 打印 hub detail 后 exit 2
         · 传输层错误每 2s 重试，上限 90s
      ←──────────────────────────────────── {ok, pubkey, auth: {v, ephemeral_pubkey,
                                             nonce, ciphertext}, client}
      └─ 用本地私钥 + ephemeral_pubkey 解 AES-256-GCM 信封（ECIES v1，src/clientauth.rs）
         · tag 校验失败（非发给这把私钥/被篡改）→ exit 2
         · 解开 → 学到 hub_pubkey → 校验是合法曲线点 → 打印 `[remote] auth ok · hub=<前缀…>`
         · hub_pubkey 存内存（client.hub_pubkey）→ 此后业务 POST 全走 v2 信封（见 §3）
      └─ 写 hub.json TOFU（与 identity.json 同目录，0600）
         · 已存 hub_pubkey ≠ 新解出的 → exit 2（删 hub.json 以接受新 hub 身份）
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
   └─ 命中：POST /api/win ──────────────→ {worker_id, chunk_id}
        （打印 [HIT]；hub 落 win 记录、置 solved；整个 run 停止，
          其它 worker 在下一次 claim/心跳读到 solved 也停止）

⑤ 背景 status ticker（独立线程，默认每 10s）
   └─ GET /api/status        ────────→   仅重绘状态行，不参与 lease 维护
```

### 4.1 续扫位置语义

| 方向 | 心跳/release 携带字段 | 含义 |
|---|---|---|
| Forward | `current_hex` | 下一个待扫 key（`remote.rs:726`） |
| Reverse | `end_hex` | 收缩后的独占上界 = `sk + 1`，`current` 保持 start 不动（`remote.rs:727`） |

hub 侧 `puzzle.py:184-200`：`current_hex`/`end_hex` 任一非空即落库对应列，然后
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

- **心跳 30s ≪ 回收 120s = 4 倍裕量**（`remote.rs:11` 注释同样强调）。
- Hub 的 `reclaim_loop` 每 **15s** 扫一次 `worker_leases`（`main.py:26-35`），按
  `heartbeat_at < now - 120s` 判过期（`puzzle.py:338-340`）。因此实际回收发生在
  「最后一次心跳后 120s ~ 135s」之间的某一刻，而非精确 120s。
- 回收动作（`puzzle.py:342-347`）：running → pending（保留 `current`）+ 删除 lease。
  下一次 claim 会从该 `current` 继续。

### 5.1 崩溃恢复（双向兜底）

| 场景 | 机制 |
|---|---|
| worker 死了 / 断网 | 心跳停止 → hub 120s 后回收，chunk 回退 pending，保留最后上报位置 |
| worker 活着但 hub 不可达 | 心跳请求超时（15s）→ 只重排节流，不丢 lease；hub 恢复后继续心跳 |
| hub 重启 | 启动即 `reclaim(conn, 0)`：所有 running 回退 pending（`main.py:54-57`） |
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
   终端状态行的刷新频率（`args.rs:32-36` 注释明确说明）；真正的 hub lease 心跳
   是硬编码的 30s，不受该参数影响。
2. **status ticker 不参与 lease**。`remote.rs:530` 的 `ticker` 线程每
   `heartbeat_secs` 查一次 `/api/status` 重绘进度行，与 lease 维护完全无关。
3. **`claim` 失败 ≠ lease 丢失**。transport 错误（hub 慢/挂）只 sleep 2s 重试；
   只有 404/409 才代表该 chunk 的 lease 已不归我们（`is_lease_lost`，`remote.rs:113`）。
4. **`win` ≠ `done`**。`/api/win` 只在命中时调用：除了 done 的最终化，还会让 hub 落
   win 记录文件并置 puzzle solved，从而广播停止其它 worker。扫完一整个区间（没命中）
   仍走 `done`。`/api/win` 不带私钥——私钥只留在命中 worker 本地的 `aman_*.txt`。
5. **proof 命中 ≠ 命中**。pow 窗口里扫到某个 proof 的 hash160 只是"确实扫过这个位置"
   的证据，**不停机、不进 `matches`、不打印 `[HIT]`、不设 `hit_flag`/`stop_flag`**
   （那两个标志全队共享，误设会让整支机队停下），只记进独立的 `found_proofs` sink，
   然后接着扫。命中（真实 target）始终优先：同一个 key 又中 target 又是 proof 时走
   `matched` → `/api/win`。
6. **pow 窗口的 `done` 恒被 409 挡**。冻结窗口只能靠 `/api/pow` 收敛；`rotate` 在
   pow 窗口上被强制关闭（`eff_rotate = None`），因为中途 park 等于弃掉整窗、无部分
   credit。

---

## 8. 工作量证明（pow 窗口，Phase 2）

> hub 侧契约见 lan-hub `docs/remote-api.md` §21（任务模型 / 状态码 / verified 记账）。
> 本节只写 worker 侧怎么接。

### 8.1 任务形状

hub 开 `[pow]` 且本 client **已登记**（identity.json 在 → auth 轨）时，claim 的 grant
项多一个 `task` 对象（`puzzle.py` 的 `task_info`）：

```json
{"id": 12, "current_hex": "…", "end_hex": "…",
 "task": {"task_id": 7, "x_hex": "…", "y_hex": "…",
          "proof_count": 6, "proof_hash160s": ["<40hex>", …]}}
```

- `x_hex`/`y_hex` **恒等于**该 grant 项的 `current_hex`/`end_hex`（hub 由同一对
  `scan_start`/`scan_end` 生成）——`pow_task()` 本地交叉校验，不符即按无任务处理。
- `proof_count` **可配置**（hub `[pow] proof_count`，默认 6）——客户端按数据走，
  绝不硬编码；`1 + proof_count ≤ 78`（GPU 候选缓冲槽位数）。
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
| 400 | digest 不符 = 客户端/hub 口径不符（真实 hash160 比对下不应发生）→ 大声警告 + `release` 弃窗 + 重领 |
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
| 心跳常量 / 旋转预算（config 解析传入） | `remote.rs:56-63`；`main.rs` `resolve_rotate` | `config.py:67-71` |
| HTTP 封装与超时 | `HubClient`（`remote.rs:300-580`） | `routes.py` |
| 启动 client-auth（条件门禁）+ 解信封 + TOFU | `remote.rs:1029` `maybe_authenticate`；`clientauth.rs` | `routes.py::auth`、`envelope.py` |
| 业务 v2 静态信道信封（包/解） | `remote.rs:373/392` `sealed_body`/`unseal_response`；`clientauth.rs::static_key/encrypt_to_peer/decrypt_from_peer` | `routes.py` 前 `transport.py`/`wire.py`（透明中间件） |
| identity.json 自检 | `clientauth.rs::load_identity` | —（hub 侧由 `nostr.py` 生成） |
| hub.json TOFU（hub 身份跨启动校验） | `clientauth.rs::remember_hub` | — |
| 启动 connect + hash160 校验 | `remote.rs:954` `connect` | `routes.py` |
| **pow 任务解析/校验** | `remote.rs:163-223` `ClaimedChunk::pow_task` | `puzzle.py:348-388`（`task_info`） |
| **pow digest（升序 32B-BE → SHA256）** | `remote.rs:558` `hashed_proof_key` | `powproof.py::expected_digest_for` |
| **pow 收敛（CPU/GPU 共用）** | `remote.rs:591` `converge_pow`；`HubClient::pow` `remote.rs:513` | `routes.py::pow_submit`、`puzzle.py::verify_pow` |
| **proof 归类（压缩公钥 hash160）** | `puzzle.rs:2385` `classify_proof`；`proofs`/`found_proofs` 见 `ScanChunkOptions` `puzzle.rs:1107`、`ProofHit` `puzzle.rs:1149` | `powproof.py::compressed_pubkey/hash160` |
| **GPU 候选表（槽 0 = target，1..=N = proof）** | `gpu/convert.rs::chunk_candidates`；`configure_chunk`（`puzzle.rs:1698` trait + 两 scanner 的 `set_candidates`） | — |
| CPU worker 主循环 | `remote.rs:1179-1481` | — |
| GPU worker 主循环 | `remote.rs:1622-2010` | — |
| status ticker | `remote.rs:1093` | — |
| lease 归属校验（404/409） | `remote.rs:229` `is_lease_lost` | `puzzle.py:160-168` |
| 心跳落库 + lease 刷新 | — | `puzzle.py:443-479`（有 active task → 只续租） |
| done / win / release / pow | `remote.rs:462/473/482/513` | `puzzle.py:605`（release 弃整窗）、`win.py`、`puzzle.py:659`（verify_pow） |
| solved 广播（claim/heartbeat/status 响应） | `remote.rs` 三处停止点 | `routes.py`、`puzzle.py:solved()` |
| 回收循环与孤儿清理 | — | `main.py:26-40`、`puzzle.py:326-379` |
