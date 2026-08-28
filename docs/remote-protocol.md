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

关键设计（`remote.rs:1-19`）：

- **hub 是唯一写者**。worker 从不打开本地 `.db`，断点位置全部通过 HTTP 上报。
- **lease 以 chunk id 区分**，不以 worker 区分——所有线程共享同一个 `worker_id`
  （`remote.rs:614-616`）。
- **崩溃恢复是 hub 的职责**：worker 死了/断网 → 心跳停止 → hub 超时回收 lease，
  chunk 回退 pending 并**保留最后上报的位置**。

---

## 2. 参数一览

### 2.1 Worker 端（`src/remote.rs`）

| 常量/配置 | 值 | 作用 | 位置 |
|---|---|---|---|
| `HEARTBEAT_INTERVAL` | 30s | 扫描期间的心跳节流 | `remote.rs:49` |
| `CLAIM_IDLE` | 2s | 无 chunk 可领时的重试间隔 | `remote.rs:53` |
| `rotate_keys` (CPU) | 默认 2²⁷ = 134,217,728 keys | CPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `cpu_rotate_keys`；`0` 禁用） |
| `gpu_rotate_keys` (GPU) | 默认 2³¹ = 2,147,483,648 keys | GPU 每扫满即 park + 重领；同时作为 claim `capability` 声明给 hub | `main.rs` `resolve_rotate`（CLI/配置 `gpu_rotate_keys`；`0` 禁用） |
| HTTP 连接超时 | 5s | 防 hub 挂死卡线程 | `remote.rs:134` |
| HTTP 单请求超时 | 15s | 同上 | `remote.rs:135` |
| 启动重连 | 每 2s 一次，上限 90s | `connect()` 拿 /api/status | `remote.rs:478` |
| claim 失败日志节流 | 30s | hub 故障时每 30s 才打印一次 | `remote.rs:631` |

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
| `/api/status` | GET | — | puzzle meta + `pending/running/finished` 统计 + 各 worker 概况；`meta.solved/win` 表明是否已有人命中 |
| `/api/chunks/claim` | POST | `{worker_id, count, capability}` | 领取 `count` 个 pending chunk；`capability` 声明"扫多少 keys 后 reclaim"，hub 据此优先分配宽度匹配的 chunk |
| `/api/chunks/{id}/heartbeat` | POST | `{worker_id, current_hex?, end_hex?, keys?, rate?}` | 刷新 lease + 可选保存进度 + 上报速率指标 |
| `/api/chunks/{id}/done` | POST | `{worker_id}` | 整段扫完：置 finished、删 lease |
| `/api/win` | POST | `{worker_id, chunk_id}` | **命中上报**（取代 done）：最终化 chunk + hub 落 win 记录、置 puzzle solved |
| `/api/chunks/{id}/release` | POST | `{worker_id, current_hex?, end_hex?}` | 旋转/放弃：保存进度、回退 pending、删 lease |

响应/错误约定：

- **claim 成功**返回 `{granted, chunks: [{id, current_hex, end_hex}], solved}`；
  `granted=0` 表示 hub 当前没有可领的 pending chunk；`solved=true` 表示别的 worker
  已命中 → worker 应停止。
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
   └─ GET /api/status        ────────→   失败每 2s 重试，上限 90s（remote.rs:478）
   ←──────────────────────────────────── 校验 hash160 一致（不匹配直接退出，exit 2）
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
   └─ 扫完：POST /api/chunks/{id}/done ─→ {worker_id}
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

---

## 8. 源码索引

| 逻辑 | Rust | hub |
|---|---|---|
| 心跳常量 / 旋转预算（config 解析传入） | `remote.rs:43-50`；`main.rs` `resolve_rotate` | `config.py:67-71` |
| HTTP 封装与超时 | `remote.rs:119-208` | `routes.py` |
| 启动 connect + hash160 校验 | `remote.rs:477-528` | — |
| CPU worker 主循环 | `remote.rs:618-826` | — |
| GPU worker 主循环 | `remote.rs:955-1253` | — |
| status ticker | `remote.rs:532-609` | — |
| lease 归属校验（404/409） | `remote.rs:113-115` | `puzzle.py:160-168` |
| 心跳落库 + lease 刷新 | — | `puzzle.py:170-204` |
| done / win / release | `remote.rs:180-235` | `puzzle.py:206-290`、`win.py` |
| solved 广播（claim/heartbeat/status 响应） | `remote.rs` 三处停止点 | `routes.py`、`puzzle.py:solved()` |
| 回收循环与孤儿清理 | — | `main.py:26-40`、`puzzle.py:326-379` |
