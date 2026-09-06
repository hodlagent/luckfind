//! client-auth：启动时用本地 Nostr 身份与 lan-hub 握手，解开 hub 的加密信封
//! 拿 hub 的 Nostr 公钥（TOFU），视为验证通过。
//!
//! 背景（lan-hub `backend/app/envelope.py`，2026-09 定版）：client 首次 auth 时还
//! 不知道 hub 的公钥（它正要从中取得），静态静态 `ECDH(hub_sec, client_pub)` 会死锁，
//! 所以 hub 用 **一次性 ECDH 密钥**（ECIES）把 `{hub_pubkey, server_time}` 加密到
//! 本 client 的 pubkey 上，明文 POST 只做登记。本模块用本地私钥 + 信封里携带的
//! `ephemeral_pubkey` 解密——解得开 = 声明的 pubkey 与本地私钥匹配（未被人冒充/篡改）；
//! 解开后学到的 `hub_pubkey` 写入 `hub.json`（TOFU），跨启动校验 hub 身份未变。
//!
//! 算法（与 envelope.py 逐字节一致，见其模块 docstring）：
//! - 公钥按 Nostr 惯例取 x-only（32B）。**双方都对 x-only 做 even 升点**（`02‖x`）
//!   再算 ECDH——`d·(−P)=−dP` 只翻 Y 不翻 X，所以各自拿到的 shared **x 坐标**相同。
//! - `shared =` ECDH 点的 **x 坐标 32B**。注意：**不用**
//!   `secp256k1::ecdh::SharedSecret`——它的默认哈希是「压缩点（带 Y 奇偶前缀）」的
//!   SHA-256，随升点奇偶而变，跨 even-lift 不对齐；这里用 `shared_secret_point` 取
//!   裸 x‖y 再截前 32B（对应 python `_shared_x` 手动取点）。
//! - `key = SHA-256(shared)`；`nonce = 12B`；`aad = 本 client pubkey 的 32B`。
//! - `AES-256-GCM`：ciphertext‖16B tag，传输小写 hex。
//!
//! 信封 JSON（/api/auth 响应 `auth` 键）：
//!     { "v": 1, "ephemeral_pubkey": "<64hex>", "nonce": "<24hex>", "ciphertext": "<hex>" }
//! 被加密的 plaintext 为 UTF-8 JSON：{"hub_pubkey": "<64hex>", "server_time": "…UTC…"}。

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use secp256k1::ecdh::shared_secret_point;
use secp256k1::{Parity, PublicKey, Secp256k1, SecretKey, XOnlyPublicKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// 与 envelope.py `ENVELOPE_VERSION` 对齐的信封版本。
const ENVELOPE_VERSION: u64 = 1;

/// 本地 Nostr 身份（identity.json 的 pubkey / secret_key，均 64-hex 小写 x-only）。
pub struct ClientIdentity {
    pub pubkey: String,
    pub secret: String,
}

/// 解开 auth 信封后得到的明文载荷。
#[derive(Debug, Clone)]
pub struct AuthReply {
    pub hub_pubkey: String,
    pub server_time: String,
}

#[derive(Deserialize)]
struct IdentityFile {
    pubkey: String,
    secret_key: String,
}

#[derive(Serialize, Deserialize)]
struct HubState {
    hub_pubkey: String,
    server_time: String,
    base: String,
    first_seen: String,
    last_seen: String,
}

// ── 小工具 ──────────────────────────────────────────────────────────────────

fn hex_bytes(s: &str, what: &str, want: Option<usize>) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("{what} 应为 hex，实为 {s:?}"));
    }
    if let Some(n) = want {
        if s.len() != n * 2 {
            return Err(format!("{what} 应为 {} 位 hex，实为 {s:?}", n * 2));
        }
    }
    hex::decode(s).map_err(|_| format!("{what} 不是合法 hex"))
}

/// 定长 32B hex 已经 `hex_bytes` 校验过长度，这里只做类型收窄。
fn arr32(v: Vec<u8>, what: &str) -> Result<[u8; 32], String> {
    v.try_into()
        .map_err(|_| format!("{what} 应为 32 字节（内部错误）"))
}

fn str_field(obj: &Value, key: &str) -> Result<String, String> {
    obj.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("缺少字符串字段 {key}"))
}

fn now_stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// 以 0600 权限写文件（身份/TOFU 文件绝不公开）。非 unix 回退普通写。
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        let mut f = opts
            .open(path)
            .map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
        f.write_all(bytes)
            .map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
        f.sync_all()
            .map_err(|e| format!("写 {} 失败：{e}", path.display()))?;
        Ok(())
    }
    #[cfg(not(unix))]
    {
        fs::write(path, bytes).map_err(|e| format!("写 {} 失败：{e}", path.display()))
    }
}

// ── 1. 身份加载 ─────────────────────────────────────────────────────────────

/// 读 identity.json 并自检：用 secret_key 由 secp256k1 派生 x-only 公钥，与文件的
/// pubkey 比对。文件缺失或不自洽都返回 Err（调用方 exit 2，绝不静默重建/降级）。
pub fn load_identity(path: &Path) -> Result<ClientIdentity, String> {
    let text = fs::read_to_string(path).map_err(|e| {
        format!(
            "读取身份文件 {} 失败：{e}\n        若尚无身份，请先在本目录运行 identify 或 `nostr.py create` 生成 identity.json",
            path.display()
        )
    })?;
    let file: IdentityFile = serde_json::from_str(&text)
        .map_err(|e| format!("{} 不是合法的 identity.json：{e}", path.display()))?;
    let secret_bytes = hex_bytes(&file.secret_key, "secret_key", Some(32))
        .map_err(|e| format!("{} {e}", path.display()))?;
    let pub_bytes = hex_bytes(&file.pubkey, "pubkey", Some(32))
        .map_err(|e| format!("{} {e}", path.display()))?;

    let secp = Secp256k1::new();
    let secret = SecretKey::from_byte_array(arr32(secret_bytes, "secret_key")?)
        .map_err(|e| format!("{} secret_key 不是合法标量：{e}", path.display()))?;
    let derived = PublicKey::from_secret_key(&secp, &secret)
        .x_only_public_key()
        .0
        .serialize();
    if derived.as_slice() != pub_bytes.as_slice() {
        return Err(format!(
            "{} 不自洽：pubkey 与由 secret_key 派生的公钥不一致。\n        请删除后重新运行 identify / `nostr.py create`",
            path.display()
        ));
    }
    Ok(ClientIdentity {
        pubkey: file.pubkey.to_ascii_lowercase(),
        secret: file.secret_key.to_ascii_lowercase(),
    })
}

// ── 2. 信封解密 ─────────────────────────────────────────────────────────────

/// 用本地私钥解开 /api/auth 响应的 `auth` 信封（ECIES v1，与 envelope.py 逐字节对齐）。
/// AES-GCM tag 校验失败 = 信封不是发给这把私钥的（pubkey 声明与私钥不符/被篡改）→ Err。
pub fn decrypt_auth(id: &ClientIdentity, env: &Value) -> Result<AuthReply, String> {
    let secret_bytes = hex_bytes(&id.secret, "secret_key", Some(32))?;
    let own_pub_bytes = hex_bytes(&id.pubkey, "pubkey", Some(32))?; // aad = 收件人 pubkey

    if env.get("v").and_then(Value::as_u64).unwrap_or(0) != ENVELOPE_VERSION {
        let v = env.get("v");
        return Err(format!(
            "不支持的信封版本 {v:?}（应为 {ENVELOPE_VERSION}）"
        ));
    }
    let eph_hex = str_field(env, "ephemeral_pubkey")?;
    let nonce_hex = str_field(env, "nonce")?;
    let ct_hex = str_field(env, "ciphertext")?;

    let eph_x = hex_bytes(&eph_hex, "ephemeral_pubkey", Some(32))?;
    let nonce = hex_bytes(&nonce_hex, "nonce", Some(12))?;
    let ct = hex_bytes(&ct_hex, "ciphertext", None)?;
    if ct.len() < 16 {
        return Err("ciphertext 长度非法（应含 16B GCM tag）".to_string());
    }

    // shared = client_sec × even-lift(eph_pub) 的裸 x 坐标（32B）。
    let eph_xonly = XOnlyPublicKey::from_byte_array(arr32(eph_x, "ephemeral_pubkey")?)
        .map_err(|e| format!("ephemeral_pubkey 不是合法曲线点：{e}"))?;
    let eph_pk = PublicKey::from_x_only_public_key(eph_xonly, Parity::Even);
    let secret = SecretKey::from_byte_array(arr32(secret_bytes, "secret_key")?)
        .map_err(|e| format!("secret_key 不是合法标量：{e}"))?;
    let xy = shared_secret_point(&eph_pk, &secret);
    let key = Sha256::digest(&xy[..32]);

    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let plain = cipher
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &ct,
                aad: &own_pub_bytes,
            },
        )
        .map_err(|_| {
            "AES-GCM 解密失败：信封不是发给这把私钥的（pubkey 声明与私钥不符或包被篡改）".to_string()
        })?;

    let payload: Value = serde_json::from_slice(&plain)
        .map_err(|e| format!("解密后不是合法 JSON：{e}"))?;
    Ok(AuthReply {
        hub_pubkey: str_field(&payload, "hub_pubkey")?,
        server_time: str_field(&payload, "server_time")?,
    })
}

// ── 3. TOFU hub.json ────────────────────────────────────────────────────────

/// hub.json 与 identity.json 同目录。
pub fn hub_json_path(identity_path: &Path) -> PathBuf {
    identity_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.join("hub.json"))
        .unwrap_or_else(|| PathBuf::from("hub.json"))
}

/// 握手成功后把 hub 公钥记入 TOFU。若旧记录与新公钥**不一致** → Err（调用方 exit 2：
/// hub 的 Nostr 身份不应改变；确需接受新 hub 时删除 hub.json 后重试）。一致则幂等
/// 更新 last_seen（first_seen 保留）。
pub fn remember_hub(path: &Path, reply: &AuthReply, base: &str) -> Result<(), String> {
    let now = now_stamp();
    let mut first_seen = now.clone();

    if let Some(text) = fs::read_to_string(path).ok() {
        match serde_json::from_str::<HubState>(&text) {
            Ok(prev) => {
                if prev.hub_pubkey != reply.hub_pubkey {
                    return Err(format!(
                        "TOFU hub 身份变化：{} 此前记录 {}，本次握手拿到 {}。\n        hub 的 Nostr 公钥不应改变；确需接受新 hub 身份时删除 {} 后重试。",
                        path.display(),
                        abbrev(&prev.hub_pubkey),
                        abbrev(&reply.hub_pubkey),
                        path.display(),
                    ));
                }
                first_seen = prev.first_seen;
            }
            Err(_) => {
                // hub.json 损坏：按无记录处理（重新写），不因坏文件误拒新握手。
                first_seen = now.clone();
            }
        }
    }

    let state = HubState {
        hub_pubkey: reply.hub_pubkey.clone(),
        server_time: reply.server_time.clone(),
        base: base.to_string(),
        first_seen,
        last_seen: now,
    };
    let text = serde_json::to_string_pretty(&state)
        .map_err(|e| format!("序列化 hub.json 失败：{e}"))?;
    write_private(path, text.as_bytes())
}

/// 打印用：取 16 位前缀 + “…”（公钥完整 64-hex 已存 hub.json，不进日志）。
fn abbrev(s: &str) -> String {
    let p: String = s.chars().take(16).collect();
    format!("{p}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    // 与 backend/scripts/test_auth.py 联动的确定性向量（由 envelope.py 固定输入生成，
    // 见验证步骤：client_secret = "11"×32，eph 为 odd-parity 定值、nonce 定值）。
    const CLIENT_PUB: &str = "4f355bdcb7cc0af728ef3cceb9615d90684bb5b2ca5f859ab0f0b704075871aa";
    const CLIENT_SECRET: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    const ENV: &str = r#"{"v":1,"ephemeral_pubkey":"117e958d8c78e4c208f4b6b6f136ce145c0e3de5665bc531e3a31f7ab3f4a26a","nonce":"000102030405060708090a0b","ciphertext":"51da6dcab8ef8e77a3b676885929757bd2ee7d9b2c94ac7681e65155dfa0cfbb4319faad05906d58b2cff84a8fb5b6752c2ea567bba3c43f56415f1036ded823f0d068023d1c87ecb9e657bfeb21321bb0ba2cb8fc3eb61559805de86d95c6209d001720a6acdaa108812183803f391918addbaa05a3f20e78c78d93ce546fbd7e4bb00d6fb4f5f4"}"#;
    const HUB_PUBKEY: &str = "a8c9d8768a894d4d1ea6aa304e4aec8a8578f4c6c58dec8a061b160ed2c9b405";
    const SERVER_TIME: &str = "2026-09-06 05:11:54";

    #[test]
    fn decrypts_python_vector() {
        let id = ClientIdentity {
            pubkey: CLIENT_PUB.to_string(),
            secret: CLIENT_SECRET.to_string(),
        };
        let env: Value = serde_json::from_str(ENV).unwrap();
        let reply = decrypt_auth(&id, &env).expect("decrypt should succeed");
        assert_eq!(reply.hub_pubkey, HUB_PUBKEY);
        assert_eq!(reply.server_time, SERVER_TIME);
    }

    #[test]
    fn wrong_secret_is_rejected() {
        // secret 改成别的值 → 派生公钥与 aad 不匹配 → GCM tag 校验失败。
        let id = ClientIdentity {
            pubkey: CLIENT_PUB.to_string(),
            secret: "2222222222222222222222222222222222222222222222222222222222222222".to_string(),
        };
        let env: Value = serde_json::from_str(ENV).unwrap();
        assert!(decrypt_auth(&id, &env).is_err());
    }
}
