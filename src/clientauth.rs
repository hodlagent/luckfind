//! client-auth：启动时用本地 Nostr 身份与 lan-hub 握手，解开 hub 的加密信封
//! 拿 hub 的 Nostr 公钥，与配置里钉住的摘要比对，视为验证通过。
//!
//! 背景（lan-hub `backend/app/envelope.py`，2026-09 定版）：client 首次 auth 时还
//! 不知道 hub 的公钥（它正要从中取得），静态 `ECDH(hub_sec, client_pub)` 会死锁，
//! 所以 hub 用 **一次性 ECDH 密钥**（ECIES）把 `{hub_pubkey, server_time}` 加密到
//! 本 client 的 pubkey 上，明文 POST 只做登记。本模块用本地私钥 + 信封里携带的
//! `ephemeral_pubkey` 解密——解得开 = 声明的 pubkey 与本地私钥匹配（未被人冒充/篡改）。
//!
//! **但这只证明「收件人能解开」，不证明「发送方持 hub 私钥」**——ECIES 的发送方用
//! 一次性密钥，任何人都能对着本 client 的公钥封一个自称 hub 的信封。所以「首次接触
//! 就学会 hub 公钥」（TOFU）不是信任根：LAN 中间人只要抢在真 hub 之前应答第一次握手，
//! 就能把本机的私钥/命中交到它手上。这里改为**钉摘要**（[`verify_hub_pinned`]）：
//! 配置 `[remote] npub_sha256` 预先钉住 hub 身份，没有可被冒充的首次接触，也不落任何
//! 状态文件。摘要 = `sha256(hub 的 npub 文本)`，与 hub 机 `identity.json` 的
//! `npub_sha256` 字段、`scripts/nostr.py` 同口径（[`npub_sha256`]）。
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
//!
//! **v2 静态信道（业务加解密，client ↔ hub 双向）**：auth 握手后 hub 公钥已知，双方
//! 各持对方静态公钥，共享 pair-key `K = SHA-256(x(ECDH(本地私钥, even-lift(对端
//! x-only))))`（同 v1 的取点路径，只是对端是一把静态公钥而非一次性 eph）。信封
//! `{ "v": 2, "nonce": "<24hex>", "ciphertext": "<hex>" }`；aad 绑定**发送方**：
//! client→hub 请求用 client 自身 pubkey（`encrypt_to_peer`），hub→client 响应用 hub
//! pubkey（`decrypt_from_peer` 的对端即发送方）。细节见 envelope.py / docs/remote-api.md。

use std::fs;
use std::path::Path;

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use bech32::{Bech32, Hrp};
use rand::TryRng;
use secp256k1::ecdh::shared_secret_point;
use secp256k1::{Parity, PublicKey, Secp256k1, SecretKey, XOnlyPublicKey};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// 与 envelope.py `ENVELOPE_VERSION` 对齐的信封版本。
const ENVELOPE_VERSION: u64 = 1;

/// 与 envelope.py `STATIC_ENVELOPE_VERSION` 对齐的 v2 静态信道版本。
const STATIC_ENVELOPE_VERSION: u64 = 2;

/// AES-GCM nonce 长度（v1/v2 一致）。
const NONCE_BYTES: usize = 12;

/// 本地 Nostr 身份（identity.json 的 pubkey / secret_key，均 64-hex 小写 x-only）。
pub struct ClientIdentity {
    pub pubkey: String,
    pub secret: String,
}

/// 解开 auth 信封后得到的明文载荷。
#[derive(Debug, Clone)]
pub struct AuthReply {
    pub hub_pubkey: String,
    /// hub 的时间戳。生产代码不再读它（身份校验已由钉摘要取代），保留是因为信封
    /// 解密单测拿它断言「整份明文都对得上 python 向量」。
    #[allow(dead_code)]
    pub server_time: String,
}

#[derive(Deserialize)]
struct IdentityFile {
    pubkey: String,
    secret_key: String,
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

    // shared = client_sec × even-lift(eph_pub) 的裸 x 坐标（32B）→ pair-key
    // （对端是一次性 eph 公钥，其余与 v2 静态信道同一条 pair_key 路径）。
    let eph_xonly = XOnlyPublicKey::from_byte_array(arr32(eph_x, "ephemeral_pubkey")?)
        .map_err(|e| format!("ephemeral_pubkey 不是合法曲线点：{e}"))?;
    let secret = SecretKey::from_byte_array(arr32(secret_bytes, "secret_key")?)
        .map_err(|e| format!("secret_key 不是合法标量：{e}"))?;
    let key = pair_key(&secret, &eph_xonly);

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
        // 统一小写（对齐另两端：hub 恒发小写、vanity 显式 .lower()）——钉扎摘要按
        // npub 文本算，收口成小写避免将来 hub 侧大小写漂移导致摘要对不上。
        hub_pubkey: str_field(&payload, "hub_pubkey")?.to_ascii_lowercase(),
        server_time: str_field(&payload, "server_time")?,
    })
}

// ── 2b. v2 静态信道（业务请求/响应加解密；见模块 docstring）──────────────────

/// 一对密钥的 pair-key = SHA-256( x(ECDH(local_secret, even-lift(peer x-only))) )。
/// v1 auth 解密与 v2 静态信道共用（对端分别是一次性 eph / 静态对端公钥）。镜像
/// envelope.py `static_key`：x-only 都做 even 升点再取 ECDH 裸 x 前 32B。
fn pair_key(secret: &SecretKey, peer_xonly: &XOnlyPublicKey) -> [u8; 32] {
    let peer_pk = PublicKey::from_x_only_public_key(*peer_xonly, Parity::Even);
    let xy = shared_secret_point(&peer_pk, secret);
    Sha256::digest(&xy[..32]).into()
}

fn secret_from_hex(secret_hex: &str) -> Result<SecretKey, String> {
    let bytes = hex_bytes(secret_hex, "secret_key", Some(32))?;
    SecretKey::from_byte_array(arr32(bytes, "secret_key")?)
        .map_err(|e| format!("secret_key 不是合法标量：{e}"))
}

fn xonly_from_hex(pub_hex: &str, what: &str) -> Result<XOnlyPublicKey, String> {
    let bytes = hex_bytes(pub_hex, what, Some(32))?;
    XOnlyPublicKey::from_byte_array(arr32(bytes, what)?)
        .map_err(|e| format!("{what} 不是合法曲线点：{e}"))
}

/// 本地身份 + 对端静态公钥这对的共享 pair-key（[u8;32]，镜像 envelope.py `static_key`）。
/// 本 client 调 (id, hub_pub) 即可派生出与 hub 相同的那把 K。
pub fn static_key(id: &ClientIdentity, peer_pub_hex: &str) -> Result<[u8; 32], String> {
    let secret = secret_from_hex(&id.secret)?;
    let peer = xonly_from_hex(peer_pub_hex, "peer_pub")?;
    Ok(pair_key(&secret, &peer))
}

/// v2 信封加密（固定 nonce——测试注入用；运行时由 encrypt_to_peer 生成随机 nonce）。
/// aad = 发送方 = 本 client 自身 pubkey（envelope.py `encrypt_static` aad_pub_hex）。
fn encrypt_with_nonce(
    id: &ClientIdentity,
    peer_pub_hex: &str,
    plaintext: &[u8],
    nonce: [u8; NONCE_BYTES],
) -> Result<Value, String> {
    let key = static_key(id, peer_pub_hex)?;
    let own_pub = hex_bytes(&id.pubkey, "pubkey", Some(32))?;
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let ct = cipher
        .encrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: plaintext,
                aad: &own_pub,
            },
        )
        .map_err(|_| "AES-256-GCM 加密失败".to_string())?;
    Ok(serde_json::json!({
        "v": STATIC_ENVELOPE_VERSION,
        "nonce": hex::encode(nonce),
        "ciphertext": hex::encode(ct),
    }))
}

/// client→hub 业务请求加密：把内层明文 JSON bytes 包成 v2 信封（aad = client pubkey），
/// remote.rs 再放到外层 `{"pubkey": …, "enc": env}`。nonce 每次随机。
pub fn encrypt_to_peer(
    id: &ClientIdentity,
    peer_pub_hex: &str,
    plaintext: &[u8],
) -> Result<Value, String> {
    let mut nonce = [0u8; NONCE_BYTES];
    rand::rngs::SysRng
        .try_fill_bytes(&mut nonce)
        .map_err(|e| format!("生成 nonce 随机数失败：{e}"))?;
    encrypt_with_nonce(id, peer_pub_hex, plaintext, nonce)
}

/// 解开对端（hub）发来的 v2 信封，返回明文 bytes。aad = 发送方 = 对端 pubkey
/// （envelope.py `decrypt_static` aad_pub_hex）。v≠2 / tag 校验失败 → Err。
pub fn decrypt_from_peer(
    id: &ClientIdentity,
    peer_pub_hex: &str,
    env: &Value,
) -> Result<Vec<u8>, String> {
    if env.get("v").and_then(Value::as_u64).unwrap_or(0) != STATIC_ENVELOPE_VERSION {
        let v = env.get("v");
        return Err(format!("不支持的信封版本 {v:?}（应为 {STATIC_ENVELOPE_VERSION}）"));
    }
    let nonce = hex_bytes(&str_field(env, "nonce")?, "nonce", Some(NONCE_BYTES))?;
    let ct = hex_bytes(&str_field(env, "ciphertext")?, "ciphertext", None)?;
    if ct.len() < 16 {
        return Err("ciphertext 长度非法（应含 16B GCM tag）".to_string());
    }
    let key = static_key(id, peer_pub_hex)?;
    let peer_pub = hex_bytes(peer_pub_hex, "peer_pub", Some(32))?; // aad = 发送方（对端）
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
    let plain = cipher
        .decrypt(
            Nonce::from_slice(&nonce),
            Payload {
                msg: &ct,
                aad: &peer_pub,
            },
        )
        .map_err(|_| {
            "AES-GCM 解密失败：信封不是发给本机 / 对端密钥不符 / 被篡改".to_string()
        })?;
    Ok(plain)
}

// ── 3. hub 身份钉扎（取代 TOFU hub.json）────────────────────────────────────

/// Nostr 的 bech32 人类可读前缀（npub = 公钥、nsec = 私钥）。小写 4 字符，满足
/// BIP-173 的 [33,126] 可打印 ASCII 约束，恒为合法 hrp。
const NPUB_HRP: &str = "npub";

/// 把 32B x-only pubkey（64-hex）编成 Nostr `npub1…`。
///
/// **用 bech32（BIP-173），不是 bech32m**：NIP-19 的地址就是这个校验和变体，换一个
/// 会得到完全不同的尾校验和，与 hub 的 `nostr.py::npub` 对不上。8→5 bit 的 convertbits
/// 由 bech32 crate 内部完成，与 python `convertbits(data, 8, 5, True)` 一致。
pub fn npub_of(pubkey_hex: &str) -> Result<String, String> {
    let bytes = hex_bytes(pubkey_hex, "pubkey", Some(32))?;
    // "npub" 恒合法（非空、全小写 ASCII）；仍走 Result 而非 unwrap，不留 panic 路径。
    let hrp = Hrp::parse(NPUB_HRP).map_err(|e| format!("npub hrp 非法（内部错误）：{e}"))?;
    bech32::encode::<Bech32>(hrp, &bytes).map_err(|e| format!("npub 编码失败：{e}"))
}

/// hub 身份钉扎摘要 = **`npub` 文本**的 SHA-256（小写 hex）——与 hub 机
/// `backend/data/identity.json` 的 `npub_sha256` 字段、`scripts/nostr.py` 同一口径。
///
/// 注意哈希的是 npub **字符串**的 utf-8 字节，**不是** pubkey 的 32B 原始字节：
/// 摘要要能和 hub 侧直接对拍，口径必须逐字节一致。全仓只此一处定义。
pub fn npub_sha256(pubkey_hex: &str) -> Result<String, String> {
    Ok(hex::encode(Sha256::digest(npub_of(pubkey_hex)?.as_bytes())))
}

/// 校验 auth 信封里学到的 hub 公钥是否就是配置钉住的那一个（[`npub_sha256`] 比对）。
///
/// 钉摘要与直接钉 npub 强度等价（SHA-256 抗原像），好处是钉扎值本身不泄露 hub 身份。
/// `pinned` 先 trim + 折成小写（手抄进配置的大小写/空白不该误判）；实算值恒为小写 hex。
pub fn verify_hub_pinned(hub_pubkey: &str, pinned: &str) -> Result<(), String> {
    let got = npub_sha256(hub_pubkey)?; // 先算：失败即返回，也保证下面的前缀切片安全
    let want = pinned.trim().to_ascii_lowercase();
    if got != want {
        return Err(format!(
            "hub 身份与配置的 [remote] npub_sha256 不符：\n        配置 {want}\n        实得 {got}\n        \
             实得 npub {}\n        若确实换过 hub，请用上面的摘要更新 config 后重试。",
            npub_of(hub_pubkey)?,
        ));
    }
    Ok(())
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

    // ── v2 静态信道向量（与 backend/scripts/test_wire.py 同一组固定密钥/nonce）──
    const V2_CLIENT_PUB: &str = "84bf7562262bbd6940085748f3be6afa52ae317155181ece31b66351ccffa4b0";
    const V2_CLIENT_SECRET: &str =
        "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20";
    const V2_HUB_PUB: &str = "207bba70bc66309baa582a6ac120fd52d68026c51f6326f8ccedcbd2c1b7eb82";
    const V2_HUB_SECRET: &str =
        "2122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f40";
    const V2_NONCE: [u8; NONCE_BYTES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    const V2_REQ_PLAIN: &str = r#"{"worker_id":"84bf7562262bbd6940085748f3be6afa52ae317155181ece31b66351ccffa4b0","current_hex":"0000000000000001"}"#;
    const V2_REQ_CIPHER: &str = "0e89f99a89a1b2c96728c353804794531da7c1403e201177cf686d47fde1c85f4f87e001c8bb1dfc8ab991b79b2b894361b8d1d44b7bd6b051fcc3f30902880973b6cb3599207691fcc0d21f1f791bde58d1b1a81df19b74f5b77f36b48510b4f53870079134091d03ed2bdd094846cac812bf7c750e4c4ea14047adbf4a0078b8";
    const V2_RESP_PLAIN: &str = r#"{"ok":true}"#;
    const V2_RESP_CIPHER: &str = "0e89e19ed9f0a3c94d24daabe641db20610853acd4a509b0da3ce2";

    fn v2_id() -> ClientIdentity {
        ClientIdentity {
            pubkey: V2_CLIENT_PUB.to_string(),
            secret: V2_CLIENT_SECRET.to_string(),
        }
    }

    #[test]
    fn v2_encrypt_matches_python_request_vector() {
        // client→hub 请求方向：Rust 加密（固定 nonce）应与 python encrypt_static 产出的
        // ciphertext 逐字节一致（aad = client 自身 pubkey）。
        let env = encrypt_with_nonce(&v2_id(), V2_HUB_PUB, V2_REQ_PLAIN.as_bytes(), V2_NONCE)
            .expect("encrypt");
        assert_eq!(env["v"].as_u64(), Some(STATIC_ENVELOPE_VERSION));
        let nonce_hex = hex::encode(V2_NONCE);
        assert_eq!(env["nonce"].as_str(), Some(nonce_hex.as_str()));
        assert_eq!(env["ciphertext"].as_str(), Some(V2_REQ_CIPHER));
    }

    #[test]
    fn v2_decrypts_python_response_vector() {
        // hub→client 响应方向：Rust 解密 python encrypt_static(hub_secret,…) 的信封
        // （aad = hub 自身 pubkey = 本 client 视角的对端）。
        let env = serde_json::json!({
            "v": STATIC_ENVELOPE_VERSION,
            "nonce": hex::encode(V2_NONCE),
            "ciphertext": V2_RESP_CIPHER,
        });
        let plain = decrypt_from_peer(&v2_id(), V2_HUB_PUB, &env).expect("decrypt");
        assert_eq!(plain, V2_RESP_PLAIN.as_bytes());
    }

    #[test]
    fn v2_static_pair_key_is_symmetric() {
        // client 用 (client_sec, hub_pub)、hub 用 (hub_sec, client_pub) 应派生出同一把 K。
        let k_client = static_key(&v2_id(), V2_HUB_PUB).unwrap();
        let hub_id = ClientIdentity {
            pubkey: V2_HUB_PUB.to_string(),
            secret: V2_HUB_SECRET.to_string(),
        };
        let k_hub = static_key(&hub_id, V2_CLIENT_PUB).unwrap();
        assert_eq!(k_client, k_hub);
        // 方向性：aad 绑定发送方，所以「hub 加密 → client 解密」才是本机可自证的一对
        // （client 只能解 hub 发来的响应；自己发的请求要 hub 那边解）。
        let env = encrypt_to_peer(&hub_id, V2_CLIENT_PUB, V2_RESP_PLAIN.as_bytes())
            .expect("hub encrypt");
        let plain = decrypt_from_peer(&v2_id(), V2_HUB_PUB, &env).expect("client decrypt");
        assert_eq!(plain, V2_RESP_PLAIN.as_bytes());
        // 对端换钥（不是真 hub）→ 解不开。
        let other_hub = ClientIdentity {
            pubkey: "2b6e171dcdeb17ea0cef3f95b1e3f3f9e8b8bb8a8f2a1b0c1d2e3f4a5b6c7d8e9".to_string(),
            secret: "303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f50".to_string(),
        };
        // 用错误 hub 公钥解密（pair-key 派生自错误对端）→ GCM tag 失败。
        assert!(decrypt_from_peer(&v2_id(), &other_hub.pubkey, &env).is_err());
    }

    // ── hub 钉扎向量：pubkey → npub → sha256(npub 文本)，与 hub（python）已知值对齐 ──
    // 取自 hub 机 backend/data/identity.json；`bin/config.toml` 的 [remote] npub_sha256
    // 也是这个值，等于把「配置里的钉扎值确实对得上真 hub」这条也锁进测试。
    const PIN_HUB_PUB: &str = "cda87e1ddb41430d6f9ce267b6b048c9b105b5bf2002c5aeac97e0cf6dbb04ae";
    const PIN_HUB_NPUB: &str = "npub1ek58u8wmg9ps6muuufnmdvzgexcstddlyqpvtt4vjlsv7mdmqjhq44q9wl";
    const PIN_HUB_SHA: &str = "709036f6387523b2203bc46047a2f13383e5ec48e86a64ddbada2a776cc00c55";

    #[test]
    fn npub_matches_hub_vector() {
        assert_eq!(npub_of(PIN_HUB_PUB).unwrap(), PIN_HUB_NPUB);
        // 大写 hex 是同一个公钥（hex_bytes 接受 A-F）→ 同一 npub。
        assert_eq!(
            npub_of(&PIN_HUB_PUB.to_ascii_uppercase()).unwrap(),
            PIN_HUB_NPUB
        );
    }

    #[test]
    fn npub_sha256_hashes_the_npub_text() {
        let got = npub_sha256(PIN_HUB_PUB).unwrap();
        assert_eq!(got, PIN_HUB_SHA);
        assert_eq!(got.len(), 64);
        assert!(got.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        // 锁死「哈希 npub 文本」这条口径：若哪天误改成哈希 32B pubkey 原始字节，值就变了。
        let raw = Sha256::digest(hex::decode(PIN_HUB_PUB).unwrap());
        assert_ne!(got, hex::encode(raw));
    }

    #[test]
    fn npub_of_rejects_non_32b_pubkeys() {
        assert!(npub_of("").is_err());
        assert!(npub_of("zz").is_err()); // 非 hex
        assert!(npub_of(&"11".repeat(31)).is_err()); // 31B
        assert!(npub_of(&"11".repeat(33)).is_err()); // 33B
    }

    #[test]
    fn verify_hub_pinned_accepts_only_the_pinned_digest() {
        assert!(verify_hub_pinned(PIN_HUB_PUB, PIN_HUB_SHA).is_ok());
        // 手抄进配置的大小写/空白容错。
        let sloppy = format!("  {}  ", PIN_HUB_SHA.to_ascii_uppercase());
        assert!(verify_hub_pinned(PIN_HUB_PUB, &sloppy).is_ok());
        // 另一个 hub（v1 向量的 HUB_PUBKEY）→ 拒绝。
        let e = verify_hub_pinned(HUB_PUBKEY, PIN_HUB_SHA).unwrap_err();
        assert!(e.contains(PIN_HUB_SHA), "错误文案要带上配置侧的摘要：{e}");
        // 非法 hub pubkey / 空 pin → Err 而非 panic（前缀/切片安全）。
        assert!(verify_hub_pinned("not-hex", PIN_HUB_SHA).is_err());
        assert!(verify_hub_pinned(PIN_HUB_PUB, "").is_err());
    }
}
