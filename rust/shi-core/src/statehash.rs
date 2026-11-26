//! 状态哈希：可复现性的证据
//!
//! 同一个种子跑两次，状态哈希序列必须完全一致。
//! 这是整个研究的地基。
//!
//! 实现
//! ----
//! 用 FNV-1a 64 位哈希（快、无依赖、确定性）。
//! 对浮点数按位哈希：-0.0 与 0.0 不同，NaN 的不同位模式也不同。
//! 这是刻意的——我们要的是「逐位一致」，不是「数值相等」。

/// FNV-1a 64 位哈希。
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a {
    state: u64,
}

impl Fnv1a {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    pub fn new() -> Self {
        Self {
            state: Self::OFFSET,
        }
    }

    #[inline]
    pub fn write_u64(&mut self, v: u64) {
        for byte in v.to_le_bytes() {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    #[inline]
    pub fn write_f64(&mut self, v: f64) {
        self.write_u64(v.to_bits());
    }

    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    pub fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

/// 计算世界状态的哈希。
///
/// 哈希内容：tick + 所有个体位置 + 环境场。
pub fn state_hash(positions: &[f64], env: &[f64], tick: u64) -> u64 {
    let mut h = Fnv1a::new();
    h.write_u64(tick);
    h.write_u64(positions.len() as u64);
    for &v in positions {
        h.write_f64(v);
    }
    h.write_u64(env.len() as u64);
    for &v in env {
        h.write_f64(v);
    }
    h.finish()
}

/// 把哈希转成 16 位十六进制字符串。
pub fn hash_hex(h: u64) -> String {
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let pos = vec![1.0, 2.0, 3.0];
        let env = vec![0.0; 100];
        assert_eq!(state_hash(&pos, &env, 0), state_hash(&pos, &env, 0));
    }

    #[test]
    fn position_change_detected() {
        let env = vec![0.0; 100];
        assert_ne!(
            state_hash(&[1.0, 2.0], &env, 0),
            state_hash(&[1.0, 2.0000001], &env, 0)
        );
    }

    #[test]
    fn env_change_detected() {
        let pos = vec![1.0, 2.0];
        let mut e1 = vec![0.0; 100];
        let e2 = {
            let mut v = vec![0.0; 100];
            v[42] = 1e-12;
            v
        };
        e1[0] = 0.0;
        assert_ne!(state_hash(&pos, &e1, 0), state_hash(&pos, &e2, 0));
    }

    #[test]
    fn tick_participates() {
        let pos = vec![1.0];
        let env = vec![0.0; 10];
        assert_ne!(state_hash(&pos, &env, 0), state_hash(&pos, &env, 1));
    }

    #[test]
    fn negative_zero_differs() {
        let env = vec![0.0; 10];
        assert_ne!(
            state_hash(&[0.0], &env, 0),
            state_hash(&[-0.0], &env, 0)
        );
    }

    #[test]
    fn hex_format() {
        let h = state_hash(&[1.0], &[0.0], 0);
        let s = hash_hex(h);
        assert_eq!(s.len(), 16);
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
