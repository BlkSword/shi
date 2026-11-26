//! 确定性随机数生成器
//!
//! 设计约束
//! --------
//! 1. 同种子必须产生完全相同的序列（跨进程、跨机器）
//! 2. 每个个体拥有独立的随机流，避免并行化破坏确定性
//! 3. 禁止使用 thread_rng 等不确定源
//!
//! 两种用法
//! --------
//! - **有状态流**（`RngRegistry`）：适合需要连续随机序列的场景
//! - **无状态派生**（`DeriveRng`）：适合「按需」随机，可在任意顺序调用

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// 无状态确定性随机派生。
///
/// 给定 (seed, tick, entity_id, salt) 直接算出结果，不依赖任何内部状态。
/// 因此可以在任意顺序、任意并行度下调用。
#[derive(Debug, Clone, Copy)]
pub struct DeriveRng {
    seed: u64,
}

impl DeriveRng {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// 派生一个 64 位整数（splitmix64 变体）。
    #[inline]
    fn hash64(&self, tick: u64, entity_id: u64, salt: u64) -> u64 {
        let mut z = self
            .seed
            .wrapping_add(tick.wrapping_mul(0x9E3779B97F4A7C15))
            .wrapping_add(entity_id.wrapping_mul(0xBF58476D1CE4E5B9))
            .wrapping_add(salt.wrapping_mul(0x94D049BB133111EB));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// 派生一个 [0, 1) 的浮点数。
    #[inline]
    pub fn derive(&self, tick: u64, entity_id: u64, salt: u64) -> f64 {
        (self.hash64(tick, entity_id, salt) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 派生一个 [0, n) 的整数。
    #[inline]
    pub fn derive_int(&self, tick: u64, entity_id: u64, n: u64, salt: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.hash64(tick, entity_id, salt) % n
    }

    /// 派生一个标准正态分布的值（Box-Muller）。
    #[inline]
    pub fn derive_normal(&self, tick: u64, entity_id: u64, salt: u64) -> f64 {
        let u1 = self.derive(tick, entity_id, salt).max(1e-300);
        let u2 = self.derive(tick, entity_id, salt.wrapping_add(1));
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// 按实体 id 管理随机流的注册表。
///
/// 用于需要「每个个体一条流」的场景。流在首次访问时惰性创建。
pub struct RngRegistry {
    seed: u64,
    streams: Vec<Option<Pcg64Mcg>>,
}

impl RngRegistry {
    pub fn new(seed: u64, capacity: usize) -> Self {
        Self {
            seed,
            streams: vec![None; capacity],
        }
    }

    /// 取得某个实体的随机流（首次调用时创建）。
    pub fn stream(&mut self, entity_id: usize) -> &mut Pcg64Mcg {
        if entity_id >= self.streams.len() {
            self.streams.resize(entity_id + 1, None);
        }
        self.streams[entity_id].get_or_insert_with(|| {
            // 用 splitmix 从 (seed, entity_id) 派生独立种子
            let mut z = self
                .seed
                .wrapping_add((entity_id as u64).wrapping_mul(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            Pcg64Mcg::seed_from_u64(z ^ (z >> 31))
        })
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

/// 为 n 个个体创建独立的确定性随机流。
pub fn make_streams(seed: u64, n: usize) -> Vec<Pcg64Mcg> {
    (0..n)
        .map(|i| {
            let mut z = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            Pcg64Mcg::seed_from_u64(z ^ (z >> 31))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_is_deterministic() {
        let r = DeriveRng::new(42);
        assert_eq!(r.derive(1, 2, 3), r.derive(1, 2, 3));
    }

    #[test]
    fn derive_in_unit_interval() {
        let r = DeriveRng::new(7);
        for i in 0..1000 {
            let v = r.derive(i, i * 3, 0);
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn derive_order_independent() {
        let r = DeriveRng::new(5);
        let a: Vec<f64> = (0..100).map(|t| r.derive(t, 9, 0)).collect();
        let b: Vec<f64> = (0..100).rev().map(|t| r.derive(t, 9, 0)).collect();
        for (i, v) in b.iter().rev().enumerate() {
            assert_eq!(a[i], *v);
        }
    }

    #[test]
    fn derive_different_inputs_differ() {
        let r = DeriveRng::new(1);
        let mut seen = std::collections::HashSet::new();
        for t in 0..20 {
            for e in 0..20 {
                seen.insert(r.derive(t, e, 0).to_bits());
            }
        }
        assert_eq!(seen.len(), 400);
    }

    #[test]
    fn derive_roughly_uniform() {
        let r = DeriveRng::new(11);
        let sum: f64 = (0..10000).map(|i| r.derive(i, i % 7, 0)).sum();
        let mean = sum / 10000.0;
        assert!((mean - 0.5).abs() < 0.02, "mean = {mean}");
    }

    #[test]
    fn registry_same_entity_same_stream() {
        let mut a = RngRegistry::new(42, 10);
        let mut b = RngRegistry::new(42, 10);
        let va: Vec<f64> = (0..10).map(|_| a.stream(7).gen()).collect();
        let vb: Vec<f64> = (0..10).map(|_| b.stream(7).gen()).collect();
        assert_eq!(va, vb);
    }

    #[test]
    fn registry_different_entities_differ() {
        let mut r = RngRegistry::new(42, 10);
        let a: f64 = r.stream(1).gen();
        let b: f64 = r.stream(2).gen();
        assert_ne!(a, b);
    }

    #[test]
    fn make_streams_deterministic() {
        let a = make_streams(42, 5);
        let b = make_streams(42, 5);
        for (mut x, mut y) in a.into_iter().zip(b) {
            for _ in 0..100 {
                assert_eq!(x.gen::<f64>(), y.gen::<f64>());
            }
        }
    }
}
