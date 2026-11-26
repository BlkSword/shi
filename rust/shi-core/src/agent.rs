//! 阈值个体
//!
//! 核心机制
//! --------
//! P1 阈值响应：个体 i 对任务 j 的响应概率
//!
//! ```text
//! P = s^2 / (s^2 + theta^2)
//! ```
//!
//! 其中 s 是刺激强度，theta 是个体阈值。阈值在群体中呈分布，
//! 分工就自动出现了，不需要任何岗位分配机制。
//!
//! 数据布局
//! --------
//! 用 SoA（Structure of Arrays）而不是 AoS。

use rand::Rng;
use rand_pcg::Pcg64Mcg;

/// 个体状态编码
pub const STATE_IDLE: i8 = 0;
pub const STATE_SEEK: i8 = 1;
pub const STATE_CARRY: i8 = 2;
pub const STATE_RETURN: i8 = 3;

/// Hill 函数形式的阈值响应概率。
///
/// P = s^2 / (s^2 + theta^2)
///
/// 当 s = theta 时 P = 0.5。负刺激钳制为 0。
#[inline]
pub fn response_probability(stimulus: f64, threshold: f64) -> f64 {
    debug_assert!(threshold > 0.0, "threshold must be positive");
    let s = stimulus.max(0.0);
    let s2 = s * s;
    s2 / (s2 + threshold * threshold)
}

/// 从对数正态分布采样个体阈值。
///
/// 用对数正态而不是正态，因为阈值必须为正，且分布天然右偏
/// （少数个体非常敏感，多数个体迟钝）——这与生物学观察一致。
pub fn sample_thresholds(
    n: usize,
    median: f64,
    sigma: f64,
    rng: &mut Pcg64Mcg,
) -> Vec<f64> {
    assert!(median > 0.0, "median must be positive");
    assert!(sigma >= 0.0, "sigma must be non-negative");

    if sigma == 0.0 {
        return vec![median; n];
    }

    let mu = median.ln();
    (0..n)
        .map(|_| {
            // Box-Muller 生成标准正态，再转对数正态
            let u1: f64 = rng.gen::<f64>().max(1e-300);
            let u2: f64 = rng.gen();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (mu + sigma * z).exp()
        })
        .collect()
}

/// 个体的 SoA 状态容器。
#[derive(Debug, Clone)]
pub struct AgentState {
    pub n: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub threshold: Vec<f64>,
    pub state: Vec<i8>,
    pub carrying: Vec<i8>,
    pub energy: Vec<f64>,
}

impl AgentState {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            x: vec![0.0; n],
            y: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            threshold: vec![1.0; n],
            state: vec![STATE_IDLE; n],
            carrying: vec![0; n],
            energy: vec![1.0; n],
        }
    }

    /// 把个体随机撒在区域内。
    pub fn place_random(&mut self, width: f64, height: f64, rng: &mut Pcg64Mcg) {
        for i in 0..self.n {
            self.x[i] = rng.gen::<f64>() * width;
            self.y[i] = rng.gen::<f64>() * height;
        }
    }

    /// 把所有个体放在同一点。
    pub fn place_at(&mut self, x: f64, y: f64) {
        self.x.fill(x);
        self.y.fill(y);
    }

    pub fn set_thresholds(&mut self, thresholds: Vec<f64>) {
        assert_eq!(thresholds.len(), self.n, "threshold length mismatch");
        self.threshold = thresholds;
    }

    /// 把个体位置钳制在区域内（硬边界）。
    pub fn clamp_to(&mut self, width: f64, height: f64) {
        for i in 0..self.n {
            self.x[i] = self.x[i].clamp(0.0, width);
            self.y[i] = self.y[i].clamp(0.0, height);
        }
    }

    /// 环形边界。
    pub fn wrap_to(&mut self, width: f64, height: f64) {
        for i in 0..self.n {
            self.x[i] = self.x[i].rem_euclid(width);
            self.y[i] = self.y[i].rem_euclid(height);
        }
    }

    /// 统计各状态的个体数。
    pub fn count_by_state(&self) -> [usize; 4] {
        let mut counts = [0usize; 4];
        for &s in &self.state {
            if (0..4).contains(&s) {
                counts[s as usize] += 1;
            }
        }
        counts
    }

    /// 阈值分布统计：(mean, std, min, max)
    pub fn threshold_stats(&self) -> (f64, f64, f64, f64) {
        if self.n == 0 {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let mean = self.threshold.iter().sum::<f64>() / self.n as f64;
        let var = self
            .threshold
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>()
            / self.n as f64;
        let min = self.threshold.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self
            .threshold
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (mean, var.sqrt(), min, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn half_at_threshold() {
        assert!((response_probability(2.0, 2.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn increases_with_stimulus() {
        assert!(response_probability(1.0, 2.0) < response_probability(4.0, 2.0));
    }

    #[test]
    fn bounded() {
        for s in [0.0, 0.1, 1.0, 10.0, 1000.0] {
            let p = response_probability(s, 1.0);
            assert!((0.0..1.0).contains(&p));
        }
    }

    #[test]
    fn negative_stimulus_clamped() {
        assert_eq!(response_probability(-1.0, 1.0), 0.0);
    }

    #[test]
    fn sigma_zero_homogeneous() {
        let mut rng = Pcg64Mcg::seed_from_u64(0);
        let t = sample_thresholds(100, 2.0, 0.0, &mut rng);
        assert!(t.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn sigma_positive_heterogeneous() {
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let t = sample_thresholds(1000, 1.0, 0.5, &mut rng);
        let mean = t.iter().sum::<f64>() / 1000.0;
        let var = t.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 1000.0;
        assert!(var.sqrt() > 0.0);
        assert!(t.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn thresholds_reproducible() {
        let mut a = Pcg64Mcg::seed_from_u64(7);
        let mut b = Pcg64Mcg::seed_from_u64(7);
        assert_eq!(
            sample_thresholds(100, 1.0, 0.5, &mut a),
            sample_thresholds(100, 1.0, 0.5, &mut b)
        );
    }

    #[test]
    fn place_random_in_bounds() {
        let mut a = AgentState::new(100);
        let mut rng = Pcg64Mcg::seed_from_u64(0);
        a.place_random(50.0, 30.0, &mut rng);
        assert!(a.x.iter().all(|&v| (0.0..=50.0).contains(&v)));
        assert!(a.y.iter().all(|&v| (0.0..=30.0).contains(&v)));
    }

    #[test]
    fn clamp_works() {
        let mut a = AgentState::new(3);
        a.x = vec![-5.0, 5.0, 100.0];
        a.y = vec![-5.0, 5.0, 100.0];
        a.clamp_to(10.0, 10.0);
        assert_eq!(a.x, vec![0.0, 5.0, 10.0]);
    }

    #[test]
    fn wrap_works() {
        let mut a = AgentState::new(3);
        a.x = vec![-1.0, 5.0, 11.0];
        a.y = vec![0.0, 0.0, 0.0];
        a.wrap_to(10.0, 10.0);
        assert_eq!(a.x, vec![9.0, 5.0, 1.0]);
    }
}
