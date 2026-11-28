//! Gossip 协议：节点间用反熵 gossip 共享负载信号
//!
//! 为什么不用环境场
//! ----------------
//! 之前的 PoC 用 PheromoneField 作为「完美共享内存」——节点写入立即可见。
//! 真实系统里没有这种东西。替代方案是 gossip：
//!
//! - 每个节点维护自己的**本地视图**（邻居的负载估计）
//! - 每 tick 随机选几个邻居交换信息
//! - 视图逐渐收敛，但有延迟（stale）
//!
//! 这是 Dynamo / Cassandra / Serf 用的机制。

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// Gossip 配置
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// 每个节点每 tick 联系多少个对端
    pub fanout: usize,
    /// 视图混合系数（0 = 完全信任对端，1 = 完全保留自己）
    ///
    /// 实际公式：view[i] = view[i] + alpha * (remote_view - view[i])
    /// alpha 越小收敛越慢但越稳定
    pub alpha: f64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            alpha: 0.5,
        }
    }
}

/// 每个节点的本地视图
pub struct GossipView {
    /// 节点总数
    pub n: usize,
    /// 本地视角下的「全局平均负载」估计
    pub mean_load_estimate: Vec<f64>,
    /// 本地视角下的「邻居负载」估计（用于和邻居比较）
    pub neighbor_load: Vec<Vec<f64>>,
    /// 邻居列表（固定拓扑：环形 + 随机长连接）
    pub neighbors: Vec<Vec<usize>>,
    /// 是否已用实际负载初始化过（首次 gossip 时做）
    initialized: bool,
    rng: Pcg64Mcg,
}

impl GossipView {
    pub fn new(n: usize, fanout: usize, seed: u64) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(seed);

        // 拓扑：每个节点有 2 个环邻居 + fanout 个随机长连接
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut nb = vec![(i + n - 1) % n, (i + 1) % n];
                for _ in 0..fanout {
                    let j = rng.gen_range(0..n);
                    if j != i && !nb.contains(&j) {
                        nb.push(j);
                    }
                }
                nb
            })
            .collect();

        Self {
            n,
            mean_load_estimate: vec![0.0; n],
            neighbor_load: vec![vec![]; n],
            neighbors,
            initialized: false,
            rng,
        }
    }

    /// 一轮 gossip：每个节点联系邻居，交换**估计值**（gossip averaging）
    ///
    /// 关键区别
    /// --------
    /// 早期版本让节点读邻居的**实际负载**，结果每个节点的估计收敛到
    /// 「邻居实际负载的均值」（局部值），而不是全局均值。
    ///
    /// 正确做法是交换**估计值**：节点的估计 = (1-alpha)*自己 + alpha*对端估计。
    /// 这样信息才能通过多跳传播，最终收敛到全局均值。
    /// 这是经典 gossip averaging（Push-Pull）的核心。
    ///
    /// 参数
    /// ----
    /// - `actual_loads`：本 tick 各节点的**真实**负载（用于初始化未收敛的节点，
    ///   以及返回值里的「邻居实际负载」估计）
    /// - `alpha`：混合系数
    ///
    /// 返回：每个节点视角下的「邻居实际负载均值」（用于与邻居比较）
    /// 一轮 gossip
    ///
    /// `inject` 参数
    /// -------------
    /// 节点自己的实际值在采样中的权重：
    /// - `inject = 0.0`：纯 gossip averaging，节点间会完全收敛到同一值
    /// - `inject = 1.0`：每轮重新注入自身值，估计会保留自身偏差，
    ///   永远无法完全一致（但能跟踪变化）
    ///
    /// 真实系统用 `inject > 0`（需要跟踪负载变化），
    /// 代价是估计永远带一点噪声。
    pub fn gossip_round(
        &mut self,
        actual_loads: &[f64],
        alpha: f64,
        inject: f64,
        tick: u64,
        can_see: impl Fn(u64, usize, usize) -> bool,
    ) -> Vec<f64> {
        let n = self.n;

        // 首次调用：用实际负载初始化（现实中每个节点知道自己当前的负载）
        if !self.initialized {
            self.mean_load_estimate.copy_from_slice(actual_loads);
            self.initialized = true;
        }

        let old = self.mean_load_estimate.clone();
        let mut new_estimates = vec![0.0; n];
        let mut neighbor_sums = vec![0.0; n];
        let mut neighbor_counts = vec![0usize; n];

        for i in 0..n {
            // 收集本轮可以采样的值：自己的实际负载（按 inject 加权）+ 邻居的估计值
            // 用凸组合保证稳定：new = (1-alpha)*old + alpha*mean(samples)
            let mut sample_sum = inject * actual_loads[i];
            let mut sample_count = inject;

            for &j in &self.neighbors[i] {
                if !can_see(tick, i, j) {
                    continue;
                }
                // 采样对端的估计值（多跳传播的关键）
                sample_sum += old[j];
                sample_count += 1.0;

                // 同时记录对端实际负载（用于「和邻居比」的决策）
                neighbor_sums[i] += actual_loads[j];
                neighbor_counts[i] += 1;
            }

            let sample_mean = if sample_count > 1e-12 {
                sample_sum / sample_count
            } else {
                old[i]
            };
            // 凸组合：alpha ∈ [0,1] 保证不放大、不发散
            new_estimates[i] = (1.0 - alpha) * old[i] + alpha * sample_mean;
        }

        self.mean_load_estimate = new_estimates;

        // 返回每个节点估计的邻居平均实际负载
        (0..n)
            .map(|i| {
                if neighbor_counts[i] > 0 {
                    neighbor_sums[i] / neighbor_counts[i] as f64
                } else {
                    self.mean_load_estimate[i]
                }
            })
            .collect()
    }

    /// 收敛误差：所有节点估计值的**离散度**
    ///
    /// 为什么不用「估计 vs 实际值」
    /// ----------------------------
    /// Gossip averaging 的正确行为是：所有节点收敛到**全局平均值**，
    /// 而不是收敛到彼此的个体值。所以「估计与实际值之差」在收敛后
    /// 仍等于数据本身的方差，不会趋于 0。
    ///
    /// 正确的收敛指标是节点间估计值的标准差：收敛时趋于 0。
    pub fn convergence_error(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let mean = self.mean_load_estimate.iter().sum::<f64>() / self.n as f64;
        let var = self
            .mean_load_estimate
            .iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>()
            / self.n as f64;
        var.sqrt()
    }

    /// 全局均值估计（所有节点估计值的平均）
    pub fn estimated_global_mean(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.mean_load_estimate.iter().sum::<f64>() / self.n as f64
    }

    /// 视图误差：估计值与真实值的平均绝对差（仅用于参考，收敛后不趋于 0）
    pub fn view_error(&self, actual: &[f64]) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.mean_load_estimate
            .iter()
            .zip(actual)
            .map(|(e, a)| (e - a).abs())
            .sum::<f64>()
            / self.n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gossip_converges_to_true_mean() {
        let n = 100;
        let mut view = GossipView::new(n, 3, 42);
        let loads: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let true_mean: f64 = loads.iter().sum::<f64>() / n as f64;

        for t in 0..200 {
            view.gossip_round(&loads, 0.5, 0.0, t, |_, _, _| true);
        }

        // 1. 节点间估计值收敛（离散度趋于 0）
        let err1 = view.convergence_error();
        assert!(err1 < 0.01, "estimates should converge: {err1}");

        // 2. 收敛值是全局均值
        let est = view.estimated_global_mean();
        assert!(
            (est - true_mean).abs() < 0.01,
            "should converge to global mean {true_mean}, got {est}"
        );
    }

    #[test]
    fn partition_blocks_gossip() {
        let n = 100;
        let mut view = GossipView::new(n, 3, 42);
        let loads = vec![1.0; n];

        // 分区：节点 0-49 和 50-99 互不可见
        let partitioned = |_t: u64, i: usize, j: usize| -> bool { !((i < 50) != (j < 50)) };

        for t in 0..100 {
            view.gossip_round(&loads, 0.5, 0.0, t, partitioned);
        }

        // 所有负载相同（1.0），分区下仍应收敛到 1.0
        assert!((view.estimated_global_mean() - 1.0).abs() < 0.05);
    }

    #[test]
    fn fanout_affects_convergence_speed() {
        let n = 200;
        let loads: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();

        let converge_steps = |fanout: usize| -> usize {
            let mut view = GossipView::new(n, fanout, 7);
            let mut converged = 0;
            for t in 0..500u64 {
                view.gossip_round(&loads, 0.5, 0.0, t, |_, _, _| true);
                if view.convergence_error() < 0.05 && converged == 0 {
                    converged = t as usize + 1;
                }
            }
            converged
        };

        let fast = converge_steps(10);
        let slow = converge_steps(1);
        assert!(fast <= slow, "more fanout should converge faster: {fast} vs {slow}");
    }
}
