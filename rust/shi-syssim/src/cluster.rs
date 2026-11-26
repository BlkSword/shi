//! 集群模型：节点、环境场、故障注入

use shi_core::{ChannelSpec, PheromoneField};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;

/// 单个节点的状态
#[derive(Debug, Clone)]
pub struct NodeState {
    /// 健康度 [0, 1]，0 = 失效
    pub health: f64,
    /// 当前负载 [0, 1]，相对于自身容量
    pub load: f64,
    /// 容量（异质：不同节点容量不同）
    pub capacity: f64,
    /// 是否失效（health == 0）
    pub failed: bool,
    /// 累计处理量
    pub processed: f64,
    /// 当前持有的任务 id 列表
    pub owned: Vec<usize>,
}

impl NodeState {
    pub fn new(capacity: f64) -> Self {
        Self {
            health: 1.0,
            load: 0.0,
            capacity,
            failed: false,
            processed: 0.0,
            owned: Vec::new(),
        }
    }

    /// 当前实际处理能力（受健康度和负载影响）
    pub fn throughput(&self) -> f64 {
        if self.failed {
            return 0.0;
        }
        // 健康度线性影响，负载过高时饱和
        self.capacity * self.health * (1.0 - self.load.min(1.0) * 0.5)
    }
}

/// 集群配置
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub n_nodes: usize,
    /// 节点容量均值
    pub capacity_mean: f64,
    /// 节点容量变异系数（异质性）
    pub capacity_cv: f64,
    /// 环境场尺寸
    pub field_width: usize,
    pub field_height: usize,
    /// 随机种子
    pub seed: u64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            n_nodes: 1000,
            capacity_mean: 1.0,
            capacity_cv: 0.3,
            field_width: 100,
            field_height: 100,
            seed: 0,
        }
    }
}

/// 集群
pub struct Cluster {
    pub config: ClusterConfig,
    pub nodes: Vec<NodeState>,
    /// 环境场：通道 0 = load_signal, 通道 1 = health_signal
    pub field: PheromoneField,
    /// 每个节点在环境场上的位置（固定）
    pub node_x: Vec<f64>,
    pub node_y: Vec<f64>,
    pub rng: Pcg64Mcg,
    pub tick: u64,
}

impl Cluster {
    pub fn new(config: ClusterConfig) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);

        // 节点容量：对数正态分布（异质）
        let nodes: Vec<NodeState> = (0..config.n_nodes)
            .map(|_| {
                let u1: f64 = rng.gen::<f64>().max(1e-300);
                let u2: f64 = rng.gen();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let sigma = (1.0 + config.capacity_cv * config.capacity_cv)
                    .ln()
                    .sqrt();
                let cap = config.capacity_mean * (sigma * z).exp();
                NodeState::new(cap)
            })
            .collect();

        // 节点位置：均匀撒在环境场上
        let mut rng2 = Pcg64Mcg::seed_from_u64(config.seed.wrapping_add(1));
        let node_x: Vec<f64> = (0..config.n_nodes)
            .map(|_| rng2.gen::<f64>() * config.field_width as f64)
            .collect();
        let node_y: Vec<f64> = (0..config.n_nodes)
            .map(|_| rng2.gen::<f64>() * config.field_height as f64)
            .collect();

        // 环境场：两个通道，衰减率不同
        let field = PheromoneField::new(
            config.field_width,
            config.field_height,
            vec![
                ChannelSpec::new(0.90, 0.10), // load_signal：快衰减 + 扩散
                ChannelSpec::new(0.95, 0.05), // health_signal：慢衰减
            ],
        );

        Self {
            config,
            nodes,
            field,
            node_x,
            node_y,
            rng,
            tick: 0,
        }
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn healthy_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.failed).count()
    }

    /// 写入环境场：每个健康节点写自己的负载和健康度
    pub fn write_signals(&mut self) {
        for i in 0..self.nodes.len() {
            if self.nodes[i].failed {
                continue;
            }
            // load_signal: 负载越高写入越强
            self.field
                .deposit(self.node_x[i], self.node_y[i], 0, self.nodes[i].load);
            // health_signal: 健康度
            self.field
                .deposit(self.node_x[i], self.node_y[i], 1, self.nodes[i].health);
        }
    }

    /// 读邻域负载信号（用于去中心化决策）
    pub fn sense_load(&self, node_idx: usize, radius: usize) -> f64 {
        self.field.sense(
            self.node_x[node_idx],
            self.node_y[node_idx],
            0,
            radius,
            true,
        )
    }

    /// 读邻域健康信号
    pub fn sense_health(&self, node_idx: usize, radius: usize) -> f64 {
        self.field.sense(
            self.node_x[node_idx],
            self.node_y[node_idx],
            1,
            radius,
            true,
        )
    }

    /// 环境场推进一个 tick
    pub fn step_field(&mut self) {
        self.field.step();
    }

    /// 注入随机节点故障
    pub fn inject_random_failure(&mut self, probability: f64) -> usize {
        let mut count = 0;
        for node in &mut self.nodes {
            if !node.failed && self.rng.gen::<f64>() < probability {
                node.failed = true;
                node.health = 0.0;
                node.load = 0.0;
                node.owned.clear();
                count += 1;
            }
        }
        count
    }

    /// 让指定比例的节点失效（从随机位置选）
    pub fn fail_fraction(&mut self, fraction: f64) -> usize {
        let n_fail = ((self.nodes.len() as f64) * fraction) as usize;
        let mut indices: Vec<usize> = (0..self.nodes.len()).collect();
        // Fisher-Yates 洗牌
        for i in (1..indices.len()).rev() {
            let j = self.rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        for &idx in indices.iter().take(n_fail) {
            self.nodes[idx].failed = true;
            self.nodes[idx].health = 0.0;
            self.nodes[idx].load = 0.0;
            self.nodes[idx].owned.clear();
        }
        n_fail
    }

    /// 负载均衡度（基尼系数，0 = 完全均衡，1 = 极端不均）
    ///
    /// 用排序法计算：G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n
    /// 复杂度 O(n log n)，而不是朴素的 O(n^2)。
    pub fn load_balance(&self) -> f64 {
        let mut loads: Vec<f64> = self
            .nodes
            .iter()
            .filter(|n| !n.failed)
            .map(|n| n.load)
            .collect();
        let n = loads.len();
        if n < 2 {
            return 0.0;
        }

        loads.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sum: f64 = loads.iter().sum();
        if sum <= 1e-12 {
            return 0.0;
        }

        // 1-based 排名
        let weighted: f64 = loads
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as f64 + 1.0) * x)
            .sum();

        let nf = n as f64;
        ((2.0 * weighted) / (nf * sum) - (nf + 1.0) / nf).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_creates_nodes() {
        let c = Cluster::new(ClusterConfig {
            n_nodes: 100,
            ..Default::default()
        });
        assert_eq!(c.n_nodes(), 100);
        assert_eq!(c.healthy_count(), 100);
    }

    #[test]
    fn capacities_are_heterogeneous() {
        let c = Cluster::new(ClusterConfig {
            n_nodes: 1000,
            capacity_cv: 0.3,
            ..Default::default()
        });
        let caps: Vec<f64> = c.nodes.iter().map(|n| n.capacity).collect();
        let mean = caps.iter().sum::<f64>() / caps.len() as f64;
        let var = caps.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / caps.len() as f64;
        assert!(var.sqrt() > 0.0);
        assert!(caps.iter().all(|&c| c > 0.0));
    }

    #[test]
    fn failure_reduces_healthy_count() {
        let mut c = Cluster::new(ClusterConfig {
            n_nodes: 100,
            ..Default::default()
        });
        let n = c.fail_fraction(0.2);
        assert_eq!(n, 20);
        assert_eq!(c.healthy_count(), 80);
    }

    #[test]
    fn failed_node_has_zero_throughput() {
        let mut node = NodeState::new(1.0);
        assert!(node.throughput() > 0.0);
        node.failed = true;
        node.health = 0.0;
        assert_eq!(node.throughput(), 0.0);
    }

    #[test]
    fn load_balance_zero_when_uniform() {
        let mut c = Cluster::new(ClusterConfig {
            n_nodes: 10,
            ..Default::default()
        });
        for n in &mut c.nodes {
            n.load = 0.5;
        }
        assert!(c.load_balance() < 1e-9);
    }

    #[test]
    fn load_balance_positive_when_uneven() {
        let mut c = Cluster::new(ClusterConfig {
            n_nodes: 10,
            ..Default::default()
        });
        for (i, n) in c.nodes.iter_mut().enumerate() {
            n.load = if i < 5 { 0.0 } else { 1.0 };
        }
        assert!(c.load_balance() > 0.4);
    }

    #[test]
    fn signals_deposited() {
        let mut c = Cluster::new(ClusterConfig {
            n_nodes: 10,
            ..Default::default()
        });
        for n in &mut c.nodes {
            n.load = 0.5;
        }
        c.write_signals();
        assert!(c.field.total(0) > 0.0);
        assert!(c.field.total(1) > 0.0);
    }
}
