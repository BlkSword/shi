//! 网络延迟与分区模型
//!
//! 为什么需要这个
//! --------------
//! 之前的 PoC 假设「完美共享内存」：节点写入环境场后立即可见。
//! 真实分布式系统里：
//! - 写操作有网络延迟（跨机架 1ms，跨区域 50ms）
//! - 节点看到的其他节点状态是**过时**的（stale read）
//! - 网络分区时，一部分节点完全看不到另一部分
//!
//! 这个模块模拟这三件事。

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;

/// 网络配置
#[derive(Debug, Clone, Serialize)]
pub struct NetworkConfig {
    /// 消息延迟（tick 数）。0 = 完美共享内存
    pub latency_ticks: u64,
    /// 延迟抖动（±jitter tick）
    pub jitter: u64,
    /// 消息丢失概率 [0, 1)
    pub drop_prob: f64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            latency_ticks: 0,
            jitter: 0,
            drop_prob: 0.0,
        }
    }
}

impl NetworkConfig {
    /// 局域网（低延迟）
    pub fn lan() -> Self {
        Self {
            latency_ticks: 1,
            jitter: 0,
            drop_prob: 0.001,
        }
    }

    /// 跨区域（高延迟）
    pub fn wan() -> Self {
        Self {
            latency_ticks: 5,
            jitter: 2,
            drop_prob: 0.01,
        }
    }

    /// 恶劣网络
    pub fn hostile() -> Self {
        Self {
            latency_ticks: 10,
            jitter: 5,
            drop_prob: 0.05,
        }
    }
}

/// 一条待投递的消息
#[derive(Debug, Clone)]
struct PendingWrite {
    /// 投递时间（tick）
    deliver_at: u64,
    /// 写入位置
    x: f64,
    y: f64,
    channel: usize,
    amount: f64,
}

/// 网络分区
#[derive(Debug, Clone, Serialize)]
pub struct Partition {
    /// 分区开始 tick
    pub start: u64,
    /// 分区结束 tick（None = 永不恢复）
    pub end: Option<u64>,
    /// 分区大小（按节点位置划分：x < split_x 为 A 侧，否则 B 侧）
    pub split_x: f64,
}

impl Partition {
    pub fn is_active(&self, tick: u64) -> bool {
        tick >= self.start && self.end.map_or(true, |e| tick < e)
    }

    /// 两个位置是否被分区隔开
    pub fn separates(&self, x1: f64, x2: f64) -> bool {
        (x1 < self.split_x) != (x2 < self.split_x)
    }
}

/// 模拟网络：延迟投递 + 丢包 + 分区
pub struct Network {
    config: NetworkConfig,
    pending: Vec<PendingWrite>,
    partitions: Vec<Partition>,
    rng: Pcg64Mcg,
    /// 统计
    pub sent: u64,
    pub delivered: u64,
    pub dropped: u64,
}

impl Network {
    pub fn new(config: NetworkConfig, seed: u64) -> Self {
        Self {
            config,
            pending: Vec::new(),
            partitions: Vec::new(),
            rng: Pcg64Mcg::seed_from_u64(seed),
            sent: 0,
            delivered: 0,
            dropped: 0,
        }
    }

    pub fn add_partition(&mut self, p: Partition) {
        self.partitions.push(p);
    }

    /// 是否有活跃分区
    pub fn has_active_partition(&self, tick: u64) -> bool {
        self.partitions.iter().any(|p| p.is_active(tick))
    }

    /// 发送一条写入（可能延迟、可能丢失）
    pub fn send(
        &mut self,
        tick: u64,
        x: f64,
        y: f64,
        channel: usize,
        amount: f64,
    ) -> bool {
        self.sent += 1;

        // 丢包
        if self.config.drop_prob > 0.0 && self.rng.gen::<f64>() < self.config.drop_prob {
            self.dropped += 1;
            return false;
        }

        // 延迟
        let jitter = if self.config.jitter > 0 {
            self.rng.gen_range(0..=self.config.jitter)
        } else {
            0
        };
        let deliver_at = tick + self.config.latency_ticks + jitter;

        self.pending.push(PendingWrite {
            deliver_at,
            x,
            y,
            channel,
            amount,
        });
        true
    }

    /// 取出当前 tick 应该投递的消息
    pub fn take_deliverable(&mut self, tick: u64) -> Vec<(f64, f64, usize, f64)> {
        let mut ready = Vec::new();
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].deliver_at <= tick {
                let w = self.pending.swap_remove(i);
                ready.push((w.x, w.y, w.channel, w.amount));
                self.delivered += 1;
            } else {
                i += 1;
            }
        }
        ready
    }

    /// 某个位置能否看到另一个位置（分区隔离）
    pub fn can_see(&self, tick: u64, x_reader: f64, x_writer: f64) -> bool {
        for p in &self.partitions {
            if p.is_active(tick) && p.separates(x_reader, x_writer) {
                return false;
            }
        }
        true
    }

    /// 网络统计
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.sent, self.delivered, self.dropped)
    }

    pub fn effective_drop_rate(&self) -> f64 {
        if self.sent == 0 {
            0.0
        } else {
            self.dropped as f64 / self.sent as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_latency_delivers_immediately() {
        let mut net = Network::new(NetworkConfig::default(), 0);
        net.send(0, 1.0, 2.0, 0, 5.0);
        let ready = net.take_deliverable(0);
        assert_eq!(ready.len(), 1);
    }

    #[test]
    fn latency_delays_delivery() {
        let mut net = Network::new(NetworkConfig { latency_ticks: 3, jitter: 0, drop_prob: 0.0 }, 0);
        net.send(0, 1.0, 2.0, 0, 5.0);
        assert_eq!(net.take_deliverable(0).len(), 0);
        assert_eq!(net.take_deliverable(2).len(), 0);
        assert_eq!(net.take_deliverable(3).len(), 1);
    }

    #[test]
    fn packet_loss() {
        let mut net = Network::new(NetworkConfig { latency_ticks: 0, jitter: 0, drop_prob: 1.0 }, 0);
        assert!(!net.send(0, 1.0, 2.0, 0, 5.0));
        assert_eq!(net.effective_drop_rate(), 1.0);
    }

    #[test]
    fn partition_separates() {
        let mut net = Network::new(NetworkConfig::default(), 0);
        net.add_partition(Partition { start: 10, end: Some(20), split_x: 50.0 });
        // 分区未开始
        assert!(net.can_see(5, 10.0, 90.0));
        // 分区期间
        assert!(!net.can_see(15, 10.0, 90.0));
        assert!(net.can_see(15, 10.0, 20.0));  // 同侧可见
        // 分区结束
        assert!(net.can_see(25, 10.0, 90.0));
    }

    #[test]
    fn permanent_partition() {
        let mut net = Network::new(NetworkConfig::default(), 0);
        net.add_partition(Partition { start: 0, end: None, split_x: 50.0 });
        assert!(!net.can_see(1000000, 10.0, 90.0));
    }
}
