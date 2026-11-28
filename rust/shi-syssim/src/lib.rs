//! 分布式系统自愈仿真
//!
//! 模型
//! ----
//! - 节点：有健康度、负载、容量
//! - 任务：有需求，需要被处理
//! - 环境场：load_signal（负载）+ health_signal（健康）
//!
//! 两种协调策略
//! ------------
//! - **Decentralized**：节点读邻域信号，自主抢单/释放
//! - **Centralized**：中心调度器全局最优分配
//!
//! 故障注入
//! --------
//! - 随机节点失效
//! - 中心调度器失效
//! - 网络分区

pub mod cluster;
pub mod coordinator;
pub mod experiment;
pub mod gossip;
pub mod ha_control;
pub mod network;
pub mod task;

pub use cluster::{Cluster, ClusterConfig, NodeState};
pub use coordinator::{CentralizedCoordinator, DecentralizedCoordinator, Coordinator};
pub use experiment::{run, ExpConfig};
pub use gossip::{GossipConfig, GossipView};
pub use ha_control::{HaConfig, HaControlPlane, ReplicaState};
pub use network::{Network, NetworkConfig, Partition};
pub use task::{Task, TaskConfig};

/// 仿真结果
#[derive(Debug, Clone, serde::Serialize)]
pub struct SimResult {
    pub strategy: String,
    pub ticks: u64,
    pub completed_ratio: f64,
    pub load_balance: f64,
    pub recovery_ticks: Option<u64>,
    pub post_failure_ratio: f64,
    pub total_tasks: usize,
    pub failed_nodes: usize,
}
