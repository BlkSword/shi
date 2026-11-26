//! 实验运行器：对比去中心化 vs 中心化

use crate::cluster::{Cluster, ClusterConfig};
use crate::coordinator::{CentralizedCoordinator, Coordinator, DecentralizedCoordinator};
use crate::task::{Task, TaskConfig};
use crate::SimResult;

/// 实验配置
#[derive(Debug, Clone)]
pub struct ExpConfig {
    pub n_nodes: usize,
    pub n_tasks: usize,
    pub ticks: u64,
    /// 中心失效的 tick（仅中心化方案用）
    pub central_fail_at: Option<u64>,
    /// 随机节点故障概率（每 tick）
    pub node_fail_prob: f64,
    /// 每 tick 新到达的任务数（持续负载）
    pub arrival_rate: f64,
    pub seed: u64,
}

impl Default for ExpConfig {
    fn default() -> Self {
        Self {
            n_nodes: 1000,
            n_tasks: 2000,
            ticks: 500,
            central_fail_at: None,
            node_fail_prob: 0.0,
            arrival_rate: 0.0,
            seed: 0,
        }
    }
}

/// 跑一次实验
pub fn run(strategy: &str, cfg: &ExpConfig) -> SimResult {
    // 场尺寸随节点数扩展，保持平均密度恒定
    // 为什么重要：若固定 100x100，20000 节点时每格 2 个节点，信号饱和，
    // 节点无法通过邻域信号区分彼此——这会让大规模测试失真。
    let field_side = ((cfg.n_nodes as f64 * 10.0).sqrt() as usize).max(100);

    let cluster_cfg = ClusterConfig {
        n_nodes: cfg.n_nodes,
        field_width: field_side,
        field_height: field_side,
        seed: cfg.seed,
        ..Default::default()
    };
    let mut cluster = Cluster::new(cluster_cfg);
    let task_cfg = TaskConfig {
        n_tasks: cfg.n_tasks,
        demand: 10.0,
    };
    let mut tasks: Vec<Task> = (0..task_cfg.n_tasks)
        .map(|i| Task::new(i as u64, task_cfg.demand))
        .collect();

    let mut completed_history: Vec<f64> = Vec::new();
    let mut failed_nodes = 0;
    let fail_tick = cfg.central_fail_at;

    // 持续任务到达：维护未分配队列
    let mut unassigned: std::collections::VecDeque<usize> = (0..tasks.len()).collect();
    let mut next_task_id = tasks.len() as u64;
    let mut total_arrived = tasks.len();
    let mut total_completed = 0usize;
    let mut arrival_acc = 0.0f64;

    match strategy {
        "decentralized" => {
            let mut coord = DecentralizedCoordinator::default();
            for t in 0..cfg.ticks {
                if cfg.node_fail_prob > 0.0 && t > 0 && t % 50 == 0 {
                    failed_nodes += cluster.inject_random_failure(cfg.node_fail_prob);
                }
                // 持续任务到达
                arrival_acc += cfg.arrival_rate;
                while arrival_acc >= 1.0 {
                    arrival_acc -= 1.0;
                    unassigned.push_back(tasks.len());
                    tasks.push(Task::new(next_task_id, task_cfg.demand));
                    next_task_id += 1;
                    total_arrived += 1;
                }
                coord.step(&mut cluster, &mut tasks, &mut unassigned, t);
                // 统计已完成
                let done = tasks.iter().filter(|x| x.finished).count();
                total_completed = done;
                // 完成率 = 已完成 / 已到达
                completed_history.push(done as f64 / total_arrived as f64);
            }
        }
        "centralized" => {
            let mut coord = CentralizedCoordinator {
                fail_at: fail_tick,
                ..Default::default()
            };
            for t in 0..cfg.ticks {
                if cfg.node_fail_prob > 0.0 && t > 0 && t % 50 == 0 {
                    failed_nodes += cluster.inject_random_failure(cfg.node_fail_prob);
                }
                arrival_acc += cfg.arrival_rate;
                while arrival_acc >= 1.0 {
                    arrival_acc -= 1.0;
                    unassigned.push_back(tasks.len());
                    tasks.push(Task::new(next_task_id, task_cfg.demand));
                    next_task_id += 1;
                    total_arrived += 1;
                }
                coord.step(&mut cluster, &mut tasks, &mut unassigned, t);
                let done = tasks.iter().filter(|x| x.finished).count();
                total_completed = done;
                completed_history.push(done as f64 / total_arrived as f64);
            }
        }
        _ => panic!("unknown strategy: {strategy}"),
    }

    let completed = total_completed;
    let completed_ratio = if total_arrived > 0 {
        completed as f64 / total_arrived as f64
    } else {
        0.0
    };

    // 恢复时间：故障后回到故障前性能水平所需的 tick 数
    let recovery_ticks = fail_tick.and_then(|ft| {
        if (ft as usize) < completed_history.len() {
            let pre_fail = completed_history[ft as usize];
            completed_history
                .iter()
                .enumerate()
                .skip(ft as usize)
                .find(|(_, &v)| v >= pre_fail * 0.9)
                .map(|(i, _)| i as u64 - ft)
        } else {
            None
        }
    });

    // 故障后性能：故障 tick 之后的完成率增量
    let post_failure_ratio = fail_tick
        .map(|ft| {
            let idx = (ft as usize).min(completed_history.len().saturating_sub(1));
            if idx + 1 < completed_history.len() {
                completed_history[completed_history.len() - 1] - completed_history[idx]
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);

    SimResult {
        strategy: strategy.to_string(),
        ticks: cfg.ticks,
        completed_ratio,
        load_balance: cluster.load_balance(),
        recovery_ticks,
        post_failure_ratio,
        total_tasks: total_arrived,
        failed_nodes,
    }
}
