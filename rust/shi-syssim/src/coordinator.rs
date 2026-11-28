//! 两种协调策略
//!
//! - **Decentralized**：节点读邻域信号，自主抢单/释放（stigmergy）
//! - **Centralized**：中心调度器全局最优分配
//!
//! 关键对比：中心失效后的性能下降。

use crate::cluster::Cluster;
use crate::task::Task;

/// 协调器接口
pub trait Coordinator {
    fn name(&self) -> &str;
    /// 执行一个 tick 的调度
    ///
    /// `unassigned` 是当前未分配任务的索引队列（FIFO）。
    /// 协调器从队列头部取任务，取走后从队列移除。
    /// 这是为了避免每次 O(n_tasks) 全表扫描。
    fn step(
        &mut self,
        cluster: &mut Cluster,
        tasks: &mut [Task],
        unassigned: &mut std::collections::VecDeque<usize>,
        tick: u64,
    );
    /// 中心是否失效（去中心化方案永远返回 false）
    fn is_dead(&self) -> bool {
        false
    }
}

// ============================================================
// 去中心化协调
// ============================================================

/// 去中心化协调器：节点只读邻域信号，自主决策
///
/// 规则
/// ----
/// 1. 节点读自己邻域的 load_signal
/// 2. 如果自己的负载低于邻域平均 → 主动接管未分配任务
/// 3. 如果自己的负载高于邻域平均 → 释放一个任务
/// 4. 处理自己持有的任务
///
/// 关键：没有调度器，没有全局视图。
pub struct DecentralizedCoordinator {
    pub sense_radius: usize,
    /// 每个 tick 每个节点最多接管多少任务
    pub max_claim_per_tick: usize,
    /// 释放阈值：负载高于邻域平均多少时释放
    pub release_threshold: f64,
}

impl Default for DecentralizedCoordinator {
    fn default() -> Self {
        Self {
            sense_radius: 3,
            max_claim_per_tick: 2,
            release_threshold: 0.15,
        }
    }
}

impl Coordinator for DecentralizedCoordinator {
    fn name(&self) -> &str {
        "decentralized"
    }

    fn step(
        &mut self,
        cluster: &mut Cluster,
        tasks: &mut [Task],
        unassigned: &mut std::collections::VecDeque<usize>,
        _tick: u64,
    ) {
        // 1. 环境场写入（节点广播自己的状态）
        cluster.write_signals();

        let n_nodes = cluster.nodes.len();

        // 2. 每个节点读邻域信号，决定接管/释放
        for i in 0..n_nodes {
            if cluster.nodes[i].failed {
                continue;
            }

            let neighbor_load = cluster.sense_load(i, self.sense_radius);
            let own_load = cluster.nodes[i].load;

            // 空闲节点：从队列取任务（O(1)，不再全表扫描）
            if own_load < neighbor_load - 1e-9 || neighbor_load < 1e-9 {
                let mut claimed = 0;
                while claimed < self.max_claim_per_tick {
                    match unassigned.pop_front() {
                        Some(task_idx) => {
                            let task = &mut tasks[task_idx];
                            if task.finished || task.owner.is_some() {
                                continue;
                            }
                            task.owner = Some(i);
                            cluster.nodes[i].owned.push(task_idx);
                            claimed += 1;
                        }
                        None => break,
                    }
                }
                if claimed > 0 {
                    cluster.nodes[i].load =
                        (cluster.nodes[i].load + 0.1 * claimed as f64).min(1.0);
                }
            }
            // 过载节点：释放任务
            else if own_load > neighbor_load + self.release_threshold
                && !cluster.nodes[i].owned.is_empty()
            {
                if let Some(task_idx) = cluster.nodes[i].owned.pop() {
                    tasks[task_idx].owner = None;
                    unassigned.push_back(task_idx);
                    cluster.nodes[i].load = (cluster.nodes[i].load - 0.1).max(0.0);
                }
            }
        }

        // 3. 处理任务
        Self::process_tasks(cluster, tasks);

        // 4. 环境场演化
        cluster.step_field();
    }
}

impl DecentralizedCoordinator {
    /// 节点处理自己持有的任务
    fn process_tasks(cluster: &mut Cluster, tasks: &mut [Task]) {
        for i in 0..cluster.nodes.len() {
            if cluster.nodes[i].failed || cluster.nodes[i].owned.is_empty() {
                continue;
            }

            let throughput = cluster.nodes[i].throughput();
            if throughput <= 0.0 {
                continue;
            }

            let n_owned = cluster.nodes[i].owned.len();
            let per_task = throughput / n_owned as f64;

            let owned_idx: Vec<usize> = cluster.nodes[i].owned.clone();
            let mut still_owned = Vec::with_capacity(n_owned);
            let mut processed_total = 0.0;

            for task_idx in owned_idx {
                let done = tasks[task_idx].process(per_task);
                processed_total += done;
                if tasks[task_idx].finished {
                    continue;
                }
                still_owned.push(task_idx);
            }

            cluster.nodes[i].owned = still_owned;
            cluster.nodes[i].processed += processed_total;
            cluster.nodes[i].load = (cluster.nodes[i].owned.len() as f64 * 0.1).min(1.0);
        }
    }
}

// ============================================================
// 中心化协调
// ============================================================

/// 中心化协调器：全局最优分配
///
/// 规则
/// ----
/// 1. 收集所有健康节点的负载和容量
/// 2. 把未分配任务分配给「容量最大且负载最低」的节点
/// 3. 处理任务
///
/// 弱点：中心失效后，系统停止调度。
pub struct CentralizedCoordinator {
    /// 中心是否已失效
    pub dead: bool,
    /// 中心失效的 tick（None = 永不失效）
    pub fail_at: Option<u64>,
    /// 每个 tick 最多分配多少任务
    pub max_assign_per_tick: usize,
}

impl Default for CentralizedCoordinator {
    fn default() -> Self {
        Self {
            dead: false,
            fail_at: None,
            max_assign_per_tick: 100,
        }
    }
}

impl Coordinator for CentralizedCoordinator {
    fn name(&self) -> &str {
        "centralized"
    }

    fn is_dead(&self) -> bool {
        self.dead
    }

    fn step(
        &mut self,
        cluster: &mut Cluster,
        tasks: &mut [Task],
        unassigned: &mut std::collections::VecDeque<usize>,
        tick: u64,
    ) {
        // 检查中心是否失效
        if let Some(fail_at) = self.fail_at {
            if tick >= fail_at {
                self.dead = true;
            }
        }

        // 中心失效：停止调度（但仍然处理已分配任务）
        if self.dead {
            Self::process_tasks(cluster, tasks);
            cluster.step_field();
            return;
        }

        // 1. 收集健康节点（按处理能力降序）
        let mut healthy: Vec<(usize, f64)> = cluster
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.failed)
            .map(|(i, n)| (i, n.throughput()))
            .collect();

        healthy.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if healthy.is_empty() {
            return;
        }

        // 2. 从队列分配任务（轮询 + 容量加权）
        let mut assigned = 0;
        while assigned < self.max_assign_per_tick {
            match unassigned.pop_front() {
                Some(task_idx) => {
                    if tasks[task_idx].finished || tasks[task_idx].owner.is_some() {
                        continue;
                    }
                    let node_idx = healthy[assigned % healthy.len()].0;
                    tasks[task_idx].owner = Some(node_idx);
                    cluster.nodes[node_idx].owned.push(task_idx);
                    assigned += 1;
                }
                None => break,
            }
        }

        // 更新负载
        for (node_idx, _) in &healthy {
            let new_load = (cluster.nodes[*node_idx].owned.len() as f64 * 0.05).min(1.0);
            cluster.nodes[*node_idx].load = new_load;
        }

        // 3. 处理任务
        Self::process_tasks(cluster, tasks);

        // 4. 环境场演化
        cluster.step_field();
    }
}

impl CentralizedCoordinator {
    /// 处理已分配任务（无 leader 时仍需处理手头的任务）
    pub fn process_tasks_public(cluster: &mut Cluster, tasks: &mut [Task]) {
        Self::process_tasks(cluster, tasks)
    }

    fn process_tasks(cluster: &mut Cluster, tasks: &mut [Task]) {
        for i in 0..cluster.nodes.len() {
            if cluster.nodes[i].failed || cluster.nodes[i].owned.is_empty() {
                continue;
            }

            let throughput = cluster.nodes[i].throughput();
            if throughput <= 0.0 {
                continue;
            }

            let n_owned = cluster.nodes[i].owned.len();
            let per_task = throughput / n_owned as f64;

            let owned_idx: Vec<usize> = cluster.nodes[i].owned.clone();
            let mut still_owned = Vec::with_capacity(n_owned);
            let mut processed_total = 0.0;

            for task_idx in owned_idx {
                let done = tasks[task_idx].process(per_task);
                processed_total += done;
                if tasks[task_idx].finished {
                    continue;
                }
                still_owned.push(task_idx);
            }

            cluster.nodes[i].owned = still_owned;
            cluster.nodes[i].processed += processed_total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::ClusterConfig;
    use crate::task::{Task, TaskConfig};

    fn make_setup(n_nodes: usize, n_tasks: usize) -> (Cluster, Vec<Task>) {
        let cluster = Cluster::new(ClusterConfig {
            n_nodes,
            field_width: 50,
            field_height: 50,
            ..Default::default()
        });
        let tasks: Vec<Task> = (0..n_tasks)
            .map(|i| Task::new(i as u64, TaskConfig::default().demand))
            .collect();
        (cluster, tasks)
    }

    fn fresh_queue(tasks: &[Task]) -> std::collections::VecDeque<usize> {
        (0..tasks.len()).collect()
    }

    #[test]
    fn decentralized_assigns_tasks() {
        let (mut cluster, mut tasks) = make_setup(50, 100);
        let mut q = fresh_queue(&tasks);
        let mut coord = DecentralizedCoordinator::default();
        for t in 0..10 {
            coord.step(&mut cluster, &mut tasks, &mut q, t);
        }
        let assigned = tasks.iter().filter(|t| t.owner.is_some()).count();
        assert!(assigned > 0, "should assign some tasks");
    }

    #[test]
    fn centralized_assigns_tasks() {
        let (mut cluster, mut tasks) = make_setup(50, 100);
        let mut q = fresh_queue(&tasks);
        let mut coord = CentralizedCoordinator::default();
        for t in 0..10 {
            coord.step(&mut cluster, &mut tasks, &mut q, t);
        }
        let assigned = tasks.iter().filter(|t| t.owner.is_some()).count();
        assert!(assigned > 0);
    }

    #[test]
    fn centralized_dies_after_fail_at() {
        let (mut cluster, mut tasks) = make_setup(20, 50);
        let mut q = fresh_queue(&tasks);
        let mut coord = CentralizedCoordinator {
            fail_at: Some(5),
            ..Default::default()
        };
        for t in 0..5 {
            coord.step(&mut cluster, &mut tasks, &mut q, t);
            assert!(!coord.is_dead());
        }
        coord.step(&mut cluster, &mut tasks, &mut q, 5);
        assert!(coord.is_dead());
    }

    #[test]
    fn decentralized_never_dies() {
        let (mut cluster, mut tasks) = make_setup(20, 50);
        let mut q = fresh_queue(&tasks);
        let mut coord = DecentralizedCoordinator::default();
        for t in 0..100 {
            coord.step(&mut cluster, &mut tasks, &mut q, t);
        }
        assert!(!coord.is_dead());
    }

    #[test]
    fn tasks_get_completed() {
        let (mut cluster, mut tasks) = make_setup(100, 50);
        let mut q = fresh_queue(&tasks);
        let mut coord = DecentralizedCoordinator::default();
        for t in 0..500 {
            coord.step(&mut cluster, &mut tasks, &mut q, t);
        }
        let finished = tasks.iter().filter(|t| t.finished).count();
        assert!(finished > 0, "some tasks should finish");
    }
}
