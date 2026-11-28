//! 高可用 control plane：多副本 + leader 选举 + 故障转移
//!
//! 为什么需要这个
//! --------------
//! 之前的「中心化」对照是单点 control plane，死掉就完全停止。
//! 这不符合真实系统的水平——Kubernetes 有多个 control plane 副本，
//! 用 etcd 的 Raft 选主，leader 挂了会在秒级内重新选主。
//!
//! 这个模块实现一个简化版：
//! - N 个 control plane 副本
//! - 任意时刻只有一个 leader 在工作
//! - leader 心跳失败后，剩余副本超时重新选举
//! - 选举期间调度暂停（这是真实系统的代价）

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;

/// Control plane 副本状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ReplicaState {
    /// 跟随者（不调度，但保持热备）
    Follower,
    /// 候选者（正在竞选）
    Candidate,
    /// 领导者（负责调度）
    Leader,
    /// 已失效
    Failed,
}

/// 高可用配置
#[derive(Debug, Clone, Serialize)]
pub struct HaConfig {
    /// 副本数（建议奇数：3 或 5）
    pub n_replicas: usize,
    /// 选举超时（tick）。真实系统里是 150-300ms
    pub election_timeout: u64,
    /// 心跳间隔（tick）
    pub heartbeat_interval: u64,
}

impl Default for HaConfig {
    fn default() -> Self {
        Self {
            n_replicas: 3,
            election_timeout: 10,
            heartbeat_interval: 3,
        }
    }
}

impl HaConfig {
    /// 单副本（等价于之前的单点对照）
    pub fn single() -> Self {
        Self {
            n_replicas: 1,
            election_timeout: 10,
            heartbeat_interval: 3,
        }
    }

    /// 三副本（K8s 默认）
    pub fn three() -> Self {
        Self::default()
    }

    /// 五副本（高要求场景）
    pub fn five() -> Self {
        Self {
            n_replicas: 5,
            ..Self::default()
        }
    }
}

/// 高可用 control plane 集群
pub struct HaControlPlane {
    pub config: HaConfig,
    pub states: Vec<ReplicaState>,
    /// 每个副本最后一次收到 leader 心跳的 tick
    last_heartbeat: Vec<u64>,
    /// 每个副本的选举计时器（距离超时还有多少 tick）
    election_timer: Vec<u64>,
    rng: Pcg64Mcg,
    /// 统计
    pub leader_changes: u64,
    /// 当前是否有 leader
    pub has_leader: bool,
}

impl HaControlPlane {
    pub fn new(config: HaConfig, seed: u64) -> Self {
        let n = config.n_replicas;
        let election_timeout = config.election_timeout;
        let has_leader = n > 0;

        let mut states = vec![ReplicaState::Follower; n];
        if n > 0 {
            states[0] = ReplicaState::Leader; // 初始 leader
        }
        Self {
            last_heartbeat: vec![0; n],
            election_timer: vec![election_timeout; n],
            states,
            config,
            rng: Pcg64Mcg::seed_from_u64(seed),
            leader_changes: 0,
            has_leader,
        }
    }

    /// 当前 leader 的索引
    pub fn leader(&self) -> Option<usize> {
        self.states
            .iter()
            .position(|&s| s == ReplicaState::Leader)
    }

    /// 杀掉指定副本（模拟故障）
    pub fn fail_replica(&mut self, idx: usize) {
        if idx < self.states.len() {
            self.states[idx] = ReplicaState::Failed;
            if self.leader() == Some(idx) {
                self.has_leader = false;
            }
        }
    }

    /// 推进一个 tick
    ///
    /// 返回：本 tick 是否有可用的 leader（能调度）
    pub fn tick(&mut self, tick: u64) -> bool {
        let n = self.config.n_replicas;
        if n == 0 {
            return false;
        }

        // 1. 存活副本数
        let alive: Vec<usize> = (0..n)
            .filter(|&i| self.states[i] != ReplicaState::Failed)
            .collect();

        if alive.is_empty() {
            self.has_leader = false;
            return false;
        }

        // 2. 如果有 leader，发心跳
        if let Some(leader_idx) = self.leader() {
            if tick % self.config.heartbeat_interval == 0 {
                for &i in &alive {
                    if i != leader_idx {
                        self.last_heartbeat[i] = tick;
                        self.election_timer[i] = self.config.election_timeout;
                    }
                }
            }
            self.has_leader = true;
            return true;
        }

        // 3. 没有 leader：选举超时后重新选
        self.has_leader = false;

        // 找出超时的候选者
        let mut candidates = Vec::new();
        for &i in &alive {
            if self.states[i] == ReplicaState::Failed {
                continue;
            }
            let since_hb = tick.saturating_sub(self.last_heartbeat[i]);
            if since_hb >= self.config.election_timeout {
                candidates.push(i);
            }
        }

        if candidates.is_empty() {
            return false;
        }

        // 4. 投票：随机选一个候选者（简化版 Raft，真实系统按 term/log 比较）
        let winner_pos = self.rng.gen_range(0..candidates.len());
        let winner = candidates[winner_pos];

        // 所有存活副本改为 Follower，winner 改为 Leader
        for &i in &alive {
            if i != winner {
                self.states[i] = ReplicaState::Follower;
                self.election_timer[i] = self.config.election_timeout;
            }
        }
        self.states[winner] = ReplicaState::Leader;
        self.leader_changes += 1;
        self.has_leader = true;

        // 通知心跳
        for &i in &alive {
            if i != winner {
                self.last_heartbeat[i] = tick;
            }
        }

        true
    }

    /// 副本状态快照
    pub fn states(&self) -> &[ReplicaState] {
        &self.states
    }

    /// 存活副本数
    pub fn alive_count(&self) -> usize {
        self.states
            .iter()
            .filter(|&&s| s != ReplicaState::Failed)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_replica_leads_immediately() {
        let mut ha = HaControlPlane::new(HaConfig::single(), 0);
        assert!(ha.tick(0));
        assert_eq!(ha.leader(), Some(0));
    }

    #[test]
    fn single_replica_dies() {
        let mut ha = HaControlPlane::new(HaConfig::single(), 0);
        ha.tick(0);
        ha.fail_replica(0);
        assert!(!ha.tick(1));
        assert_eq!(ha.leader(), None);
    }

    #[test]
    fn three_replicas_survive_one_failure() {
        let mut ha = HaControlPlane::new(HaConfig::three(), 0);
        ha.tick(0);
        let leader = ha.leader().unwrap();
        assert_eq!(leader, 0);

        // 杀掉 leader
        ha.fail_replica(leader);

        // 等待选举超时
        let mut recovered = false;
        for t in 1..50 {
            if ha.tick(t) && ha.leader().is_some() {
                recovered = true;
                break;
            }
        }
        assert!(recovered, "should elect new leader after timeout");
        assert_ne!(ha.leader(), Some(leader));
        assert!(ha.leader_changes > 0);
    }

    #[test]
    fn three_replicas_survive_two_failures() {
        let mut ha = HaControlPlane::new(HaConfig::three(), 0);
        ha.tick(0);
        ha.fail_replica(0);
        ha.fail_replica(1);

        // 只剩一个副本，仍能工作
        let mut recovered = false;
        for t in 1..50 {
            if ha.tick(t) && ha.leader().is_some() {
                recovered = true;
                break;
            }
        }
        assert!(recovered);
        assert_eq!(ha.leader(), Some(2));
    }

    #[test]
    fn three_replicas_all_dead() {
        let mut ha = HaControlPlane::new(HaConfig::three(), 0);
        ha.tick(0);
        ha.fail_replica(0);
        ha.fail_replica(1);
        ha.fail_replica(2);
        assert!(!ha.tick(10));
        assert_eq!(ha.leader(), None);
        assert_eq!(ha.alive_count(), 0);
    }

    #[test]
    fn five_replicas_more_fault_tolerant() {
        let mut ha = HaControlPlane::new(HaConfig::five(), 0);
        ha.tick(0);
        // 杀 2 个副本
        ha.fail_replica(0);
        ha.fail_replica(1);

        let mut recovered = false;
        for t in 1..50 {
            if ha.tick(t) && ha.leader().is_some() {
                recovered = true;
                break;
            }
        }
        assert!(recovered);
        assert!(ha.alive_count() >= 3);
    }
}
