//! 多智能体共享记忆：协调通道污染在 AI 系统里的对应物
//!
//! 为什么这个模型能映射到 LLM 多智能体
//! ------------------------------------
//! 当前主流的多智能体 LLM 架构里，agent 通过**共享记忆/消息板**协作：
//! - 一个 agent 完成子任务，往共享区写「我做完了」
//! - 其他 agent 读到后不再重复
//!
//! 这就是 stigmergy —— 和蜂群的共享标记场结构完全相同。
//!
//! 于是蜂群的结论可以迁移：
//! - 「假标记」= 被劫持的 agent 往共享记忆写假完成
//! - 「淹没」= 大量无关内容冲淡有效信息
//! - 「信源认证」= 条目溯源（谁写的、可验证吗）
//! - 「交叉验证」= 需要多个独立来源确认
//!
//! 模型
//! ----
//! - M 个任务，每个需要被某个 agent 真正执行一次
//! - N 个 agent，每个 tick 挑一个任务执行，完成后写记忆
//! - 共享记忆记录「哪些任务已完成」
//! - 攻击者是被劫持的 agent，可以写假完成记录

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use serde::Serialize;

/// 一条共享记忆条目
#[derive(Debug, Clone)]
pub struct Entry {
    /// 任务 id
    pub task_id: usize,
    /// 作者（agent id）
    pub author: usize,
    /// 写入 tick
    pub tick: u64,
    /// 是否经由可验证的信源（溯源认证）
    pub authenticated: bool,
}

/// 配置
#[derive(Debug, Clone, Serialize)]
pub struct MultiMemConfig {
    pub n_agents: usize,
    pub n_tasks: usize,
    /// 每个任务需要的执行量
    pub work_per_task: f64,
    /// 每个 agent 每 tick 的工作量
    pub work_rate: f64,
    /// 被劫持的 agent 数量
    pub n_compromised: usize,
    /// 被劫持 agent 每 tick 写假记录的概率
    pub attack_rate: f64,
    /// 防御 1：是否启用溯源认证（假记录被拒）
    pub defense_provenance: bool,
    /// 防御 2：交叉验证阈值（需要几个不同作者确认才信任；0 = 不启用）
    pub defense_cross_validate: usize,
    pub seed: u64,
}

impl Default for MultiMemConfig {
    fn default() -> Self {
        Self {
            n_agents: 20,
            n_tasks: 40,
            work_per_task: 1.0,
            work_rate: 0.1,
            n_compromised: 0,
            attack_rate: 0.0,
            defense_provenance: false,
            defense_cross_validate: 0,
            seed: 0,
        }
    }
}

/// 统计
#[derive(Debug, Clone, Default, Serialize)]
pub struct MultiMemStats {
    /// 真实完成的任务数
    pub tasks_done: usize,
    /// 总任务数
    pub n_tasks: usize,
    /// 假记录数
    pub fake_entries: usize,
    /// 被假记录骗过、跳过的任务次数
    pub misled_skips: usize,
    /// 被拒绝的假记录（溯源生效）
    pub rejected: usize,
}

/// 多智能体共享记忆仿真
pub struct MultiMem {
    pub config: MultiMemConfig,
    /// 每个任务的真实完成量
    pub work: Vec<f64>,
    /// 共享记忆
    pub memory: Vec<Entry>,
    /// 哪些 agent 被劫持
    pub compromised: Vec<bool>,
    pub tick: u64,
    pub stats: MultiMemStats,
    rng: Pcg64Mcg,
}

impl MultiMem {
    pub fn new(config: MultiMemConfig) -> Self {
        let mut rng = Pcg64Mcg::seed_from_u64(config.seed);
        let n = config.n_agents;

        // 随机选被劫持的 agent
        let mut compromised = vec![false; n];
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        for &idx in indices.iter().take(config.n_compromised.min(n)) {
            compromised[idx] = true;
        }

        Self {
            config: config.clone(),
            work: vec![0.0; config.n_tasks],
            memory: Vec::new(),
            compromised,
            tick: 0,
            stats: MultiMemStats {
                n_tasks: config.n_tasks,
                ..Default::default()
            },
            rng,
        }
    }

    /// 某个任务是否被「可信地」标记为完成
    fn trust_completion(&self, task_id: usize) -> bool {
        let entries: Vec<&Entry> = self.memory.iter().filter(|e| e.task_id == task_id).collect();
        if entries.is_empty() {
            return false;
        }

        // 交叉验证：需要 K 个不同作者
        if self.config.defense_cross_validate > 0 {
            let mut authors = std::collections::HashSet::new();
            for e in &entries {
                if self.config.defense_provenance && !e.authenticated {
                    continue;
                }
                authors.insert(e.author);
            }
            return authors.len() >= self.config.defense_cross_validate;
        }

        // 默认：任意一条可信记录就算完成
        entries
            .iter()
            .any(|e| !self.config.defense_provenance || e.authenticated)
    }

    /// 推进一个 tick
    pub fn step(&mut self) {
        let n_agents = self.config.n_agents;
        let n_tasks = self.config.n_tasks;

        // ---- 1. 被劫持的 agent 写假记录 ----
        for a in 0..n_agents {
            if !self.compromised[a] {
                continue;
            }
            if self.rng.gen::<f64>() >= self.config.attack_rate {
                continue;
            }
            // 随机选一个还没真正完成的任务，谎称完成
            let undone: Vec<usize> = (0..n_tasks)
                .filter(|&t| self.work[t] < self.config.work_per_task)
                .collect();
            if undone.is_empty() {
                continue;
            }
            let target = undone[self.rng.gen_range(0..undone.len())];

            // 溯源认证：假记录被拒
            let authenticated = !self.config.defense_provenance || !self.compromised[a];
            if !authenticated {
                self.stats.rejected += 1;
                continue;
            }

            self.memory.push(Entry {
                task_id: target,
                author: a,
                tick: self.tick,
                authenticated,
            });
            self.stats.fake_entries += 1;
        }

        // ---- 2. 正常 agent 挑任务执行 ----
        for a in 0..n_agents {
            if self.compromised[a] {
                continue; // 被劫持的 agent 不干活
            }

            // 找一个「看起来没完成」的任务
            let mut chosen: Option<usize> = None;
            let mut misled = false;
            for _ in 0..10 {
                let t = self.rng.gen_range(0..n_tasks);
                if self.work[t] >= self.config.work_per_task {
                    continue; // 真的完成了
                }
                if self.trust_completion(t) {
                    // 被记忆骗了：以为完成，实际没完成
                    misled = true;
                    continue;
                }
                chosen = Some(t);
                break;
            }

            // 如果所有尝试都被记忆骗过，记一次
            if chosen.is_none() && misled {
                self.stats.misled_skips += 1;
            }

            if let Some(t) = chosen {
                // 真的干活
                self.work[t] += self.config.work_rate;
                if self.work[t] >= self.config.work_per_task {
                    self.work[t] = self.config.work_per_task;
                    self.memory.push(Entry {
                        task_id: t,
                        author: a,
                        tick: self.tick,
                        authenticated: true,
                    });
                }
            }
        }

        // ---- 3. 统计 ----
        self.stats.tasks_done = self
            .work
            .iter()
            .filter(|&&w| w >= self.config.work_per_task)
            .count();

        self.tick += 1;
    }

    pub fn completion(&self) -> f64 {
        if self.config.n_tasks == 0 {
            return 0.0;
        }
        self.stats.tasks_done as f64 / self.config.n_tasks as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(cfg: MultiMemConfig, ticks: u64) -> MultiMem {
        let mut m = MultiMem::new(cfg);
        for _ in 0..ticks {
            m.step();
        }
        m
    }

    #[test]
    fn no_attack_completes_all() {
        let m = run(
            MultiMemConfig {
                n_agents: 20,
                n_tasks: 40,
                work_rate: 0.2,
                ..Default::default()
            },
            2000,
        );
        assert!(m.completion() > 0.95, "completion = {}", m.completion());
    }

    #[test]
    fn attack_reduces_completion() {
        let clean = run(
            MultiMemConfig {
                n_agents: 20,
                n_tasks: 40,
                work_rate: 0.2,
                ..Default::default()
            },
            2000,
        );
        let attacked = run(
            MultiMemConfig {
                n_agents: 20,
                n_tasks: 40,
                work_rate: 0.2,
                n_compromised: 3,
                attack_rate: 1.0,
                ..Default::default()
            },
            2000,
        );
        assert!(
            attacked.completion() < clean.completion() - 0.1,
            "attack should hurt: clean={:.2} attacked={:.2}",
            clean.completion(),
            attacked.completion()
        );
        assert!(attacked.stats.fake_entries > 0);
    }

    #[test]
    fn provenance_defense_blocks_fake_entries() {
        let m = run(
            MultiMemConfig {
                n_agents: 20,
                n_tasks: 40,
                work_rate: 0.2,
                n_compromised: 3,
                attack_rate: 1.0,
                defense_provenance: true,
                ..Default::default()
            },
            2000,
        );
        assert_eq!(m.stats.fake_entries, 0, "all fakes should be rejected");
        assert!(m.stats.rejected > 0);
        assert!(m.completion() > 0.9);
    }

    #[test]
    fn cross_validation_also_works() {
        let m = run(
            MultiMemConfig {
                n_agents: 20,
                n_tasks: 40,
                work_rate: 0.2,
                n_compromised: 3,
                attack_rate: 1.0,
                defense_cross_validate: 2,
                ..Default::default()
            },
            2000,
        );
        // 交叉验证需要 2 个不同作者，单个攻击者无法伪造
        assert!(
            m.completion() > 0.5,
            "cross-validation should help: {:.2}",
            m.completion()
        );
    }

    #[test]
    fn determinism() {
        let build = || {
            let m = run(
                MultiMemConfig {
                    n_agents: 15,
                    n_tasks: 30,
                    n_compromised: 2,
                    attack_rate: 0.5,
                    seed: 42,
                    ..Default::default()
                },
                500,
            );
            (m.completion(), m.stats.fake_entries)
        };
        assert_eq!(build(), build());
    }
}
