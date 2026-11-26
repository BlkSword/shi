//! 主循环：固定 tick + 确定性推进
//!
//! 设计约束
//! --------
//! - 固定时间步长（逻辑 tick），不用真实时间
//! - 每 tick 的执行顺序必须固定：个体行为 -> 环境演化 -> tick 计数
//! - 每 N tick 计算一次状态哈希，用于可复现性验证

use crate::agent::AgentState;
use crate::rng::DeriveRng;
use crate::statehash::state_hash;
use crate::world::PheromoneField;

/// 个体行为更新函数。
///
/// 必须原地修改 agents 和 world，不返回任何东西。
pub trait Stepper {
    fn step(&mut self, agents: &mut AgentState, world: &mut PheromoneField, tick: u64);
}

/// 闭包形式的 Stepper。
impl<F> Stepper for F
where
    F: FnMut(&mut AgentState, &mut PheromoneField, u64),
{
    fn step(&mut self, agents: &mut AgentState, world: &mut PheromoneField, tick: u64) {
        self(agents, world, tick)
    }
}

/// 仿真主循环。
pub struct Engine<S: Stepper> {
    pub world: PheromoneField,
    pub agents: AgentState,
    pub seed: u64,
    pub stepper: S,
    pub tick: u64,
    pub hash_every: u64,
    pub hashes: Vec<(u64, u64)>,
    pub rng: DeriveRng,
}

impl<S: Stepper> Engine<S> {
    pub fn new(
        world: PheromoneField,
        agents: AgentState,
        seed: u64,
        stepper: S,
        hash_every: u64,
    ) -> Self {
        Self {
            world,
            agents,
            seed,
            stepper,
            tick: 0,
            hash_every,
            hashes: Vec::new(),
            rng: DeriveRng::new(seed),
        }
    }

    /// 推进一个 tick。
    pub fn step(&mut self) {
        self.stepper.step(&mut self.agents, &mut self.world, self.tick);
        self.world.step();
        self.tick += 1;

        if self.hash_every > 0 && self.tick % self.hash_every == 0 {
            let h = self.current_hash();
            self.hashes.push((self.tick, h));
        }
    }

    /// 连续推进 n 个 tick。
    pub fn run(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// 当前世界状态的哈希。
    pub fn current_hash(&self) -> u64 {
        // 把 (x, y) 交错成位置数组
        let mut positions = Vec::with_capacity(self.agents.n * 2);
        for i in 0..self.agents.n {
            positions.push(self.agents.x[i]);
            positions.push(self.agents.y[i]);
        }
        state_hash(&positions, &self.world.data, self.tick)
    }

    /// 已记录的所有哈希值。
    pub fn hash_sequence(&self) -> Vec<u64> {
        self.hashes.iter().map(|(_, h)| *h).collect()
    }

    /// 重置 tick 计数和记录（不重置世界状态）。
    pub fn reset(&mut self) {
        self.tick = 0;
        self.hashes.clear();
        self.rng = DeriveRng::new(self.seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::ChannelSpec;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn make_engine(seed: u64, hash_every: u64) -> Engine<impl FnMut(&mut AgentState, &mut PheromoneField, u64)> {
        let world = PheromoneField::new(30, 30, vec![ChannelSpec::new(0.98, 0.0)]);
        let mut agents = AgentState::new(50);
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        agents.place_random(30.0, 30.0, &mut rng);

        let stepper = move |ag: &mut AgentState, w: &mut PheromoneField, _tick: u64| {
            for i in 0..ag.n {
                ag.x[i] += 0.1;
                ag.y[i] += 0.05;
                ag.x[i] = ag.x[i].clamp(0.0, 29.0);
                ag.y[i] = ag.y[i].clamp(0.0, 29.0);
                w.deposit(ag.x[i], ag.y[i], 0, 0.1);
            }
        };

        Engine::new(world, agents, seed, stepper, hash_every)
    }

    #[test]
    fn tick_increments() {
        let mut e = make_engine(42, 10);
        e.step();
        assert_eq!(e.tick, 1);
    }

    #[test]
    fn hash_recorded_at_interval() {
        let mut e = make_engine(42, 10);
        e.run(50);
        assert_eq!(e.hashes.len(), 5);
        assert_eq!(
            e.hashes.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            vec![10, 20, 30, 40, 50]
        );
    }

    #[test]
    fn same_seed_same_hash_sequence() {
        let mut a = make_engine(123, 10);
        let mut b = make_engine(123, 10);
        a.run(200);
        b.run(200);
        assert_eq!(a.hash_sequence(), b.hash_sequence());
    }

    #[test]
    fn different_seed_different_hash() {
        let mut a = make_engine(1, 10);
        let mut b = make_engine(2, 10);
        a.run(100);
        b.run(100);
        assert_ne!(a.hash_sequence(), b.hash_sequence());
    }

    #[test]
    fn repeated_runs_stable() {
        let baseline = {
            let mut e = make_engine(7, 10);
            e.run(100);
            e.hash_sequence()
        };
        for _ in 0..10 {
            let mut e = make_engine(7, 10);
            e.run(100);
            assert_eq!(e.hash_sequence(), baseline);
        }
    }
}
