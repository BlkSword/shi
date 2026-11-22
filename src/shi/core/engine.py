"""主循环：固定 tick + 确定性推进.

设计约束
--------
- 固定时间步长（逻辑 tick），不用真实时间
- 每 tick 的执行顺序必须固定：感知 -> 决策 -> 动作 -> 环境更新
- 每 N tick 计算一次状态哈希，用于可复现性验证

为什么顺序必须固定
------------------
如果「环境更新」和「个体动作」的顺序不确定，同样的输入会产生不同结果。
这是并行化最容易引入的 bug，也是可复现性最大的敌人。
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from shi.core.agent import AgentState
from shi.core.metrics import MetricsRecorder
from shi.core.statehash import state_hash
from shi.core.world import PheromoneField

__all__ = ["Engine", "Stepper"]


class Stepper(Protocol):
    """个体行为更新函数。

    签名：(agents, world, tick, rng) -> None
    必须原地修改 agents 和 world，不返回任何东西。
    """

    def __call__(
        self,
        agents: AgentState,
        world: PheromoneField,
        tick: int,
        rng: np.random.Generator,
    ) -> None: ...


class Engine:
    """仿真主循环。

    Parameters
    ----------
    world : PheromoneField
        环境场。
    agents : AgentState
        个体集合。
    seed : int
        主种子。所有随机性都从它派生。
    stepper : Stepper
        个体行为更新函数。
    tick_rate : int
        逻辑 tick 频率（仅用于记录，不影响仿真逻辑）。
    hash_every : int
        每多少个 tick 计算一次状态哈希。0 表示不计算。
    """

    def __init__(
        self,
        world: PheromoneField,
        agents: AgentState,
        seed: int,
        stepper: Stepper,
        tick_rate: int = 20,
        hash_every: int = 100,
    ):
        self.world = world
        self.agents = agents
        self.seed = int(seed)
        self.stepper = stepper
        self.tick_rate = int(tick_rate)
        self.hash_every = int(hash_every)

        self.tick = 0
        self.metrics = MetricsRecorder()
        self.hashes: list[tuple[int, str]] = []

        # 主随机流：用于全局决策（例如环境扰动）
        self._rng = np.random.default_rng(np.random.SeedSequence(self.seed))

    # ---------- 推进 ----------

    def step(self) -> None:
        """推进一个 tick。

        固定顺序：
        1. 个体行为（感知 -> 决策 -> 动作）
        2. 环境演化（衰减 + 扩散）
        3. tick 计数
        4. 可选：状态哈希
        """
        self.stepper(self.agents, self.world, self.tick, self._rng)
        self.world.step()
        self.tick += 1

        if self.hash_every > 0 and self.tick % self.hash_every == 0:
            self.hashes.append((self.tick, self.current_hash()))

    def run(self, n_ticks: int, record_every: int = 0, **record_kwargs) -> None:
        """连续推进 n_ticks 个 tick。

        Parameters
        ----------
        n_ticks : int
            推进的 tick 数。
        record_every : int
            每多少个 tick 记录一次指标。0 表示不记录。
        **record_kwargs
            额外的指标列（固定值，例如实验配置）。
        """
        for _ in range(n_ticks):
            self.step()
            if record_every > 0 and self.tick % record_every == 0:
                self.metrics.record(tick=self.tick, **record_kwargs)

    # ---------- 状态 ----------

    def current_hash(self) -> str:
        """当前世界状态的哈希。"""
        return state_hash(self.agents.positions, self.world.data, self.tick)

    def hash_sequence(self) -> list[str]:
        """已记录的所有哈希值（按 tick 顺序）。"""
        return [h for _, h in self.hashes]

    def reset(self) -> None:
        """重置 tick 计数和记录（不重置世界状态）。"""
        self.tick = 0
        self.metrics.clear()
        self.hashes.clear()
        self._rng = np.random.default_rng(np.random.SeedSequence(self.seed))

    def __repr__(self) -> str:
        return (
            f"Engine(tick={self.tick}, n_agents={self.agents.n}, "
            f"hashes={len(self.hashes)})"
        )
