"""主循环：固定 tick + 确定性推进.

设计约束
--------
- 固定时间步长（默认 20 Hz 逻辑 tick），不用真实时间
- 每 tick 的执行顺序必须固定：感知 -> 决策 -> 动作 -> 环境更新
- 每 N tick 计算一次状态哈希，用于可复现性验证
"""

from __future__ import annotations

__all__ = ["Engine"]


class Engine:
    """仿真主循环。"""

    def __init__(self, world, agents, seed: int, tick_rate: int = 20):
        raise NotImplementedError("P0 待实现")

    def step(self) -> None:
        raise NotImplementedError("P0 待实现")

    def run(self, n_ticks: int) -> None:
        raise NotImplementedError("P0 待实现")
