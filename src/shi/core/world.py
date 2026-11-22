"""多通道环境场.

形式化定义
----------
环境 E = (C, W, D)

- C: 通道集合，例如 ["threat", "supply", "order", "corpse"]
- W: 写入算子，个体在所在格叠加浓度
- D: 衰减/扩散算子，每 tick 按通道衰减并可选扩散

设计要点（待实现）
------------------
- 稀疏哈希网格 + 稠密数组混合：小规模用稠密（快），大规模用稀疏（省内存）
- 所有运算用 float64，固定求和顺序，保证确定性
- 提供 sense(entity, channel, radius) 返回邻域加权梯度
"""

from __future__ import annotations

__all__ = ["PheromoneField"]


class PheromoneField:
    """多通道信息素场。"""

    def __init__(self, width: int, height: int, channels: list[str], decay: float = 0.98):
        raise NotImplementedError("P0 待实现")

    def deposit(self, x: float, y: float, channel: str, amount: float) -> None:
        raise NotImplementedError("P0 待实现")

    def sense(self, x: float, y: float, channel: str, radius: int) -> float:
        raise NotImplementedError("P0 待实现")

    def step(self) -> None:
        """推进一个 tick：衰减 + 扩散。"""
        raise NotImplementedError("P0 待实现")
