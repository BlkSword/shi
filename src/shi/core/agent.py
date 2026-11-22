"""阈值个体.

核心机制
--------
P1 阈值响应：个体 i 对任务 j 的响应概率

    P = s^2 / (s^2 + theta^2)

其中 s 是刺激强度，theta 是个体阈值。阈值在群体中呈分布，
分工就自动出现了，不需要任何岗位分配机制。

设计要点（待实现）
------------------
- 数据布局用 SoA（Structure of Arrays），不用 AoS
- 每个个体：位置、阈值、状态、随机流
- 阈值分布可配置（默认 LogNormal）
"""

from __future__ import annotations

__all__ = ["AgentState", "response_probability"]


def response_probability(stimulus: float, threshold: float) -> float:
    """Hill 函数形式的阈值响应概率。"""
    raise NotImplementedError("P0 待实现")


class AgentState:
    """个体的 SoA 状态容器。"""

    def __init__(self, n: int):
        raise NotImplementedError("P0 待实现")
