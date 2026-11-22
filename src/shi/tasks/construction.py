"""建造任务：按指定形状堆叠方块.

用于 P1 的基准任务之一。对应 TERMES 的 stigmergy 建造。
"""

from __future__ import annotations

__all__ = ["ConstructionTask"]


class ConstructionTask:
    """集体建造。"""

    def __init__(self, world, target_shape):
        raise NotImplementedError("P0 待实现")
