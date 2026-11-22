"""分配任务：把 N 个任务分配给 M 个个体.

用于 P1 的基准任务之一。
"""

from __future__ import annotations

__all__ = ["AllocationTask"]


class AllocationTask:
    """任务分配。"""

    def __init__(self, world, n_tasks: int = 20):
        raise NotImplementedError("P0 待实现")
