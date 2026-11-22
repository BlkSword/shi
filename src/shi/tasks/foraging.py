"""觅食任务：找到并搬运资源回巢.

用于 P1（环境可写性）和 P2（异质性）的基准任务之一。
"""

from __future__ import annotations

__all__ = ["ForagingTask"]


class ForagingTask:
    """觅食任务。"""

    def __init__(self, world, n_resources: int = 50, n_nests: int = 1):
        raise NotImplementedError("P0 待实现")
