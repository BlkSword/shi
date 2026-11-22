"""参数扫描：多进程并行跑多组配置.

设计要点（待实现）
------------------
- 用 multiprocessing.Pool 而不是线程（Python 有 GIL）
- 进程隔离天然保证确定性
- 支持网格搜索和随机搜索
"""

from __future__ import annotations

__all__ = ["grid_sweep"]


def grid_sweep(param_grid: dict, n_repeats: int = 30, n_workers: int = 8):
    """网格扫描。"""
    raise NotImplementedError("P0 待实现")
