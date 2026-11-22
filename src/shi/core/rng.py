"""确定性随机数生成器.

设计约束
--------
1. 同种子必须产生完全相同的序列（跨进程、跨机器）。
2. 每个个体拥有独立的随机流，避免并行化破坏确定性。
3. 禁止使用全局 random / np.random 的隐式状态。

实现要点（待实现）
------------------
- 用 numpy.random.SeedSequence(seed).spawn(n) 为每个个体派生独立子流
- 每个子流用 PCG64 生成器
- 提供 derive(seed, tick, entity_id) 用于「按需派生」的确定性随机
"""

from __future__ import annotations

__all__ = ["make_streams", "derive"]


def make_streams(seed: int, n: int):
    """为 n 个个体创建独立的确定性随机流。"""
    raise NotImplementedError("P0 待实现")


def derive(seed: int, tick: int, entity_id: int) -> float:
    """按 (seed, tick, entity_id) 派生一个确定性随机数，范围 [0, 1)。"""
    raise NotImplementedError("P0 待实现")
