"""状态哈希：可复现性的证据.

用途
----
同一个种子跑两次，状态哈希序列必须完全一致。
这是 P0 阶段的硬性验收标准，也是整个研究的地基。

实现要点（待实现）
------------------
- 用 hashlib.blake2b(digest_size=16)
- 对 np.ascontiguousarray(x).tobytes() 做哈希
- 哈希内容：所有个体位置 + 环境场 + tick 号
"""

from __future__ import annotations

__all__ = ["state_hash"]


def state_hash(positions, env, tick: int) -> str:
    """计算当前世界状态的哈希。"""
    raise NotImplementedError("P0 待实现")
