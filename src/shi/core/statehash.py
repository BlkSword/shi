"""状态哈希：可复现性的证据.

用途
----
同一个种子跑两次，状态哈希序列必须完全一致。
这是 P0 阶段的硬性验收标准，也是整个研究的地基。

设计要点
--------
- 用 blake2b（digest_size=16）：速度快，碰撞概率可忽略。
- 对 np.ascontiguousarray(x).tobytes() 做哈希，保证内存布局一致。
- 浮点数按位哈希：-0.0 与 0.0 不同，NaN 的不同位模式也不同。
  这是刻意的——我们要的是「逐位一致」，不是「数值相等」。
- 每个数组前写入 dtype、ndim、shape 和 size，防止不同形状/类型产生相同哈希。
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable

import numpy as np

__all__ = ["state_hash", "hash_arrays", "hash_bytes"]

_DIGEST_SIZE = 16


def hash_bytes(*chunks: bytes, tick: int = 0) -> str:
    """把若干字节块 + tick 号哈希成一个十六进制字符串。

    Parameters
    ----------
    *chunks : bytes
        任意数量的字节块。
    tick : int
        当前 tick 号，参与哈希。

    Returns
    -------
    str
        32 个十六进制字符。
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    h.update(struct.pack("<q", int(tick)))
    for chunk in chunks:
        h.update(struct.pack("<Q", len(chunk)))
        h.update(chunk)
    return h.hexdigest()


def hash_arrays(arrays: Iterable[np.ndarray], tick: int = 0) -> str:
    """把若干数组按位哈希。

    每个数组的 dtype、维度数、各维长度、元素个数都会参与哈希，
    因此形状不同或类型不同的数组不会产生相同哈希。
    """
    chunks: list[bytes] = []
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        chunks.append(a.dtype.str.encode("ascii"))
        # ndim + 各维长度：保证 (12,) 与 (3,4) 不会碰撞
        chunks.append(struct.pack("<Q", a.ndim))
        chunks.append(struct.pack(f"<{a.ndim}Q", *a.shape))
        chunks.append(struct.pack("<Q", a.size))
        chunks.append(a.tobytes())
    return hash_bytes(*chunks, tick=tick)


def state_hash(positions, env, tick: int) -> str:
    """计算当前世界状态的哈希。

    Parameters
    ----------
    positions : array-like
        个体位置，形状 (n, 2) 或 (n, d)。
    env : array-like
        环境场，任意形状。
    tick : int
        当前 tick 号。

    Returns
    -------
    str
        32 个十六进制字符。
    """
    return hash_arrays([np.asarray(positions), np.asarray(env)], tick=tick)
