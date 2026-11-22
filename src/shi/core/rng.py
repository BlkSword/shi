"""确定性随机数生成器.

设计约束
--------
1. 同种子必须产生完全相同的序列（跨进程、跨机器）。
2. 每个个体拥有独立的随机流，避免并行化破坏确定性。
3. 禁止使用全局 random / np.random 的隐式状态。

两种用法
--------
**有状态流**（make_streams）
    适合需要连续随机序列的场景，例如个体的随机游走。
    每个个体一条独立流，互不干扰。

**无状态派生**（derive）
    适合「按需」随机，例如「这一发该不该暴击」。
    给定 (seed, tick, entity_id, salt) 直接算出结果，不依赖任何内部状态，
    因此可以在任意顺序、任意并行度下调用。
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

__all__ = [
    "make_streams",
    "derive",
    "derive_int",
    "derive_normal",
    "RngRegistry",
]

_UINT64 = 1 << 64
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _check_int64(value, name: str) -> int:
    """校验并归一化为 Python int（必须在 int64 范围内）。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} 必须是整数，收到 {type(value).__name__}")
    v = int(value)
    if not (_INT64_MIN <= v <= _INT64_MAX):
        raise ValueError(f"{name} 超出 int64 范围: {v}")
    return v


def _hash64(seed: int, tick: int, entity_id: int, salt: int) -> int:
    """把四个整数压成一个 64 位无符号整数。

    用 blake2b 而不是 Python 内置 hash()：后者带随机化种子，
    跨进程不稳定，会直接破坏可复现性。
    """
    payload = struct.pack(
        "<qqqq",
        _check_int64(seed, "seed"),
        _check_int64(tick, "tick"),
        _check_int64(entity_id, "entity_id"),
        _check_int64(salt, "salt"),
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def make_streams(seed: int, n: int) -> list[np.random.Generator]:
    """为 n 个个体创建独立的确定性随机流。

    Parameters
    ----------
    seed : int
        主种子。相同种子 + 相同 n 必然产生相同的流集合。
    n : int
        流的数量（通常等于个体数量）。

    Returns
    -------
    list of numpy.random.Generator
        长度为 n 的生成器列表，每个使用独立的 PCG64 子流。
    """
    if n < 0:
        raise ValueError(f"n 必须非负，收到 {n}")
    ss = np.random.SeedSequence(_check_int64(seed, "seed"))
    children = ss.spawn(n)
    return [np.random.Generator(np.random.PCG64(child)) for child in children]


def derive(seed: int, tick: int, entity_id: int, salt: int = 0) -> float:
    """按 (seed, tick, entity_id, salt) 派生一个确定性随机数。

    无状态：任意顺序、任意并行度调用，结果都相同。

    Returns
    -------
    float
        范围 [0, 1) 的浮点数。
    """
    return _hash64(seed, tick, entity_id, salt) / _UINT64


def derive_int(seed: int, tick: int, entity_id: int, n: int, salt: int = 0) -> int:
    """派生一个 [0, n) 的确定性整数。

    注意：这里用取模而不是拒绝采样，会有极微弱的偏差
    （n 远小于 2^64 时可忽略）。如果需要严格均匀，改用拒绝采样。
    """
    if n <= 0:
        raise ValueError(f"n 必须为正，收到 {n}")
    return _hash64(seed, tick, entity_id, salt) % n


def derive_normal(seed: int, tick: int, entity_id: int, salt: int = 0) -> float:
    """派生一个标准正态分布的确定性随机数（Box-Muller 变换）。"""
    u1 = derive(seed, tick, entity_id, salt)
    u2 = derive(seed, tick, entity_id, salt + 1)
    u1 = max(u1, 1e-300)  # 避免 log(0)
    return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))


class RngRegistry:
    """按实体 id 管理随机流的注册表。

    用于需要「每个个体一条流」的场景。流在首次访问时惰性创建，
    因此可以只给真正需要的实体分配流。

    与 make_streams 的区别：make_streams 一次性分配 n 条流，
    适合个体数量固定的场景；RngRegistry 按需分配，适合个体动态增删的场景。
    """

    def __init__(self, seed: int):
        self._seed = _check_int64(seed, "seed")
        self._streams: dict[int, np.random.Generator] = {}

    @property
    def seed(self) -> int:
        return self._seed

    def stream(self, entity_id: int) -> np.random.Generator:
        """取得某个实体的随机流（首次调用时创建）。"""
        eid = _check_int64(entity_id, "entity_id")
        gen = self._streams.get(eid)
        if gen is None:
            ss = np.random.SeedSequence([self._seed, eid])
            gen = np.random.Generator(np.random.PCG64(ss))
            self._streams[eid] = gen
        return gen

    def __len__(self) -> int:
        return len(self._streams)

    def __contains__(self, entity_id: int) -> bool:
        return int(entity_id) in self._streams
