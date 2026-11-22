"""阈值个体.

核心机制
--------
P1 阈值响应：个体 i 对任务 j 的响应概率

    P = s^2 / (s^2 + theta^2)

其中 s 是刺激强度，theta 是个体阈值。阈值在群体中呈分布，
分工就自动出现了，不需要任何岗位分配机制。

为什么用 Hill 函数
------------------
- 连续、单调、有饱和：刺激很小时几乎不响应，刺激很大时几乎必然响应
- 参数 theta 有明确含义：它是「响应概率达到 0.5 时的刺激强度」
- 阈值分布的形状直接决定分工结构

数据布局
--------
用 SoA（Structure of Arrays）而不是 AoS：
    positions_x : float64[n]
    positions_y : float64[n]
    thresholds  : float64[n]
    states      : int8[n]

理由：SoA 能被 NumPy 高效向量化，也能被 Numba 直接编译。
AoS（对象列表）在 n 大时会慢一到两个数量级。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "response_probability",
    "AgentState",
    "sample_thresholds",
    "STATE_IDLE",
    "STATE_SEEK",
    "STATE_CARRY",
    "STATE_RETURN",
]

# 个体状态编码（int8，便于紧凑存储）
STATE_IDLE = 0
STATE_SEEK = 1
STATE_CARRY = 2
STATE_RETURN = 3

_STATE_NAMES = {
    STATE_IDLE: "idle",
    STATE_SEEK: "seek",
    STATE_CARRY: "carry",
    STATE_RETURN: "return",
}


def response_probability(stimulus, threshold):
    """Hill 函数形式的阈值响应概率。

    P = s^2 / (s^2 + theta^2)

    支持标量和数组。当 s = theta 时 P = 0.5。

    Parameters
    ----------
    stimulus : float or ndarray
        刺激强度，必须非负。
    threshold : float or ndarray
        个体阈值，必须为正。

    Returns
    -------
    float or ndarray
        响应概率，范围 [0, 1)。
    """
    s = np.asarray(stimulus, dtype=np.float64)
    theta = np.asarray(threshold, dtype=np.float64)

    if np.any(theta <= 0):
        raise ValueError("threshold 必须为正")

    # 负刺激钳制到 0：环境场可能因数值误差出现极小负值，
    # 抛错会让长跑实验崩溃。物理上负浓度无意义，钳制是正确语义。
    s = np.maximum(s, 0.0)

    s2 = s * s
    return s2 / (s2 + theta * theta)


def sample_thresholds(
    n: int,
    median: float = 1.0,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    n_types: int = 1,
) -> np.ndarray:
    """从对数正态分布采样个体阈值。

    用对数正态而不是正态，因为阈值必须为正，且分布天然右偏
    （少数个体非常敏感，多数个体迟钝）——这与生物学观察一致。

    Parameters
    ----------
    n : int
        个体数量。
    median : float
        阈值的中位数。sigma=0 时所有个体阈值都等于它。
    sigma : float
        对数标准差，即「异质性」参数。sigma=0 表示完全同质。
    rng : numpy.random.Generator | None
        随机源。为 None 时用固定种子（保证可复现）。

    Returns
    -------
    ndarray
        形状 (n,) 的阈值数组。
    """
    if n < 0:
        raise ValueError(f"n 必须非负，收到 {n}")
    if median <= 0:
        raise ValueError(f"median 必须为正，收到 {median}")
    if sigma < 0:
        raise ValueError(f"sigma 必须非负，收到 {sigma}")

    if rng is None:
        rng = np.random.default_rng(0)

    shape = (n,) if n_types == 1 else (n, n_types)

    if sigma == 0.0:
        return np.full(shape, float(median), dtype=np.float64)

    mu = np.log(median)
    return rng.lognormal(mean=mu, sigma=sigma, size=shape).astype(np.float64)


class AgentState:
    """个体的 SoA 状态容器。

    所有数组长度相同，索引 i 对应同一个体。
    """

    __slots__ = (
        "n",
        "x",
        "y",
        "vx",
        "vy",
        "threshold",
        "state",
        "carrying",
        "energy",
    )

    def __init__(self, n: int):
        if n < 0:
            raise ValueError(f"n 必须非负，收到 {n}")
        self.n = int(n)
        self.x = np.zeros(n, dtype=np.float64)
        self.y = np.zeros(n, dtype=np.float64)
        self.vx = np.zeros(n, dtype=np.float64)
        self.vy = np.zeros(n, dtype=np.float64)
        self.threshold = np.ones(n, dtype=np.float64)
        self.state = np.full(n, STATE_IDLE, dtype=np.int8)
        self.carrying = np.zeros(n, dtype=np.int8)  # 0/1，是否携带资源
        self.energy = np.ones(n, dtype=np.float64)

    # ---------- 初始化 ----------

    def place_random(self, width: float, height: float, rng: np.random.Generator) -> None:
        """把个体随机撒在区域内。"""
        self.x[:] = rng.random(self.n) * width
        self.y[:] = rng.random(self.n) * height

    def place_at(self, x: float, y: float, jitter: float = 0.0, rng=None) -> None:
        """把所有个体放在同一点（可加抖动）。"""
        self.x[:] = x
        self.y[:] = y
        if jitter > 0:
            if rng is None:
                rng = np.random.default_rng(0)
            self.x += rng.normal(0.0, jitter, self.n)
            self.y += rng.normal(0.0, jitter, self.n)

    def set_thresholds(self, thresholds: np.ndarray) -> None:
        """设置阈值。

        支持两种形状：
        - (n,)    标量阈值（单任务）
        - (n, K)  阈值向量（多任务，每个个体对每类任务一个阈值）
        """
        arr = np.asarray(thresholds, dtype=np.float64)
        if arr.shape == (self.n,):
            self.threshold[:] = arr
        elif arr.shape[0] == self.n:
            # 向量阈值：重新分配存储
            self.threshold = arr.copy()
        else:
            raise ValueError(
                f"阈值数组形状应为 ({self.n},) 或 ({self.n}, K)，收到 {arr.shape}"
            )

    # ---------- 查询 ----------

    @property
    def positions(self) -> np.ndarray:
        """返回 (n, 2) 的位置数组（副本）。"""
        return np.stack([self.x, self.y], axis=1)

    def count_by_state(self) -> dict[str, int]:
        """统计各状态的个体数（用于指标采集）。"""
        counts: dict[str, int] = {}
        for code, name in _STATE_NAMES.items():
            counts[name] = int(np.count_nonzero(self.state == code))
        return counts

    def threshold_stats(self) -> dict[str, float]:
        """阈值分布的统计量（用于验证异质性设置）。"""
        return {
            "mean": float(self.threshold.mean()),
            "std": float(self.threshold.std()),
            "min": float(self.threshold.min()),
            "max": float(self.threshold.max()),
        }

    # ---------- 边界处理 ----------

    def clamp_to(self, width: float, height: float) -> None:
        """把个体位置钳制在区域内（硬边界）。"""
        np.clip(self.x, 0.0, width, out=self.x)
        np.clip(self.y, 0.0, height, out=self.y)

    def wrap_to(self, width: float, height: float) -> None:
        """环形边界：越界后从另一侧出现。"""
        np.mod(self.x, width, out=self.x)
        np.mod(self.y, height, out=self.y)

    def __repr__(self) -> str:
        return f"AgentState(n={self.n}, states={self.count_by_state()})"
