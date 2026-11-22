"""多通道环境场.

形式化定义
----------
环境 E = (C, W, D)

- C: 通道集合，例如 ["threat", "supply", "order"]
- W: 写入算子，个体在所在格叠加浓度
- D: 衰减/扩散算子，每 tick 按通道衰减并可选扩散

为什么用环境而不是消息
----------------------
个体之间不直接通信，只读写环境。这使得通信成本从 O(n^2) 降到 O(n)，
是系统能扩展到大规模的唯一可行方案（stigmergy）。

确定性保证
----------
- 全部用 float64
- 衰减/扩散用固定顺序的逐元素运算，不用可能改变归约顺序的 BLAS
- 不依赖任何全局状态
"""

from __future__ import annotations

import numpy as np

__all__ = ["PheromoneField", "ChannelSpec"]

# 默认通道配置：名称 -> 衰减率
DEFAULT_CHANNELS: dict[str, float] = {
    "threat": 0.90,   # 威胁：衰减快，反映即时危险
    "supply": 0.98,   # 补给：衰减慢，反映资源分布
    "order": 0.995,   # 指令：衰减最慢，反映长期目标
}


class ChannelSpec:
    """单个通道的配置。"""

    __slots__ = ("name", "decay", "diffusion")

    def __init__(self, name: str, decay: float = 0.98, diffusion: float = 0.0):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay 必须在 (0, 1] 内，收到 {decay}")
        if not 0.0 <= diffusion <= 0.25:
            raise ValueError(f"diffusion 必须在 [0, 0.25] 内，收到 {diffusion}")
        self.name = name
        self.decay = float(decay)
        self.diffusion = float(diffusion)

    def __repr__(self) -> str:
        return f"ChannelSpec({self.name!r}, decay={self.decay}, diffusion={self.diffusion})"


class PheromoneField:
    """多通道信息素场。

    数据布局：(n_channels, height, width) 的 float64 数组。
    通道维度在最前面，便于按通道做整体衰减。

    Parameters
    ----------
    width, height : int
        网格尺寸。
    channels : dict[str, float] | list[ChannelSpec] | None
        通道配置。dict 形式为 {名称: 衰减率}；None 时用 DEFAULT_CHANNELS。
    diffusion : float
        全局扩散系数，默认 0（不扩散）。
    """

    def __init__(
        self,
        width: int,
        height: int,
        channels: dict[str, float] | list[ChannelSpec] | None = None,
        diffusion: float = 0.0,
    ):
        if width <= 0 or height <= 0:
            raise ValueError(f"网格尺寸必须为正，收到 {width}x{height}")

        if channels is None:
            specs = [ChannelSpec(name, decay) for name, decay in DEFAULT_CHANNELS.items()]
        elif isinstance(channels, dict):
            specs = [ChannelSpec(name, decay, diffusion) for name, decay in channels.items()]
        else:
            specs = list(channels)

        if not specs:
            raise ValueError("至少需要一个通道")

        self.width = int(width)
        self.height = int(height)
        self.specs = specs
        self.channel_names = [s.name for s in specs]
        self._index = {name: i for i, name in enumerate(self.channel_names)}

        # (C, H, W) float64
        self.data = np.zeros((len(specs), self.height, self.width), dtype=np.float64)

        # 预计算衰减向量，避免每 tick 重复构造
        self._decay = np.array([s.decay for s in specs], dtype=np.float64).reshape(-1, 1, 1)
        self._diffusion = np.array([s.diffusion for s in specs], dtype=np.float64).reshape(-1, 1, 1)

    # ---------- 基本属性 ----------

    @property
    def n_channels(self) -> int:
        return len(self.specs)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def channel_index(self, channel: str) -> int:
        try:
            return self._index[channel]
        except KeyError:
            raise KeyError(
                f"未知通道 {channel!r}，可用通道: {self.channel_names}"
            ) from None

    # ---------- 坐标处理 ----------

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        """把连续坐标映射到格子索引，越界时钳制到边界。"""
        cx = int(np.floor(x))
        cy = int(np.floor(y))
        if cx < 0:
            cx = 0
        elif cx >= self.width:
            cx = self.width - 1
        if cy < 0:
            cy = 0
        elif cy >= self.height:
            cy = self.height - 1
        return cy, cx

    # ---------- 写入 ----------

    def deposit(self, x: float, y: float, channel: str, amount: float) -> None:
        """在 (x, y) 所在格叠加浓度。

        负的 amount 表示「擦除」（例如个体消耗了资源）。
        """
        ci = self.channel_index(channel)
        cy, cx = self._cell(x, y)
        self.data[ci, cy, cx] += float(amount)

    def deposit_many(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        channel: str,
        amounts: np.ndarray,
    ) -> None:
        """批量写入（向量化版本）。

        注意：同一格被多次写入时，np.add.at 保证累加正确
        （普通花式索引赋值会丢失重复项）。
        """
        ci = self.channel_index(channel)
        cxs = np.clip(np.floor(xs).astype(np.int64), 0, self.width - 1)
        cys = np.clip(np.floor(ys).astype(np.int64), 0, self.height - 1)
        np.add.at(self.data[ci], (cys, cxs), amounts)

    # ---------- 读取 ----------

    def read(self, x: float, y: float, channel: str) -> float:
        """读取 (x, y) 所在格的浓度。"""
        ci = self.channel_index(channel)
        cy, cx = self._cell(x, y)
        return float(self.data[ci, cy, cx])

    def sense(
        self,
        x: float,
        y: float,
        channel: str,
        radius: int = 1,
        normalize: bool = False,
    ) -> float:
        """读取以 (x, y) 为中心、半径 radius 的方形邻域内的浓度。

        Parameters
        ----------
        normalize : bool
            False（默认）返回邻域总和，体现「有多少人在做这件事」。
            True 返回邻域平均值，使刺激强度与单格浓度同量级。

            这一点很关键：如果阈值是「单格浓度」量级，而刺激是
            「邻域总和」，两者会差 radius^2 倍，导致响应概率饱和，
            阈值分布完全失去作用。做阈值实验时必须用 normalize=True。
        """
        if radius < 0:
            raise ValueError(f"radius 必须非负，收到 {radius}")
        ci = self.channel_index(channel)
        cy, cx = self._cell(x, y)
        y0 = max(0, cy - radius)
        y1 = min(self.height, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(self.width, cx + radius + 1)
        block = self.data[ci, y0:y1, x0:x1]
        if normalize:
            return float(block.mean())
        return float(block.sum())

    def gradient(self, x: float, y: float, channel: str) -> tuple[float, float]:
        """用中心差分估计 (x, y) 处的梯度。

        返回 (dx, dy)。个体可以沿梯度爬升（趋化性），
        这是「跟随信息素」的最简实现。
        """
        ci = self.channel_index(channel)
        cy, cx = self._cell(x, y)
        xm = max(0, cx - 1)
        xp = min(self.width - 1, cx + 1)
        ym = max(0, cy - 1)
        yp = min(self.height - 1, cy + 1)

        gx = self.data[ci, cy, xp] - self.data[ci, cy, xm]
        gy = self.data[ci, yp, cx] - self.data[ci, ym, cx]
        return float(gx), float(gy)

    def total(self, channel: str) -> float:
        """通道总浓度（用于指标采集）。"""
        ci = self.channel_index(channel)
        return float(self.data[ci].sum())

    # ---------- 演化 ----------

    def step(self) -> None:
        """推进一个 tick：先扩散，再衰减。

        顺序很重要：先扩散再衰减，等价于「扩散后的量一起衰减」，
        与「先衰减再扩散」在数学上不同。固定这个顺序以保证确定性。
        """
        # 扩散（仅对有 diffusion > 0 的通道）
        if np.any(self._diffusion > 0):
            self._diffuse()

        # 衰减
        self.data *= self._decay

    def _diffuse(self) -> None:
        """五点差分扩散。

        用显式欧拉格式，稳定性要求 diffusion <= 0.25。
        ChannelSpec 已做校验。
        """
        src = self.data
        # 四邻域平均（边界用自身填充，等价于零通量边界）
        padded = np.pad(src, ((0, 0), (1, 1), (1, 1)), mode="edge")
        neighbor_sum = (
            padded[:, :-2, 1:-1]   # 上
            + padded[:, 2:, 1:-1]  # 下
            + padded[:, 1:-1, :-2] # 左
            + padded[:, 1:-1, 2:]  # 右
        )
        laplacian = neighbor_sum - 4.0 * src
        self.data = src + self._diffusion * laplacian

    # ---------- 静态场 ----------

    def set_radial_gradient(
        self,
        channel: str,
        cx: float,
        cy: float,
        amplitude: float = 1.0,
        length_scale: float = 20.0,
    ) -> None:
        """把某个通道设成以 (cx, cy) 为中心的静态径向梯度场。

        浓度 = amplitude * exp(-distance / length_scale)

        为什么需要这个
        --------------
        点源 + 扩散产生的场，衰减长度约为 sqrt(D / (1 - decay))。
        在 60x60 的世界里，要让梯度延伸到边界，需要极慢的衰减，
        而极慢的衰减意味着场要很久才能达到稳态。

        静态梯度场绕开了这个问题：它直接定义「环境长什么样」，
        而不是靠动力学演化出来。这在生物学上也是合理的——
        巢穴的气味是环境的持久属性，不是每 tick 重新扩散出来的。

        关键：个体依然只感知局部梯度，不知道巢穴的绝对位置。
        信息存在环境里，不在个体里。

        Parameters
        ----------
        channel : str
            要设置的通道名。
        cx, cy : float
            梯度中心（例如巢穴位置）。
        amplitude : float
            中心处的浓度。
        length_scale : float
            衰减长度（格数）。越大，梯度越平缓、传得越远。
        """
        if length_scale <= 0:
            raise ValueError(f"length_scale 必须为正，收到 {length_scale}")

        ci = self.channel_index(channel)
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        dist = np.hypot(xs - cx, ys - cy)
        self.data[ci] = amplitude * np.exp(-dist / length_scale)

    # ---------- 工具 ----------

    def clear(self) -> None:
        """清空所有通道。"""
        self.data.fill(0.0)

    def snapshot(self) -> np.ndarray:
        """返回当前状态的副本（用于哈希或可视化）。"""
        return self.data.copy()

    def __repr__(self) -> str:
        return (
            f"PheromoneField({self.width}x{self.height}, "
            f"channels={self.channel_names})"
        )
