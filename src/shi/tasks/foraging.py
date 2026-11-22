"""觅食任务：找到并搬运资源回巢.

任务描述
--------
- 环境里散布着资源点
- 个体从巢穴出发，找到资源，搬回巢穴
- 没有中央调度，个体只靠局部感知和信息素场协作

三个通道，各司其职
------------------
这是本任务最重要的设计决策。早期版本把「资源」和「招募信息素」
混在同一个 supply 通道里，导致个体留下的信息素淹没了真实资源信号，
系统在 500 tick 后就失去了资源分布的信息。

现在拆成三个通道：

    supply   资源。静态分布，被采集时减少。个体只读不写。
    recruit  招募信息素。个体写入，扩散 + 衰减。表示「这里有人找到东西了」。
    order    巢穴标记。在巢穴处持续强化并扩散。个体据此找到回家的方向。

通道分离是「环境设计」的一个具体例子：不同的信息需要不同的
生命周期（衰减率、扩散率），混在一起就会互相污染。

个体行为
--------
1. 感知：读取所在格的 supply / recruit / order（归一化，与阈值同量级）
2. 决策：用 Hill 函数算响应概率，与个体阈值比较
3. 动作：沿目标通道梯度移动，或随机游走
4. 交互：采集资源、交付资源、留下招募信息素

关键设计：个体不「知道」巢穴在哪，它只知道 order 通道的梯度。
巢穴是环境的一部分，不是个体记忆的一部分。
"""

from __future__ import annotations

import numpy as np

from shi.core.agent import (
    STATE_CARRY,
    STATE_IDLE,
    STATE_RETURN,
    STATE_SEEK,
    AgentState,
    response_probability,
)
from shi.core.world import PheromoneField

__all__ = ["ForagingTask", "make_foraging_stepper", "FORAGING_CHANNELS"]

# 觅食任务的通道配置：名称 -> (衰减率, 扩散率)
FORAGING_CHANNELS: dict[str, tuple[float, float]] = {
    "supply": (1.0, 0.0),    # 资源：静态分布，不衰减不扩散
    "recruit": (0.95, 0.10), # 招募信息素：快衰减 + 扩散
    "order": (1.0, 0.0),     # 巢穴标记：静态势场（由 setup 直接写入）
}


class ForagingTask:
    """觅食任务的环境与规则。

    Parameters
    ----------
    world : PheromoneField
        环境场。必须包含 supply / recruit / order 三个通道。
    nest_x, nest_y : float
        巢穴位置。
    n_resources : int
        资源点数量。
    resource_amount : float
        每个资源点的初始浓度。
    rng : numpy.random.Generator
        用于放置资源的随机源。
    """

    def __init__(
        self,
        world: PheromoneField,
        nest_x: float,
        nest_y: float,
        n_resources: int = 50,
        resource_amount: float = 10.0,
        rng: np.random.Generator | None = None,
    ):
        for ch in ("supply", "recruit", "order"):
            if ch not in world.channel_names:
                raise ValueError(f"觅食任务需要 {ch!r} 通道")

        self.world = world
        self.nest_x = float(nest_x)
        self.nest_y = float(nest_y)
        self.n_resources = int(n_resources)
        self.resource_amount = float(resource_amount)
        self._rng = rng if rng is not None else np.random.default_rng(0)

        self.resources: list[tuple[float, float]] = []
        self.collected = 0
        self.delivered = 0

    def setup(self) -> None:
        """放置资源点，并在巢穴处建立 order 通道的标记。"""
        self.world.clear()
        self.resources = []
        for _ in range(self.n_resources):
            x = float(self._rng.random() * self.world.width)
            y = float(self._rng.random() * self.world.height)
            self.resources.append((x, y))
            self.world.deposit(x, y, "supply", self.resource_amount)

        # 巢穴标记：静态径向梯度场
        # 中心浓度 10，衰减长度 25 格 —— 覆盖整个 60x60 世界
        self.world.set_radial_gradient(
            "order", self.nest_x, self.nest_y, amplitude=10.0, length_scale=25.0
        )

    def maintain_nest_marker(self) -> None:
        """巢穴标记是静态场，无需每 tick 维护。

        保留这个方法是为了接口一致性（未来可能改成动态维护）。
        """
        pass

    def collect_at(self, x: float, y: float) -> bool:
        """尝试在 (x, y) 采集资源。成功返回 True。"""
        available = self.world.read(x, y, "supply")
        if available >= 1.0:
            self.world.deposit(x, y, "supply", -1.0)
            self.collected += 1
            return True
        return False

    def deliver_at(self, x: float, y: float, radius: float = 2.0) -> bool:
        """尝试在巢穴附近交付资源。成功返回 True。"""
        dx = x - self.nest_x
        dy = y - self.nest_y
        if (dx * dx + dy * dy) <= radius * radius:
            self.delivered += 1
            return True
        return False

    def coverage(self) -> float:
        """已采集比例（0-1），作为稳态性能指标。"""
        total = self.n_resources * self.resource_amount
        if total <= 0:
            return 0.0
        return min(1.0, self.collected / total)

    def delivery_rate(self) -> float:
        """交付率：交付数 / 采集数。衡量「找到资源后能不能带回来」。"""
        if self.collected == 0:
            return 0.0
        return self.delivered / self.collected


def make_foraging_stepper(
    task: ForagingTask,
    speed: float = 0.5,
    noise: float = 0.3,
    sense_radius: int = 2,
    recruit_amount: float = 0.05,
):
    """构造觅食任务的个体行为更新函数。

    Parameters
    ----------
    task : ForagingTask
        任务实例。
    speed : float
        每 tick 移动的最大距离。
    noise : float
        随机游走的强度。
    sense_radius : int
        感知半径。
    recruit_amount : float
        携带资源的个体每 tick 留下的招募信息素量。

    Returns
    -------
    Stepper
        可直接传给 Engine 的函数。
    """

    def step(agents: AgentState, world: PheromoneField, tick: int, rng: np.random.Generator) -> None:
        n = agents.n
        if n == 0:
            return

        task.maintain_nest_marker()

        # ---- 1. 感知（归一化，与阈值同量级）----
        supply_here = np.empty(n, dtype=np.float64)
        order_here = np.empty(n, dtype=np.float64)
        recruit_here = np.empty(n, dtype=np.float64)
        for i in range(n):
            supply_here[i] = world.sense(
                agents.x[i], agents.y[i], "supply", sense_radius, normalize=True
            )
            order_here[i] = world.sense(
                agents.x[i], agents.y[i], "order", sense_radius, normalize=True
            )
            recruit_here[i] = world.sense(
                agents.x[i], agents.y[i], "recruit", sense_radius, normalize=True
            )

        # ---- 2. 决策：阈值响应 ----
        # 三个刺激，三个响应概率，各自与个体阈值比较
        p_seek = response_probability(supply_here, agents.threshold)
        p_return = response_probability(order_here, agents.threshold)
        p_recruit = response_probability(recruit_here, agents.threshold)

        carrying = agents.carrying == 1

        # 携带者：order 信号够强就回巢
        go_return = carrying & (p_return > 0.5)
        # 空手者：supply 信号够强就去采集
        go_seek = (~carrying) & (p_seek > 0.5)
        # 空手者：招募信号够强就跟着走（但优先级低于直接看到资源）
        go_recruit = (~carrying) & (~go_seek) & (p_recruit > 0.5)

        agents.state[:] = STATE_IDLE
        agents.state[go_seek] = STATE_SEEK
        agents.state[go_recruit] = STATE_SEEK
        agents.state[go_return] = STATE_RETURN
        agents.state[carrying & ~go_return] = STATE_CARRY

        # ---- 3. 动作：沿目标通道梯度移动 ----
        for i in range(n):
            if agents.state[i] == STATE_RETURN:
                gx, gy = world.gradient(agents.x[i], agents.y[i], "order")
            elif agents.state[i] == STATE_SEEK:
                # 直接看到资源就追资源，否则追招募信息素
                if p_seek[i] > 0.5:
                    gx, gy = world.gradient(agents.x[i], agents.y[i], "supply")
                else:
                    gx, gy = world.gradient(agents.x[i], agents.y[i], "recruit")
            else:
                gx = gy = 0.0

            norm = np.hypot(gx, gy)
            if norm > 1e-12:
                agents.vx[i] = speed * gx / norm
                agents.vy[i] = speed * gy / norm
            else:
                # 无梯度时随机游走
                agents.vx[i] = noise * (rng.random() - 0.5) * 2.0
                agents.vy[i] = noise * (rng.random() - 0.5) * 2.0

        agents.x += agents.vx
        agents.y += agents.vy
        agents.clamp_to(float(world.width - 1), float(world.height - 1))

        # ---- 4. 采集与交付 ----
        for i in range(n):
            if agents.carrying[i] == 0:
                if task.collect_at(agents.x[i], agents.y[i]):
                    agents.carrying[i] = 1
            else:
                if task.deliver_at(agents.x[i], agents.y[i]):
                    agents.carrying[i] = 0

        # ---- 5. 留下招募信息素（写到 recruit 通道，不污染 supply）----
        carriers = agents.carrying == 1
        if np.any(carriers):
            world.deposit_many(
                agents.x[carriers],
                agents.y[carriers],
                "recruit",
                np.full(int(np.count_nonzero(carriers)), recruit_amount, dtype=np.float64),
            )

    return step
