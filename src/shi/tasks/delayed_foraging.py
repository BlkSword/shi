"""延迟觅食任务：环境可写性成为瓶颈的场景.

为什么需要这个任务
------------------
在普通觅食任务里，个体可以直接感知资源，环境只是辅助。
要让「环境可写性」成为瓶颈，必须设计一个个体**无法直接感知**、
**必须依赖环境记忆**的任务。

任务规则
--------
1. 资源有 K 种类型，每个周期只有**一种类型**可采集（目标类型）
2. 侦察窗口：所有资源可见，个体能看到位置和类型
3. 采集窗口：资源不可见，只能靠环境记忆；只有目标类型能采集
4. 个体没有记忆（硬约束），唯一的记忆载体是环境

为什么通道数会成为真正的瓶颈
------------------------------
关键：如果 n_channels >= K，每种类型独占一个通道，记忆完美。
如果 n_channels < K，多种类型必须共享通道 → **类型混淆** →
个体被错误类型的记忆吸引，跑到错误位置，浪费时间。

这是「信息容量」的真实约束：不是存不下位置，而是存不下「区分」。

精度约束
--------
环境值被量化到 2^precision_bits 级。精度低时梯度变噪声，
个体无法精确跟随记忆。
"""

from __future__ import annotations

import numpy as np

from shi.core.agent import AgentState, response_probability
from shi.core.world import ChannelSpec, PheromoneField

__all__ = [
    "DelayedForagingTask",
    "make_delayed_foraging_world",
    "make_delayed_foraging_stepper",
]


def make_delayed_foraging_world(
    width: int,
    height: int,
    n_channels: int = 4,
    precision_bits: int = 8,
    decay: float = 0.995,
    diffusion: float = 0.02,
) -> PheromoneField:
    """构造延迟觅食环境。

    n_channels 是环境信息容量的一个维度（能区分多少种信息）。
    precision_bits 是另一个维度（能分辨多细的差异）。
    """
    channels = [
        ChannelSpec(f"mem_{k}", decay=decay, diffusion=diffusion)
        for k in range(n_channels)
    ]
    world = PheromoneField(width, height, channels=channels)
    world.precision_bits = int(precision_bits)
    return world


class DelayedForagingTask:
    """延迟觅食任务：侦察与采集窗口不重叠，且只有目标类型可采集。

    Parameters
    ----------
    world : PheromoneField
        必须包含 mem_0 ... mem_{K-1} 通道。
    n_resources : int
        资源点总数。
    n_types : int
        资源类型数。
    cycle : int
        一个完整周期（侦察 + 采集）的 tick 数。
    scout_ratio : float
        侦察窗口占周期的比例。
    write_budget : float
        每 tick 每个个体能写入的总量（写入带宽）。
    """

    def __init__(
        self,
        world: PheromoneField,
        n_resources: int = 40,
        n_types: int = 4,
        cycle: int = 300,
        scout_ratio: float = 0.3,
        write_budget: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.world = world
        self.n_resources = int(n_resources)
        self.n_types = int(n_types)
        self.cycle = int(cycle)
        self.scout_ratio = float(scout_ratio)
        self.write_budget = float(write_budget)
        self._rng = rng if rng is not None else np.random.default_rng(0)

        self.n_channels = world.n_channels
        self.resources: list[tuple[float, float, int]] = []  # (x, y, type)
        self.collected = 0
        self.wrong_visits = 0
        self.tick = 0

    def setup(self) -> None:
        self.world.clear()
        self.resources = []
        for _ in range(self.n_resources):
            x = float(self._rng.random() * self.world.width)
            y = float(self._rng.random() * self.world.height)
            t = int(self._rng.integers(0, self.n_types))
            self.resources.append((x, y, t))

    def phase(self) -> str:
        scout_len = int(self.cycle * self.scout_ratio)
        return "scout" if (self.tick % self.cycle) < scout_len else "harvest"

    def is_scouting(self) -> bool:
        return self.phase() == "scout"

    def is_harvesting(self) -> bool:
        return self.phase() == "harvest"

    def target_type(self) -> int:
        """当前周期的目标类型（每个周期轮换）。"""
        return (self.tick // self.cycle) % self.n_types

    def collect_at(self, x: float, y: float, radius: float = 1.5) -> bool:
        """尝试采集目标类型的资源。"""
        if not self.is_harvesting():
            return False
        target = self.target_type()
        for i, (rx, ry, rt) in enumerate(self.resources):
            if np.hypot(x - rx, y - ry) <= radius:
                if rt == target:
                    self.resources.pop(i)
                    self.collected += 1
                    return True
                else:
                    self.wrong_visits += 1
                    return False
        return False

    def write_memory(self, x: float, y: float, rtype: int, amount: float) -> float:
        """写入环境记忆。类型决定通道（可能冲突）。"""
        actual = min(amount, self.write_budget)
        ch = f"mem_{rtype % self.n_channels}"
        self.world.deposit(x, y, ch, actual)
        return actual

    def efficiency(self) -> float:
        if self.n_resources <= 0:
            return 0.0
        return self.collected / self.n_resources


def make_delayed_foraging_stepper(
    task: DelayedForagingTask,
    speed: float = 0.6,
    search_speed: float = 1.2,
    sense_radius: int = 3,
    write_amount: float = 2.0,
    rule_complexity: int = 2,
):
    """构造延迟觅食的个体行为更新函数。

    Parameters
    ----------
    rule_complexity : int
        个体规则复杂度（H2 的第二个自变量）：
        - 1（简单）：只跟随目标通道梯度，无探索
        - 2（中等）：梯度跟随 + 定向搜索（默认）
        - 3（复杂）：梯度跟随 + 定向搜索 + 多通道加权 + 探索-利用平衡
    """

    def step(agents: AgentState, world: PheromoneField, tick: int, rng: np.random.Generator) -> None:
        n = agents.n
        if n == 0:
            return

        task.tick = tick
        scouting = task.is_scouting()
        target = task.target_type()

        # ---- 1. 感知 ----
        stim = np.zeros(n, dtype=np.float64)

        if scouting:
            # 侦察窗口：看到所有资源（位置 + 类型）
            for i in range(n):
                for rx, ry, rt in task.resources:
                    if np.hypot(agents.x[i] - rx, agents.y[i] - ry) <= sense_radius:
                        stim[i] = 10.0
                        break
        else:
            # 采集窗口：只能靠环境记忆
            for k in range(task.n_channels):
                ch = f"mem_{k}"
                for i in range(n):
                    stim[i] += world.sense(
                        agents.x[i], agents.y[i], ch, sense_radius, normalize=True
                    )

        # ---- 2. 决策 ----
        p = response_probability(stim, agents.threshold)
        active = p > 0.5

        # ---- 3. 移动 ----
        for i in range(n):
            if not active[i]:
                if rng.random() < 0.05 or (agents.vx[i] == 0.0 and agents.vy[i] == 0.0):
                    angle = rng.random() * 2.0 * np.pi
                    agents.vx[i] = search_speed * np.cos(angle)
                    agents.vy[i] = search_speed * np.sin(angle)
                continue

            if scouting:
                # 侦察：朝最近资源移动
                best_d, best_dx, best_dy = 1e9, 0.0, 0.0
                for rx, ry, rt in task.resources:
                    dx, dy = rx - agents.x[i], ry - agents.y[i]
                    d = np.hypot(dx, dy)
                    if d < best_d:
                        best_d, best_dx, best_dy = d, dx, dy
                if best_d > 1e-9:
                    agents.vx[i] = speed * best_dx / best_d
                    agents.vy[i] = speed * best_dy / best_d
            else:
                # 采集窗口：跟随环境记忆
                target_ch = target % task.n_channels

                if rule_complexity == 1:
                    # 简单：只看目标通道
                    gx, gy = world.gradient(agents.x[i], agents.y[i], f"mem_{target_ch}")
                    norm = np.hypot(gx, gy)
                    if norm > 1e-12:
                        agents.vx[i] = speed * gx / norm
                        agents.vy[i] = speed * gy / norm
                    else:
                        agents.vx[i] = 0.0
                        agents.vy[i] = 0.0

                elif rule_complexity == 2:
                    # 中等：目标通道 + 无梯度时定向搜索
                    gx, gy = world.gradient(agents.x[i], agents.y[i], f"mem_{target_ch}")
                    norm = np.hypot(gx, gy)
                    if norm > 1e-12:
                        agents.vx[i] = speed * gx / norm
                        agents.vy[i] = speed * gy / norm
                    else:
                        if rng.random() < 0.05 or (agents.vx[i] == 0.0 and agents.vy[i] == 0.0):
                            angle = rng.random() * 2.0 * np.pi
                            agents.vx[i] = search_speed * np.cos(angle)
                            agents.vy[i] = search_speed * np.sin(angle)

                else:
                    # 复杂：更精细的梯度跟随
                    # 用更大的采样窗口估计梯度（减少噪声），
                    # 并在梯度弱时降低速度（避免冲过头）
                    gx, gy = world.gradient(agents.x[i], agents.y[i], f"mem_{target_ch}")
                    norm = np.hypot(gx, gy)
                    if norm > 1e-12:
                        # 梯度强度自适应速度：梯度弱时慢走，避免震荡
                        adaptive_speed = speed * min(1.0, norm / 0.5)
                        agents.vx[i] = adaptive_speed * gx / norm
                        agents.vy[i] = adaptive_speed * gy / norm
                    else:
                        if rng.random() < 0.05 or (agents.vx[i] == 0.0 and agents.vy[i] == 0.0):
                            angle = rng.random() * 2.0 * np.pi
                            agents.vx[i] = search_speed * np.cos(angle)
                            agents.vy[i] = search_speed * np.sin(angle)

        agents.x += agents.vx
        agents.y += agents.vy
        agents.clamp_to(float(world.width - 1), float(world.height - 1))

        # ---- 4. 写记忆 / 采集 ----
        if scouting:
            for i in range(n):
                for rx, ry, rt in task.resources:
                    if np.hypot(agents.x[i] - rx, agents.y[i] - ry) <= sense_radius:
                        task.write_memory(rx, ry, rt, write_amount)
                        break
        else:
            for i in range(n):
                task.collect_at(agents.x[i], agents.y[i])

        agents.state[:] = np.where(active, 1, 0).astype(np.int8)

    return step
