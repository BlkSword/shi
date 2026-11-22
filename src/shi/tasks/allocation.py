"""多任务分配：异质性真正能起作用的任务.

为什么需要这个任务
------------------
觅食任务只有一个目标（找资源），阈值异质性只是整体缩放，
没有分工空间。实验证实：sigma 从 0 到 1.0，覆盖率完全平坦。

异质性要起作用，必须有**多个任务**，且**分工有收益**。
这是响应阈值模型的经典设定（Bonabeau / Theraulaz）。

任务设计
--------
- K 个任务类型，每个类型有独立的刺激通道 task_0 ... task_{K-1}
- 每个类型有若干任务点，每个点有需求 demand（会缓慢再生）
- 每个个体有**阈值向量** theta[K]（不是标量）
- 个体每 tick 感知所有通道，算 p_k = s_k^2 / (s_k^2 + theta_k^2)，选 argmax
- 移动到该类型任务点，做工作，减少 demand

为什么异质性会有倒 U 形
------------------------
- sigma = 0（全同）：所有个体偏好同一个任务类型，扎堆，其他类型饿死
- sigma 适中：个体自然分化到不同类型，并行工作，总产出最高
- sigma 过大：部分个体阈值高到永不响应（闲置），部分低到频繁切换（抖动）

这个机制是 H3 的直接检验。
"""

from __future__ import annotations

import numpy as np

from shi.core.agent import AgentState, response_probability
from shi.core.world import ChannelSpec, PheromoneField

__all__ = ["AllocationTask", "make_allocation_stepper", "make_allocation_world"]


def make_allocation_world(width: int, height: int, n_types: int) -> PheromoneField:
    """构造多任务环境：每个任务类型一个通道。"""
    channels = [
        ChannelSpec(f"task_{k}", decay=1.0, diffusion=0.0) for k in range(n_types)
    ]
    return PheromoneField(width, height, channels=channels)


class AllocationTask:
    """多任务分配任务。

    Parameters
    ----------
    world : PheromoneField
        必须包含 task_0 ... task_{K-1} 通道。
    n_types : int
        任务类型数。
    sites_per_type : int
        每个类型的任务点数量。
    demand : float
        每个任务点的初始需求。
    regen : float
        每 tick 需求再生量（维持持续刺激）。
    rng : numpy.random.Generator
        放置任务点的随机源。
    """

    def __init__(
        self,
        world: PheromoneField,
        n_types: int = 3,
        sites_per_type: int = 6,
        demand: float = 20.0,
        regen: float = 0.05,
        rng: np.random.Generator | None = None,
    ):
        self.world = world
        self.n_types = int(n_types)
        self.sites_per_type = int(sites_per_type)
        self.demand = float(demand)
        self.regen = float(regen)
        self._rng = rng if rng is not None else np.random.default_rng(0)

        for k in range(self.n_types):
            if f"task_{k}" not in world.channel_names:
                raise ValueError(f"缺少通道 task_{k}")

        # sites[k] = [(x, y), ...]
        self.sites: list[list[tuple[float, float]]] = [[] for _ in range(self.n_types)]
        self.work_done = 0.0
        self.work_by_type = np.zeros(self.n_types, dtype=np.float64)

    def setup(self) -> None:
        """放置任务点并初始化需求。"""
        self.world.clear()
        self.sites = [[] for _ in range(self.n_types)]

        for k in range(self.n_types):
            for _ in range(self.sites_per_type):
                x = float(self._rng.random() * self.world.width)
                y = float(self._rng.random() * self.world.height)
                self.sites[k].append((x, y))
                self.world.deposit(x, y, f"task_{k}", self.demand)

    def regenerate(self) -> None:
        """任务需求缓慢再生，维持持续刺激。"""
        for k in range(self.n_types):
            for x, y in self.sites[k]:
                cur = self.world.read(x, y, f"task_{k}")
                if cur < self.demand:
                    self.world.deposit(x, y, f"task_{k}", self.regen)

    def work_at(self, x: float, y: float, k: int, amount: float = 1.0) -> float:
        """在 (x, y) 处对类型 k 做工作。返回实际完成量。"""
        ch = f"task_{k}"
        available = self.world.read(x, y, ch)
        if available <= 0:
            return 0.0
        done = min(amount, available)
        self.world.deposit(x, y, ch, -done)
        self.work_done += done
        self.work_by_type[k] += done
        return done

    def total_demand(self) -> float:
        return self.n_types * self.sites_per_type * self.demand

    def completion(self) -> float:
        """累计完成量占总需求的比例（可超过 1，因为需求会再生）。"""
        total = self.total_demand()
        if total <= 0:
            return 0.0
        return self.work_done / total

    def balance(self) -> float:
        """工作在各类型间的均衡度（0-1，1 表示完全均衡）。"""
        total = self.work_by_type.sum()
        if total <= 0:
            return 0.0
        p = self.work_by_type / total
        # 归一化熵
        ent = -np.sum(p * np.log(p + 1e-12))
        return float(ent / np.log(self.n_types))


def make_allocation_stepper(
    task: AllocationTask,
    speed: float = 0.6,
    noise: float = 0.2,
    sense_radius: int = 2,
    work_amount: float = 1.0,
):
    """构造多任务分配的个体行为更新函数。

    个体每 tick：
    1. 感知所有任务类型的刺激
    2. 用阈值向量算响应概率，选 argmax
    3. 沿该类型梯度移动
    4. 到达任务点则做工作
    """
    K = task.n_types

    def step(agents: AgentState, world: PheromoneField, tick: int, rng: np.random.Generator) -> None:
        n = agents.n
        if n == 0:
            return

        task.regenerate()

        # ---- 1. 感知所有类型 ----
        stim = np.empty((n, K), dtype=np.float64)
        for k in range(K):
            ch = f"task_{k}"
            for i in range(n):
                stim[i, k] = world.sense(
                    agents.x[i], agents.y[i], ch, sense_radius, normalize=True
                )

        # ---- 2. 阈值向量响应，选 argmax ----
        # agents.threshold 形状 (n, K)
        p = response_probability(stim, agents.threshold)
        choice = np.argmax(p, axis=1)
        best_p = p[np.arange(n), choice]

        # 响应概率太低则闲置
        active = best_p > 0.5

        # ---- 3. 移动 ----
        for i in range(n):
            if not active[i]:
                agents.vx[i] = noise * (rng.random() - 0.5) * 2.0
                agents.vy[i] = noise * (rng.random() - 0.5) * 2.0
                continue
            k = int(choice[i])
            gx, gy = world.gradient(agents.x[i], agents.y[i], f"task_{k}")
            norm = np.hypot(gx, gy)
            if norm > 1e-12:
                agents.vx[i] = speed * gx / norm
                agents.vy[i] = speed * gy / norm
            else:
                agents.vx[i] = noise * (rng.random() - 0.5) * 2.0
                agents.vy[i] = noise * (rng.random() - 0.5) * 2.0

        agents.x += agents.vx
        agents.y += agents.vy
        agents.clamp_to(float(world.width - 1), float(world.height - 1))

        # ---- 4. 做工作 ----
        for i in range(n):
            if not active[i]:
                continue
            k = int(choice[i])
            task.work_at(agents.x[i], agents.y[i], k, work_amount)

        # 记录状态（用 state 字段存当前选择的类型 + 1，0 表示闲置）
        agents.state[:] = 0
        agents.state[active] = (choice[active] + 1).astype(np.int8)

    return step
