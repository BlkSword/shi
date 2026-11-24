"""目标聚集任务：G2 目标注入的实验平台.

核心设计
--------
个体规则**固定且极简**：只做一件事——跟随环境梯度移动。
没有任何记忆、没有阈值、没有内部状态。

这样，系统的行为完全由环境决定。
环境是唯一的控制手段，这正是 G2 要研究的。

目标
----
让所有个体聚集到指定位置 g。

三个环境塑造方法
----------------
A. 势能景观：在 g 处放吸引子（高浓度），形成全局最小值
B. 边界条件：用不可通行区域形成漏斗，把个体导向 g
C. 反馈通道：环境对「靠近 g」的行为增强痕迹，对「远离」加速衰减
"""

from __future__ import annotations

import numpy as np

from shi.core.agent import AgentState
from shi.core.world import ChannelSpec, PheromoneField

__all__ = [
    "AggregationTask",
    "make_aggregation_world",
    "make_aggregation_stepper",
    "shape_potential",
    "shape_boundary",
    "shape_feedback",
]


def make_aggregation_world(
    width: int,
    height: int,
    decay: float = 1.0,
    diffusion: float = 0.0,
) -> PheromoneField:
    """构造聚集任务的环境。

    单通道 "field"，静态（decay=1.0, diffusion=0.0），
    因为环境由塑造方法直接写入，不需要动力学演化。
    """
    channels = [ChannelSpec("field", decay=decay, diffusion=diffusion)]
    return PheromoneField(width, height, channels=channels)


class AggregationTask:
    """目标聚集任务。

    Parameters
    ----------
    world : PheromoneField
        必须包含 "field" 通道。
    target_x, target_y : float
        目标位置 g。
    """

    def __init__(
        self,
        world: PheromoneField,
        target_x: float,
        target_y: float,
    ):
        if "field" not in world.channel_names:
            raise ValueError("聚集任务需要 'field' 通道")
        self.world = world
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.blocked: np.ndarray | None = None  # 边界条件用

    def mean_distance(self, agents: AgentState) -> float:
        """个体到目标的平均距离（越小越好）。"""
        if agents.n == 0:
            return float("nan")
        d = np.hypot(agents.x - self.target_x, agents.y - self.target_y)
        return float(d.mean())

    def aggregation_score(self, agents: AgentState, scale: float = 10.0) -> float:
        """聚集度（0-1，越大越好）。

        用指数衰减把距离映射到 [0, 1]：
        score = exp(-mean_distance / scale)
        """
        d = self.mean_distance(agents)
        if np.isnan(d):
            return 0.0
        return float(np.exp(-d / scale))

    def is_blocked(self, x: float, y: float) -> bool:
        """检查位置是否被边界条件阻挡。"""
        if self.blocked is None:
            return False
        cy = int(np.clip(np.floor(y), 0, self.world.height - 1))
        cx = int(np.clip(np.floor(x), 0, self.world.width - 1))
        return bool(self.blocked[cy, cx])


# ============================================================
# 三个环境塑造方法
# ============================================================


def shape_potential(
    task: AggregationTask,
    amplitude: float = 100.0,
    length_scale: float = 15.0,
    n_repellers: int = 0,
    repeller_amplitude: float = 0.0,
    rng: np.random.Generator | None = None,
) -> None:
    """方法 A：势能景观塑造。

    在目标位置放一个高斯吸引子，形成全局最小值。
    可选在随机位置放排斥子（负浓度），把个体推离。

    Parameters
    ----------
    amplitude : float
        吸引子中心浓度。
    length_scale : float
        吸引子的空间尺度（越大越平缓）。
    n_repellers : int
        排斥子数量。
    repeller_amplitude : float
        排斥子强度（负值）。
    """
    w = task.world
    w.clear()

    # 吸引子：高斯分布
    ys, xs = np.mgrid[0 : w.height, 0 : w.width]
    dist = np.hypot(xs - task.target_x, ys - task.target_y)
    w.data[0] = amplitude * np.exp(-(dist ** 2) / (2.0 * length_scale ** 2))

    # 排斥子
    if n_repellers > 0 and repeller_amplitude != 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        for _ in range(n_repellers):
            rx = float(rng.random() * w.width)
            ry = float(rng.random() * w.height)
            # 排斥子不能放在目标附近
            if np.hypot(rx - task.target_x, ry - task.target_y) < 3 * length_scale:
                continue
            d = np.hypot(xs - rx, ys - ry)
            w.data[0] += repeller_amplitude * np.exp(-(d ** 2) / (2.0 * (length_scale * 0.5) ** 2))


def shape_boundary(
    task: AggregationTask,
    funnel_width: float = 8.0,
    wall_thickness: int = 2,
) -> None:
    """方法 B：边界条件设计。

    用不可通行区域形成漏斗，把个体导向目标。

    实现：在目标周围画一个「漏斗」——
    从世界边缘到目标，逐渐收窄的通道。
    通道外的区域被标记为 blocked。
    """
    w = task.world
    w.clear()

    blocked = np.zeros((w.height, w.width), dtype=bool)
    ys, xs = np.mgrid[0 : w.height, 0 : w.width]

    # 到目标的距离
    dist = np.hypot(xs - task.target_x, ys - task.target_y)

    # 漏斗：距离目标越远，允许的横向偏移越小
    # 用极坐标：角度偏离目标方向超过阈值就阻挡
    angle_to_target = np.arctan2(task.target_y - ys, task.target_x - xs)
    angle_from_center = np.arctan2(ys - w.height / 2.0, xs - w.width / 2.0)

    # 角度差（归一化到 [-pi, pi]）
    angle_diff = np.abs(np.arctan2(
        np.sin(angle_from_center - angle_to_target),
        np.cos(angle_from_center - angle_to_target),
    ))

    # 允许的角度范围随距离收窄
    max_angle = np.pi * np.clip(dist / (w.width * 0.8), 0.15, 1.0)
    blocked = angle_diff > max_angle

    # 目标附近不阻挡
    blocked &= dist > funnel_width

    task.blocked = blocked


def shape_feedback(
    task: AggregationTask,
    base_level: float = 1.0,
    gradient_strength: float = 10.0,
) -> None:
    """方法 C：反馈通道设计。

    环境对「靠近目标」的位置给更高浓度，
    形成连续的反馈信号（而不是方法 A 的单一吸引子）。

    与方法 A 的区别：
    - A 是局部吸引子（高斯），远处梯度为零
    - C 是全局线性梯度，任何位置都有指向目标的信号
    """
    w = task.world
    w.clear()

    ys, xs = np.mgrid[0 : w.height, 0 : w.width]
    dist = np.hypot(xs - task.target_x, ys - task.target_y)
    max_dist = np.hypot(w.width, w.height)

    # 线性衰减：距离越近浓度越高
    w.data[0] = base_level + gradient_strength * (1.0 - dist / max_dist)


def make_aggregation_stepper(
    task: AggregationTask,
    speed: float = 0.8,
    noise: float = 0.1,
):
    """构造聚集任务的个体行为更新函数。

    个体规则**固定且极简**：
    1. 读环境梯度
    2. 沿梯度爬升
    3. 无梯度时随机游走

    没有记忆、没有阈值、没有内部状态。
    这是 G2 的关键：规则固定，只有环境可变。
    """

    def step(agents: AgentState, world: PheromoneField, tick: int, rng: np.random.Generator) -> None:
        n = agents.n
        if n == 0:
            return

        for i in range(n):
            gx, gy = world.gradient(agents.x[i], agents.y[i], "field")
            norm = np.hypot(gx, gy)

            if norm > 1e-12:
                agents.vx[i] = speed * gx / norm
                agents.vy[i] = speed * gy / norm
            else:
                agents.vx[i] = noise * (rng.random() - 0.5) * 2.0
                agents.vy[i] = noise * (rng.random() - 0.5) * 2.0

        # 尝试移动，被阻挡则原地不动
        new_x = agents.x + agents.vx
        new_y = agents.y + agents.vy

        for i in range(n):
            nx = float(np.clip(new_x[i], 0.0, world.width - 1))
            ny = float(np.clip(new_y[i], 0.0, world.height - 1))
            if not task.is_blocked(nx, ny):
                agents.x[i] = nx
                agents.y[i] = ny
            else:
                # 被阻挡：尝试只沿 x 或只沿 y 移动（滑动）
                if not task.is_blocked(nx, agents.y[i]):
                    agents.x[i] = nx
                elif not task.is_blocked(agents.x[i], ny):
                    agents.y[i] = ny

    return step
