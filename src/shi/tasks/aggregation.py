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
    "shape_aligned_boundary",
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
    n_rings: int = 3,
    ring_gap: float = 6.0,
    gap_ratio: float = 0.15,
    rng: np.random.Generator | None = None,
) -> None:
    """方法 B：边界条件设计（同心环 + 缺口）。

    设计
    ----
    在目标周围画 n_rings 个同心环墙，每个环上开一个缺口。
    缺口的位置逐环错开，形成一条需要「绕行」的路径。

    为什么这样设计
    --------------
    早期版本用「漏斗」设计，结果阻挡了 94.5% 的格子，
    个体一开始就站在被阻挡的格子上，完全冻住——这不是「边界无效」，
    是边界实现错误。

    同心环 + 缺口的设计更合理：
    - 只阻挡环本身（细线），不阻挡大片区域
    - 个体必须找到缺口才能进入下一层
    - 缺口错开，迫使个体绕行
    - 这正是「探索能力」应该起作用的地方

    Parameters
    ----------
    n_rings : int
        同心环数量。
    ring_gap : float
        环之间的间距。
    gap_ratio : float
        缺口占整圈的比例（0-1）。0.15 表示缺口约 54 度。
    rng : numpy.random.Generator | None
        缺口位置的随机源。
    """
    w = task.world
    w.clear()

    if rng is None:
        rng = np.random.default_rng(0)

    blocked = np.zeros((w.height, w.width), dtype=bool)
    ys, xs = np.mgrid[0 : w.height, 0 : w.width]

    # 到目标的距离和角度
    dx = xs - task.target_x
    dy = ys - task.target_y
    dist = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    # 环的厚度：至少 1.5 格，否则在整数网格上采不到
    thickness = max(1.5, ring_gap * 0.25)

    for k in range(1, n_rings + 1):
        radius = k * ring_gap
        # 环本身：距离在 [radius-thickness/2, radius+thickness/2] 内
        on_ring = np.abs(dist - radius) <= thickness / 2.0

        # 缺口：随机角度，逐环错开
        gap_angle = rng.random() * 2.0 * np.pi
        # 角度差（归一化到 [-pi, pi]）
        angle_diff = np.abs(
            np.arctan2(np.sin(angle - gap_angle), np.cos(angle - gap_angle))
        )
        in_gap = angle_diff < (gap_ratio * 2.0 * np.pi) / 2.0

        blocked |= on_ring & ~in_gap

    task.blocked = blocked


def shape_aligned_boundary(
    task: AggregationTask,
    grad_threshold: float = 0.5,
    min_target_dist: float = 5.0,
) -> float:
    """方法 B'：对齐约束（只在梯度弱处放边界）。

    设计动机
    --------
    实验发现：冲突的约束会破坏梯度引导。
    势能场单独用聚集度 0.926，加上同心环边界后掉到 0.209。

    机制：个体沿梯度朝目标走，撞上墙被挡住，梯度仍指向目标（穿过墙），
    个体在墙边反复尝试无法前进，永久卡住。

    对齐约束的思路
    --------------
    只在**梯度接近零**的地方放边界。这些地方个体本来也不会主动去，
    所以不会造成「想做但做不到」的冲突。

    效果（实测）
    ------------
    - 梯度阈值 0.5：边界密度 0.7%，聚集度 0.921（几乎无损）
    - 梯度阈值 2.0：边界密度 26.8%，聚集度 0.505
    - 对比冲突边界：密度 1.3% 时聚集度已掉到 0.575

    同样密度下，对齐约束的性能显著更高。

    Parameters
    ----------
    grad_threshold : float
        梯度强度阈值。低于此值的地方才放边界。
    min_target_dist : float
        目标附近不阻挡的最小距离。

    Returns
    -------
    float
        边界密度（被阻挡格子的比例）。
    """
    w = task.world
    ys, xs = np.mgrid[0 : w.height, 0 : w.width]

    # 计算全场的梯度强度
    field = w.data[0]
    gy, gx = np.gradient(field)
    grad_mag = np.hypot(gx, gy)

    # 只在梯度弱的地方放边界
    blocked = grad_mag < grad_threshold

    # 目标附近不阻挡
    dist = np.hypot(xs - task.target_x, ys - task.target_y)
    blocked &= dist > min_target_dist

    task.blocked = blocked
    return float(blocked.mean())


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
    exploration: float = 0.0,
    explore_steps: int = 20,
):
    """构造聚集任务的个体行为更新函数。

    个体规则**固定且极简**：
    1. 读环境梯度
    2. 沿梯度爬升
    3. 无梯度时随机游走

    没有记忆、没有阈值、没有内部状态。
    这是 G2 的关键：规则固定，只有环境可变。

    Parameters
    ----------
    exploration : float
        探索强度（0 = 纯梯度跟随，>0 = 被阻挡时随机探索）。
        这是「同构原则」实验的关键参数：
        - exploration=0：个体只会跟随梯度，边界条件应该无效
        - exploration>0：个体有探索能力，边界条件可能变有效
    explore_steps : int
        被阻挡后持续探索的 tick 数。
    """

    # 探索状态：每个个体剩余探索 tick 数（0 表示不探索）
    explore_left = np.zeros(0, dtype=np.int32)
    explore_angle = np.zeros(0, dtype=np.float64)

    def step(agents: AgentState, world: PheromoneField, tick: int, rng: np.random.Generator) -> None:
        nonlocal explore_left, explore_angle
        n = agents.n
        if n == 0:
            return

        # 惰性初始化探索状态（个体数可能变化）
        if explore_left.shape[0] != n:
            explore_left = np.zeros(n, dtype=np.int32)
            explore_angle = np.zeros(n, dtype=np.float64)

        for i in range(n):
            # 探索模式：沿固定方向走，直到探索结束
            if explore_left[i] > 0:
                explore_left[i] -= 1
                agents.vx[i] = speed * np.cos(explore_angle[i])
                agents.vy[i] = speed * np.sin(explore_angle[i])
                continue

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

            # 若个体当前站在被阻挡的格子上（例如初始位置），
            # 允许它移动到任意相邻的非阻挡格（逃逸规则）。
            # 没有这条规则，个体会被永久冻住。
            if task.is_blocked(agents.x[i], agents.y[i]):
                escaped = False
                for _ in range(8):
                    ang = rng.random() * 2.0 * np.pi
                    ex = float(np.clip(agents.x[i] + np.cos(ang), 0.0, world.width - 1))
                    ey = float(np.clip(agents.y[i] + np.sin(ang), 0.0, world.height - 1))
                    if not task.is_blocked(ex, ey):
                        agents.x[i] = ex
                        agents.y[i] = ey
                        escaped = True
                        break
                if escaped:
                    continue

            if not task.is_blocked(nx, ny):
                agents.x[i] = nx
                agents.y[i] = ny
            else:
                # 被阻挡：尝试只沿 x 或只沿 y 移动（滑动）
                if not task.is_blocked(nx, agents.y[i]):
                    agents.x[i] = nx
                elif not task.is_blocked(agents.x[i], ny):
                    agents.y[i] = ny

                # 探索：被阻挡且探索强度 > 0 时，随机选方向探索
                if exploration > 0.0 and rng.random() < exploration:
                    explore_left[i] = explore_steps
                    explore_angle[i] = rng.random() * 2.0 * np.pi

    return step
