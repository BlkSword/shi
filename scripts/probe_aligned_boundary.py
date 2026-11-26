"""能否设计「不冲突」的约束？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask, make_aggregation_stepper, make_aggregation_world,
    shape_potential,
)


def make_aligned_boundary(task, threshold=0.5):
    """只在梯度弱的地方放边界（不阻挡有梯度的路径）。

    思路：如果某处的梯度很强，说明个体需要经过这里，不应该阻挡。
    只在梯度接近零的地方放墙，这些地方个体本来也不会主动去。
    """
    w = task.world
    ys, xs = np.mgrid[0:w.height, 0:w.width]

    # 计算全场的梯度强度
    field = w.data[0]
    gy, gx = np.gradient(field)
    grad_mag = np.hypot(gx, gy)

    # 只在梯度弱的地方放边界
    blocked = grad_mag < threshold

    # 目标附近不阻挡
    dist = np.hypot(xs - task.target_x, ys - task.target_y)
    blocked &= dist > 5.0

    task.blocked = blocked
    return blocked.mean()


def run(method, seed=0, ticks=800, n_agents=100):
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)
    shape_potential(t, amplitude=100.0, length_scale=15.0)

    density = 0.0
    if method == "aligned":
        density = make_aligned_boundary(t, threshold=0.5)
    elif method == "aligned_strict":
        density = make_aligned_boundary(t, threshold=2.0)

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return t.aggregation_score(a), density


print("=== 对齐约束 vs 冲突约束 ===")
print(f"{'方法':>16} {'边界密度':>10} {'聚集度':>10}")
print("-" * 40)

for method in ["none", "aligned", "aligned_strict"]:
    scores, densities = [], []
    for seed in range(5):
        s, d = run(method, seed=seed)
        scores.append(s)
        densities.append(d)
    print(f"{method:>16} {np.mean(densities):>9.1%} {np.mean(scores):>10.3f}")

print()
print("=== 对齐约束的阈值扫描 ===")
print(f"{'梯度阈值':>10} {'边界密度':>10} {'聚集度':>10}")
print("-" * 32)
for th in [0.1, 0.5, 1.0, 2.0, 5.0]:
    scores, densities = [], []
    for seed in range(5):
        w = make_aggregation_world(60, 60)
        t = AggregationTask(w, target_x=30.0, target_y=30.0)
        shape_potential(t, amplitude=100.0, length_scale=15.0)
        d = make_aligned_boundary(t, threshold=th)
        a = AgentState(100)
        a.place_random(60.0, 60.0, np.random.default_rng(seed))
        e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
        e.run(800)
        scores.append(t.aggregation_score(a))
        densities.append(d)
    print(f"{th:>10.1f} {np.mean(densities):>9.1%} {np.mean(scores):>10.3f}")
