"""G2 不可能性边界：什么样的目标无法注入？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask, make_aggregation_stepper, make_aggregation_world,
    shape_feedback, shape_potential,
)


def run_target(target_x, target_y, method="potential", seed=0, ticks=800, n_agents=100):
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=target_x, target_y=target_y)

    if method == "potential":
        shape_potential(t, amplitude=100.0, length_scale=15.0)
    elif method == "feedback":
        shape_feedback(t, base_level=1.0, gradient_strength=10.0)

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return t.mean_distance(a), t.aggregation_score(a)


print("=== 不可能性测试 1：目标在边界上 ===")
print("假设：目标在边界时，势能场被截断，梯度不对称，可能无法聚集")
print(f"{'目标位置':>16} {'最终距离':>10} {'聚集度':>9}")
print("-" * 40)
for name, (tx, ty) in [
    ("中心 (30,30)", (30.0, 30.0)),
    ("边缘 (0,30)", (0.0, 30.0)),
    ("角落 (0,0)", (0.0, 0.0)),
    ("近角 (2,2)", (2.0, 2.0)),
]:
    d, s = run_target(tx, ty)
    print(f"{name:>16} {d:>10.2f} {s:>9.3f}")

print()
print("=== 不可能性测试 2：目标在环境外 ===")
print("假设：目标在网格外时，个体被边界钳制，无法到达")
print(f"{'目标位置':>16} {'最终距离':>10} {'聚集度':>9}")
print("-" * 40)
for name, (tx, ty) in [
    ("内部 (30,30)", (30.0, 30.0)),
    ("外部 (-10,30)", (-10.0, 30.0)),
    ("远处 (100,100)", (100.0, 100.0)),
]:
    d, s = run_target(tx, ty)
    print(f"{name:>16} {d:>10.2f} {s:>9.3f}")

print()
print("=== 不可能性测试 3：多个等强吸引子 ===")
print("假设：两个等强吸引子会导致系统分裂，无法聚集到单一目标")

def run_two_attractors(sep, seed=0, ticks=800, n_agents=100):
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)
    w.clear()
    ys, xs = np.mgrid[0:60, 0:60]
    # 两个吸引子
    d1 = np.hypot(xs - (30 - sep/2), ys - 30)
    d2 = np.hypot(xs - (30 + sep/2), ys - 30)
    w.data[0] = 100.0 * np.exp(-(d1**2)/(2*8**2)) + 100.0 * np.exp(-(d2**2)/(2*8**2))

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    # 统计有多少个体在左吸引子、多少在右
    left = np.sum(a.x < 30)
    right = np.sum(a.x >= 30)
    return t.mean_distance(a), t.aggregation_score(a), left, right

print(f"{'吸引子间距':>10} {'最终距离':>10} {'聚集度':>9} {'左/右':>10}")
print("-" * 45)
for sep in [0, 5, 10, 20, 30]:
    d, s, l, r = run_two_attractors(sep)
    print(f"{sep:>10} {d:>10.2f} {s:>9.3f} {l:>5}/{r:<5}")
