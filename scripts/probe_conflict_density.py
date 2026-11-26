"""冲突的量化边界：边界密度多大时开始破坏梯度？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask, make_aggregation_stepper, make_aggregation_world,
    shape_boundary, shape_potential,
)


def run(n_rings, gap_ratio, seed=0, ticks=800, n_agents=100):
    """势能场 + 指定密度的边界。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)

    # 先放势能场
    shape_potential(t, amplitude=100.0, length_scale=15.0)
    saved = w.data[0].copy()

    # 再叠加边界（shape_boundary 会 clear，所以要恢复）
    if n_rings > 0:
        shape_boundary(t, n_rings=n_rings, ring_gap=6.0, gap_ratio=gap_ratio,
                       rng=np.random.default_rng(seed))
        w.data[0] = saved
    else:
        t.blocked = None

    density = t.blocked.mean() if t.blocked is not None else 0.0

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return t.aggregation_score(a), density


print("=== 边界密度 vs 性能（势能场 + 边界）===")
print(f"{'环数':>5} {'缺口比例':>9} {'边界密度':>10} {'聚集度':>10}")
print("-" * 42)

# 扫描环数（密度随环数增加）
for n_rings in [0, 1, 2, 3, 4, 5, 6]:
    scores, densities = [], []
    for seed in range(5):
        s, d = run(n_rings, 0.15, seed=seed)
        scores.append(s)
        densities.append(d)
    print(f"{n_rings:>5} {0.15:>9.2f} {np.mean(densities):>9.1%} {np.mean(scores):>10.3f}")

print()
print("=== 缺口比例 vs 性能（固定 3 环）===")
print(f"{'缺口比例':>9} {'边界密度':>10} {'聚集度':>10}")
print("-" * 32)
for gap in [0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]:
    scores, densities = [], []
    for seed in range(5):
        s, d = run(3, gap, seed=seed)
        scores.append(s)
        densities.append(d)
    print(f"{gap:>9.2f} {np.mean(densities):>9.1%} {np.mean(scores):>10.3f}")
