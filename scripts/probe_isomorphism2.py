"""同构原则的决定性实验：边界是「替代品」还是「补充品」？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask, make_aggregation_stepper, make_aggregation_world,
    shape_boundary, shape_feedback, shape_potential,
)


def run(method, exploration=0.0, seed=0, ticks=800, n_agents=100):
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)

    if method == "baseline":
        w.clear()
    elif method == "potential":
        shape_potential(t, amplitude=100.0, length_scale=15.0)
    elif method == "boundary_only":
        shape_boundary(t, n_rings=3, ring_gap=6.0, gap_ratio=0.15)
    elif method == "potential+boundary":
        shape_potential(t, amplitude=100.0, length_scale=15.0)
        # 保留势能场，叠加边界
        saved = w.data[0].copy()
        shape_boundary(t, n_rings=3, ring_gap=6.0, gap_ratio=0.15)
        w.data[0] = saved
    elif method == "feedback+boundary":
        shape_feedback(t, base_level=1.0, gradient_strength=10.0)
        saved = w.data[0].copy()
        shape_boundary(t, n_rings=3, ring_gap=6.0, gap_ratio=0.15)
        w.data[0] = saved

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))

    stepper = make_aggregation_stepper(t, exploration=exploration)
    e = Engine(w, a, seed=seed, stepper=stepper, hash_every=0)
    e.run(ticks)

    return t.aggregation_score(a), t.mean_distance(a)


methods = ["baseline", "boundary_only", "potential", "potential+boundary", "feedback+boundary"]

print("=== 边界是替代品还是补充品？===")
print()
print(f"{'方法':>20} {'聚集度':>10} {'平均距离':>10}")
print("-" * 45)
for m in methods:
    s, d = run(m)
    print(f"{m:>20} {s:>10.3f} {d:>10.2f}")

print()
print("=== 探索能力的影响（potential+boundary）===")
print(f"{'探索强度':>8} {'聚集度':>10} {'平均距离':>10}")
print("-" * 32)
for expl in [0.0, 0.05, 0.2, 0.5]:
    s, d = run("potential+boundary", exploration=expl)
    print(f"{expl:>8.2f} {s:>10.3f} {d:>10.2f}")
