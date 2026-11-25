"""G2 快速验证：三个环境塑造方法能否把个体导向目标？"""
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


def run(method, seed=0, ticks=800, n_agents=100):
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)

    if method == "baseline":
        w.clear()
    elif method == "potential":
        shape_potential(t, amplitude=100.0, length_scale=15.0)
    elif method == "boundary":
        shape_boundary(t, n_rings=3, ring_gap=6.0, gap_ratio=0.15)
    elif method == "feedback":
        shape_feedback(t, base_level=1.0, gradient_strength=10.0)

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))

    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)

    d0 = t.mean_distance(a)
    e.run(ticks)
    d1 = t.mean_distance(a)

    return {
        "method": method,
        "d_initial": d0,
        "d_final": d1,
        "score": t.aggregation_score(a),
        "improvement": (d0 - d1) / d0 if d0 > 0 else 0.0,
    }


print("=== G2 快速验证：三个方法对比 ===")
print(f"{'方法':>12} {'初始距离':>10} {'最终距离':>10} {'聚集度':>9} {'改善':>8}")
print("-" * 60)
for m in ["baseline", "potential", "boundary", "feedback"]:
    r = run(m)
    print(f"{r['method']:>12} {r['d_initial']:>10.2f} {r['d_final']:>10.2f} "
          f"{r['score']:>9.3f} {r['improvement']:>7.1%}")
