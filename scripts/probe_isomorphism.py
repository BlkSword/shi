"""同构原则的关键实验：给个体加探索能力，边界条件是否变有效？"""
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


def run(method, exploration, seed=0, ticks=800, n_agents=100):
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

    stepper = make_aggregation_stepper(t, exploration=exploration)
    e = Engine(w, a, seed=seed, stepper=stepper, hash_every=0)
    e.run(ticks)

    return t.aggregation_score(a)


print("=== 同构原则实验：探索能力 x 塑造方法 ===")
print()
print(f"{'探索强度':>8} {'baseline':>10} {'potential':>11} {'boundary':>10} {'feedback':>10}")
print("-" * 55)
for expl in [0.0, 0.01, 0.05, 0.2, 0.5]:
    row = f"{expl:>8.2f}"
    for m in ["baseline", "potential", "boundary", "feedback"]:
        s = run(m, expl)
        row += f" {s:>10.3f}"
    print(row)

print()
print("=== 关键判据 ===")
print("如果 boundary 随 exploration 上升而改善 -> 同构原则成立")
print("如果 boundary 始终不变 -> 同构原则需要修正")
