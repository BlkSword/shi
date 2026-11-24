"""验证：规则复杂度是否影响性能？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask, make_delayed_foraging_stepper, make_delayed_foraging_world,
)


def run(n_channels, complexity, ticks=1000, seed=0):
    w = make_delayed_foraging_world(60, 60, n_channels=n_channels)
    t = DelayedForagingTask(w, n_resources=200, n_types=8, cycle=150,
                            scout_ratio=0.15, rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(60)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(60, sigma=0.5,
                                      rng=np.random.default_rng(seed + 2)))
    stepper = make_delayed_foraging_stepper(t, rule_complexity=complexity)
    e = Engine(w, a, seed=seed, stepper=stepper, hash_every=0)
    e.run(ticks)
    return t.collected, t.efficiency(), t.wrong_visits


print("=== 规则复杂度 vs 性能（通道=4）===")
print(f"{'复杂度':>7} {'采集':>6} {'效率':>8} {'错误':>9}")
for c in [1, 2, 3]:
    col, eff, wr = run(4, c)
    print(f"{c:>7} {col:>6} {eff:>7.1%} {wr:>9}")

print()
print("=== 规则复杂度 vs 性能（通道=1，容量不足）===")
print(f"{'复杂度':>7} {'采集':>6} {'效率':>8} {'错误':>9}")
for c in [1, 2, 3]:
    col, eff, wr = run(1, c)
    print(f"{c:>7} {col:>6} {eff:>7.1%} {wr:>9}")
