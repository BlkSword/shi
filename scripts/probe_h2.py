"""验证：通道数不足导致类型混淆，性能下降。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask, make_delayed_foraging_stepper, make_delayed_foraging_world,
)


def run(n_channels, bits=8, budget=1.0, seed=0, ticks=3000, n_agents=100, n_types=4):
    w = make_delayed_foraging_world(60, 60, n_channels=n_channels, precision_bits=bits)
    t = DelayedForagingTask(w, n_resources=40, n_types=n_types, cycle=300,
                            scout_ratio=0.3, write_budget=budget,
                            rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(n_agents, sigma=0.5,
                                      rng=np.random.default_rng(seed + 2)))
    e = Engine(w, a, seed=seed, stepper=make_delayed_foraging_stepper(t), hash_every=0)
    e.run(ticks)
    return t.collected, t.efficiency(), t.wrong_visits


print("=== 通道数 vs 性能（4 种资源类型）===")
print(f"{'通道':>5} {'采集':>6} {'效率':>8} {'错误访问':>9}")
print("-" * 35)
for n_ch in [1, 2, 3, 4, 6, 8]:
    c, eff, wrong = run(n_ch)
    print(f"{n_ch:>5} {c:>6} {eff:>7.1%} {wrong:>9}")

print()
print("=== 写入带宽 vs 性能（通道=4）===")
print(f"{'带宽':>5} {'采集':>6} {'效率':>8}")
print("-" * 25)
for b in [0.05, 0.2, 0.5, 1.0, 2.0]:
    c, eff, _ = run(4, budget=b)
    print(f"{b:>5.2f} {c:>6} {eff:>7.1%}")
