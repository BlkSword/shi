"""最终调参：缩短时程，让效率也分化。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask, make_delayed_foraging_stepper, make_delayed_foraging_world,
)


def run(n_channels, ticks, n_resources=200, cycle=150, scout_ratio=0.15,
        n_types=8, n_agents=60, seed=0):
    w = make_delayed_foraging_world(60, 60, n_channels=n_channels)
    t = DelayedForagingTask(w, n_resources=n_resources, n_types=n_types,
                            cycle=cycle, scout_ratio=scout_ratio,
                            rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(n_agents, sigma=0.5,
                                      rng=np.random.default_rng(seed + 2)))
    e = Engine(w, a, seed=seed, stepper=make_delayed_foraging_stepper(t), hash_every=0)
    e.run(ticks)
    return t.collected, t.efficiency(), t.wrong_visits


for ticks in [600, 1000, 1500]:
    print(f"=== ticks={ticks} ===")
    print(f"{'通道':>5} {'采集':>6} {'效率':>8} {'错误':>9} {'错误/采集':>10}")
    for n_ch in [1, 2, 3, 4, 6, 8]:
        c, eff, w = run(n_ch, ticks)
        ratio = w / max(c, 1)
        print(f"{n_ch:>5} {c:>6} {eff:>7.1%} {w:>9} {ratio:>10.1f}")
    print()
