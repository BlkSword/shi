"""调参：让任务足够难，使环境容量成为真正的瓶颈。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask, make_delayed_foraging_stepper, make_delayed_foraging_world,
)


def run(n_channels, n_resources=40, cycle=300, scout_ratio=0.3, n_types=4,
        n_agents=100, ticks=3000, seed=0):
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


print("=== 调参：让任务变难 ===")
print()
print("配置 A: 资源 100, 周期 200, 侦察 20%, 类型 6")
print(f"{'通道':>5} {'采集':>6} {'效率':>8} {'错误':>9}")
for n_ch in [1, 2, 3, 4, 6, 8]:
    c, eff, w = run(n_ch, n_resources=100, cycle=200, scout_ratio=0.2, n_types=6)
    print(f"{n_ch:>5} {c:>6} {eff:>7.1%} {w:>9}")

print()
print("配置 B: 资源 200, 周期 150, 侦察 15%, 类型 8")
print(f"{'通道':>5} {'采集':>6} {'效率':>8} {'错误':>9}")
for n_ch in [1, 2, 3, 4, 6, 8]:
    c, eff, w = run(n_ch, n_resources=200, cycle=150, scout_ratio=0.15, n_types=8)
    print(f"{n_ch:>5} {c:>6} {eff:>7.1%} {w:>9}")

print()
print("配置 C: 资源 300, 周期 120, 侦察 10%, 类型 8, 个体 50")
print(f"{'通道':>5} {'采集':>6} {'效率':>8} {'错误':>9}")
for n_ch in [1, 2, 3, 4, 6, 8]:
    c, eff, w = run(n_ch, n_resources=300, cycle=120, scout_ratio=0.1, n_types=8, n_agents=50)
    print(f"{n_ch:>5} {c:>6} {eff:>7.1%} {w:>9}")
