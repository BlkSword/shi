"""标定 demand：找到让活跃比例落在合理区间（30%-80%）的值."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.allocation import AllocationTask, make_allocation_stepper, make_allocation_world


def run(demand, sigma=0.5, seed=0, ticks=800):
    w = make_allocation_world(60, 60, 3)
    t = AllocationTask(w, n_types=3, sites_per_type=6, demand=demand, regen=0.05,
                       rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(150)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(150, sigma=sigma,
                                      rng=np.random.default_rng(seed + 2), n_types=3))
    e = Engine(w, a, seed=seed, stepper=make_allocation_stepper(t), hash_every=0)

    active_series = []
    for i in range(ticks):
        e.step()
        if i % 50 == 0:
            active_series.append(np.count_nonzero(a.state > 0))

    tail = active_series[len(active_series)*3//4:]
    return {
        "demand": demand,
        "active_ratio": float(np.mean(tail)) / 150,
        "work": t.work_done,
        "balance": t.balance(),
    }


print(f"{'demand':>8} {'活跃比例':>10} {'总产出':>10} {'均衡度':>9}")
print("-" * 45)
for demand in [20, 40, 60, 80, 120, 200]:
    r = run(demand)
    print(f"{r['demand']:>8} {r['active_ratio']:>9.1%} {r['work']:>10.1f} {r['balance']:>9.3f}")
