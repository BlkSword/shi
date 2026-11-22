"""检查同质 vs 异质群体的实际均衡度."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.allocation import AllocationTask, make_allocation_stepper, make_allocation_world


def run(sigma, seed=0, ticks=800):
    w = make_allocation_world(40, 40, 3)
    t = AllocationTask(w, n_types=3, sites_per_type=5, rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(80)
    a.place_random(40.0, 40.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(80, sigma=sigma, rng=np.random.default_rng(seed + 2), n_types=3))
    e = Engine(w, a, seed=seed, stepper=make_allocation_stepper(t), hash_every=0)
    e.run(ticks)
    return t.balance(), t.work_done, t.work_by_type.tolist()


for sigma in [0.0, 0.1, 0.3, 0.5, 1.0]:
    bals, works = [], []
    for seed in range(5):
        b, wd, by = run(sigma, seed)
        bals.append(b)
        works.append(wd)
    print(f"sigma={sigma:.1f}  balance={np.mean(bals):.3f}+-{np.std(bals):.3f}  work={np.mean(works):.1f}")
