"""快速验证：多任务分配任务中 sigma 是否有影响."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.allocation import AllocationTask, make_allocation_stepper, make_allocation_world


def run(sigma: float, seed: int = 0, ticks: int = 1500, n_types: int = 3) -> dict:
    world = make_allocation_world(60, 60, n_types)
    task = AllocationTask(
        world, n_types=n_types, sites_per_type=6, demand=20.0, regen=0.05,
        rng=np.random.default_rng(seed),
    )
    task.setup()

    agents = AgentState(150)
    agents.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    agents.set_thresholds(
        sample_thresholds(
            150, median=1.0, sigma=sigma,
            rng=np.random.default_rng(seed + 2), n_types=n_types,
        )
    )

    stepper = make_allocation_stepper(task)
    engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=0)
    engine.run(ticks)

    return {
        "sigma": sigma,
        "work_done": task.work_done,
        "completion": task.completion(),
        "balance": task.balance(),
        "by_type": task.work_by_type.tolist(),
    }


def main() -> None:
    print(f"{'sigma':>6} {'总工作量':>10} {'完成度':>9} {'均衡度':>9}  各类型工作量")
    print("-" * 75)
    for sigma in [0.0, 0.1, 0.3, 0.6, 1.0, 1.5]:
        r = run(sigma)
        by = " ".join(f"{v:7.1f}" for v in r["by_type"])
        print(
            f"{r['sigma']:>6.1f} {r['work_done']:>10.1f} "
            f"{r['completion']:>8.2%} {r['balance']:>9.3f}  {by}"
        )


if __name__ == "__main__":
    main()
