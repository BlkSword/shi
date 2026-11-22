"""诊断：资源密度对刺激分布的影响."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.engine import Engine
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.foraging import FORAGING_CHANNELS, ForagingTask, make_foraging_stepper


def probe(n_resources: int, cluster: bool, ticks: int = 600, seed: int = 0) -> dict:
    channels = [
        ChannelSpec(n, decay=dc, diffusion=df)
        for n, (dc, df) in FORAGING_CHANNELS.items()
    ]
    world = PheromoneField(60, 60, channels=channels)
    task = ForagingTask(
        world, 30.0, 30.0, n_resources=n_resources,
        rng=np.random.default_rng(seed),
    )
    if cluster:
        task.setup_clustered(n_clusters=8, per_cluster=n_resources // 8, spread=3.0)
    else:
        task.setup()

    agents = AgentState(150)
    agents.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    agents.set_thresholds(
        sample_thresholds(150, median=1.0, sigma=0.5, rng=np.random.default_rng(seed + 2))
    )

    stepper = make_foraging_stepper(task, recruit_amount=2.0)
    engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=0)
    engine.run(ticks)

    supply = np.array([
        world.sense(agents.x[i], agents.y[i], "supply", 2, normalize=True)
        for i in range(agents.n)
    ])
    p_seek = response_probability(supply, agents.threshold)

    return {
        "n_resources": n_resources,
        "cluster": cluster,
        "supply_mean": float(supply.mean()),
        "supply_nonzero": int(np.count_nonzero(supply)),
        "p_seek_mid": float(np.mean((p_seek > 0.05) & (p_seek < 0.95))),
        "coverage": task.coverage(),
        "collected": task.collected,
        "delivered": task.delivered,
    }


def main() -> None:
    print(f"{'资源数':>8} {'成簇':>6} {'supply均值':>11} {'非零个体':>9} {'p_seek过渡':>11} {'覆盖率':>9}")
    print("-" * 70)
    for n in [80, 200, 500, 1000]:
        for cluster in [False, True]:
            r = probe(n, cluster)
            print(
                f"{r['n_resources']:>8} {str(r['cluster']):>6} "
                f"{r['supply_mean']:>11.3f} {r['supply_nonzero']:>9} "
                f"{r['p_seek_mid']:>10.1%} {r['coverage']:>8.2%}"
            )


if __name__ == "__main__":
    main()
