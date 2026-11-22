"""标定 recruit 通道：找到能让 p_recruit 落在过渡区的参数."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.engine import Engine
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.foraging import FORAGING_CHANNELS, ForagingTask, make_foraging_stepper


def probe(recruit_amount: float, ticks: int = 800, seed: int = 0) -> dict:
    channels = [
        ChannelSpec(n, decay=dc, diffusion=df)
        for n, (dc, df) in FORAGING_CHANNELS.items()
    ]
    world = PheromoneField(60, 60, channels=channels)
    task = ForagingTask(world, 30.0, 30.0, n_resources=80, rng=np.random.default_rng(seed))
    task.setup()

    agents = AgentState(150)
    agents.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    agents.set_thresholds(
        sample_thresholds(150, median=1.0, sigma=0.5, rng=np.random.default_rng(seed + 2))
    )

    stepper = make_foraging_stepper(task, recruit_amount=recruit_amount)
    engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=0)
    engine.run(ticks)

    recruit = np.array([
        world.sense(agents.x[i], agents.y[i], "recruit", 2, normalize=True)
        for i in range(agents.n)
    ])
    supply = np.array([
        world.sense(agents.x[i], agents.y[i], "supply", 2, normalize=True)
        for i in range(agents.n)
    ])

    p_rec = response_probability(recruit, agents.threshold)
    p_seek = response_probability(supply, agents.threshold)

    return {
        "recruit_amount": recruit_amount,
        "recruit_mean": float(recruit.mean()),
        "recruit_max": float(recruit.max()),
        "p_recruit_mean": float(p_rec.mean()),
        "p_recruit_mid": float(np.mean((p_rec > 0.05) & (p_rec < 0.95))),
        "p_seek_mid": float(np.mean((p_seek > 0.05) & (p_seek < 0.95))),
        "coverage": task.coverage(),
        "collected": task.collected,
        "delivered": task.delivered,
    }


def main() -> None:
    print(f"{'amount':>8} {'rec_mean':>9} {'rec_max':>9} {'p_rec':>8} {'p_rec过渡':>10} {'p_seek过渡':>11} {'覆盖率':>9}")
    print("-" * 80)
    for amount in [0.05, 0.2, 0.5, 1.0, 2.0, 5.0]:
        r = probe(amount)
        print(
            f"{r['recruit_amount']:>8.2f} {r['recruit_mean']:>9.3f} {r['recruit_max']:>9.3f} "
            f"{r['p_recruit_mean']:>8.3f} {r['p_recruit_mid']:>9.1%} "
            f"{r['p_seek_mid']:>10.1%} {r['coverage']:>8.2%}"
        )


if __name__ == "__main__":
    main()
