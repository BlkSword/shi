"""诊断：刺激强度的实际分布 vs 阈值分布.

如果刺激强度的动态范围远大于阈值，阈值就形同虚设——
因为响应概率要么接近 0 要么接近 1，中间没有过渡区。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.foraging import FORAGING_CHANNELS, ForagingTask, make_foraging_stepper


def main() -> None:
    channels = [
        ChannelSpec(n, decay=dc, diffusion=df)
        for n, (dc, df) in FORAGING_CHANNELS.items()
    ]
    world = PheromoneField(60, 60, channels=channels)
    task = ForagingTask(world, 30.0, 30.0, n_resources=80, rng=np.random.default_rng(0))
    task.setup()

    agents = AgentState(150)
    agents.place_random(60.0, 60.0, np.random.default_rng(1))
    agents.set_thresholds(
        sample_thresholds(150, median=1.0, sigma=0.5, rng=np.random.default_rng(2))
    )

    stepper = make_foraging_stepper(task)

    print("=== 阈值分布 ===")
    print(
        f"  median={np.median(agents.threshold):.3f}  "
        f"mean={agents.threshold.mean():.3f}  "
        f"min={agents.threshold.min():.3f}  max={agents.threshold.max():.3f}"
    )
    print()

    print("=== 刺激强度分布（归一化邻域均值，半径 2）===")
    prev = 0
    for t in [0, 50, 200, 500]:
        for _ in range(t - prev):
            stepper(agents, world, 0, np.random.default_rng(0))
            world.step()
        prev = t

        supply = np.array([
            world.sense(agents.x[i], agents.y[i], "supply", 2, normalize=True)
            for i in range(agents.n)
        ])
        order = np.array([
            world.sense(agents.x[i], agents.y[i], "order", 2, normalize=True)
            for i in range(agents.n)
        ])
        recruit = np.array([
            world.sense(agents.x[i], agents.y[i], "recruit", 2, normalize=True)
            for i in range(agents.n)
        ])

        print(f"  tick={t}:")
        for name, arr in [("supply", supply), ("order", order), ("recruit", recruit)]:
            print(
                f"    {name:<8} mean={arr.mean():8.4f} median={np.median(arr):8.4f} "
                f"max={arr.max():8.4f} nonzero={np.count_nonzero(arr):>3}/{len(arr)}"
            )

        p_seek = response_probability(supply, agents.threshold)
        p_ret = response_probability(order, agents.threshold)
        mid_seek = np.mean((p_seek > 0.05) & (p_seek < 0.95))
        mid_ret = np.mean((p_ret > 0.05) & (p_ret < 0.95))
        print(f"    p_seek   mean={p_seek.mean():.4f}  过渡区比例={mid_seek:.1%}")
        print(f"    p_return mean={p_ret.mean():.4f}  过渡区比例={mid_ret:.1%}")
        print(f"    已采集={task.collected}  已交付={task.delivered}  覆盖率={task.coverage():.2%}")
        print()


if __name__ == "__main__":
    main()
