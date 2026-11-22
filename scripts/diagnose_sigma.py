"""诊断：为什么异质性 sigma 对结果没有影响？

假设：阈值响应机制实际上没有在「做决策」——
因为 p_seek 和 p_return 要么接近 0 要么接近 1，
阈值只是整体缩放，不改变相对排序。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.engine import Engine
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.foraging import FORAGING_CHANNELS, ForagingTask, make_foraging_stepper


def run(sigma: float, seed: int = 0, ticks: int = 1500):
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
        sample_thresholds(150, median=1.0, sigma=sigma, rng=np.random.default_rng(seed + 2))
    )

    stepper = make_foraging_stepper(task)
    engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=0)

    # 记录状态分布随时间的变化
    state_history = []
    for t in range(ticks):
        engine.step()
        if t % 300 == 0:
            state_history.append((t, agents.count_by_state()))

    return {
        "sigma": sigma,
        "coverage": task.coverage(),
        "collected": task.collected,
        "delivered": task.delivered,
        "delivery_rate": task.delivery_rate(),
        "state_history": state_history,
        "threshold_std": float(agents.threshold.std()),
    }


def main() -> None:
    print("=== 不同 sigma 下的行为差异 ===")
    print(f"{'sigma':>6} {'覆盖率':>9} {'采集':>6} {'交付':>6} {'交付率':>8} {'阈值std':>9}")
    print("-" * 55)
    for sigma in [0.0, 0.2, 0.5, 1.0]:
        r = run(sigma)
        print(
            f"{sigma:>6.1f} {r['coverage']:>8.2%} {r['collected']:>6} "
            f"{r['delivered']:>6} {r['delivery_rate']:>7.1%} {r['threshold_std']:>9.3f}"
        )

    print()
    print("=== 状态分布演化（sigma=0.0 vs sigma=1.0）===")
    for sigma in [0.0, 1.0]:
        r = run(sigma)
        print(f"\nsigma={sigma}:")
        for t, counts in r["state_history"]:
            print(f"  t={t:>5}: {counts}")


if __name__ == "__main__":
    main()
