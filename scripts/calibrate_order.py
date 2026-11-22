"""标定 order 通道：找到能让 p_return 落在过渡区的参数.

目标：order 通道在巢穴附近 ~5-20，在远处 ~0.1-1，
使得阈值中位数 1.0 的个体在靠近巢穴时响应概率显著上升。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.world import ChannelSpec, PheromoneField


def probe(deposit: float, decay: float, diffusion: float, ticks: int = 300) -> dict:
    world = PheromoneField(
        60, 60,
        channels=[ChannelSpec("order", decay=decay, diffusion=diffusion)],
    )
    nest = (30.0, 30.0)
    for _ in range(ticks):
        world.deposit(nest[0], nest[1], "order", deposit)
        world.step()

    # 采样不同距离处的浓度
    samples = {}
    for dist in [0, 2, 5, 10, 20, 30]:
        x = min(59.0, nest[0] + dist)
        samples[dist] = world.read(x, nest[1], "order")
    return samples


def main() -> None:
    print(f"{'deposit':>8} {'decay':>7} {'diff':>6} | " +
          " ".join(f"d={d:<6}" for d in [0, 2, 5, 10, 20, 30]))
    print("-" * 90)

    for deposit in [2.0, 10.0, 50.0, 200.0]:
        for decay, diffusion in [(0.99, 0.20), (0.995, 0.10), (0.999, 0.05)]:
            s = probe(deposit, decay, diffusion)
            row = " ".join(f"{s[d]:<8.3f}" for d in [0, 2, 5, 10, 20, 30])
            print(f"{deposit:>8.1f} {decay:>7.3f} {diffusion:>6.2f} | {row}")


if __name__ == "__main__":
    main()
