"""H2 正式实验：环境可写性 vs 个体规则复杂度.

设计
----
双因素：
- 因子 A：环境信息容量（通道数 1/2/4/8）
- 因子 B：个体规则复杂度（1/2/3）

因变量：
- 采集效率（主指标）
- 错误访问率（信息保真度）

预期（H2 成立）：
- 环境容量主效应显著（效应量大）
- 个体复杂度主效应不显著（效应量小）
- 交互效应：容量不足时复杂度有用，容量充足时复杂度无用甚至有害

用法
----
    python scripts/run_h2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask,
    make_delayed_foraging_stepper,
    make_delayed_foraging_world,
)


def run_one(n_channels: int, complexity: int, seed: int, ticks: int, n_agents: int) -> dict:
    w = make_delayed_foraging_world(60, 60, n_channels=n_channels)
    t = DelayedForagingTask(
        w, n_resources=200, n_types=8, cycle=150, scout_ratio=0.15,
        rng=np.random.default_rng(seed),
    )
    t.setup()

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(
        sample_thresholds(n_agents, sigma=0.5, rng=np.random.default_rng(seed + 2))
    )

    stepper = make_delayed_foraging_stepper(t, rule_complexity=complexity)
    e = Engine(w, a, seed=seed, stepper=stepper, hash_every=0)
    e.run(ticks)

    return {
        "n_channels": n_channels,
        "complexity": complexity,
        "seed": seed,
        "collected": t.collected,
        "efficiency": t.efficiency(),
        "wrong_visits": t.wrong_visits,
        "error_ratio": t.wrong_visits / max(t.collected, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--agents", type=int, default=60)
    ap.add_argument("--out", type=str, default="experiments/h2")
    args = ap.parse_args()

    channels_list = [1, 2, 4, 8]
    complexities = [1, 2, 3]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(channels_list) * len(complexities) * args.repeats
    done = 0

    for n_ch in channels_list:
        for comp in complexities:
            for rep in range(args.repeats):
                results.append(
                    run_one(n_ch, comp, rep * 1000, args.ticks, args.agents)
                )
                done += 1
                print(f"\r  进度: {done}/{total}", end="", flush=True)
    print()
    print()

    # 汇总表
    print("=== 采集效率（均值 ± 标准差）===")
    header = f"{'通道\\复杂度':>12}" + "".join(f"{c:>18}" for c in complexities)
    print(header)
    print("-" * len(header))

    grid = {}
    for n_ch in channels_list:
        row = f"{n_ch:>12}"
        for comp in complexities:
            rs = [r for r in results if r["n_channels"] == n_ch and r["complexity"] == comp]
            effs = [r["efficiency"] for r in rs]
            mean, std = np.mean(effs), np.std(effs, ddof=1)
            grid[(n_ch, comp)] = {"mean": float(mean), "std": float(std), "n": len(rs)}
            row += f"{mean:>10.1%} ±{std:>5.1%}"
        print(row)

    print()
    print("=== 错误访问率（错误/采集，越小越好）===")
    header = f"{'通道\\复杂度':>12}" + "".join(f"{c:>18}" for c in complexities)
    print(header)
    print("-" * len(header))
    for n_ch in channels_list:
        row = f"{n_ch:>12}"
        for comp in complexities:
            rs = [r for r in results if r["n_channels"] == n_ch and r["complexity"] == comp]
            ratios = [r["error_ratio"] for r in rs]
            row += f"{np.mean(ratios):>10.1f} ±{np.std(ratios, ddof=1):>5.1f}"
        print(row)

    # 效应量分析
    print()
    print("=== 主效应分析 ===")

    # 环境容量主效应：比较通道 1 vs 8（跨所有复杂度）
    low_ch = [r["efficiency"] for r in results if r["n_channels"] == 1]
    high_ch = [r["efficiency"] for r in results if r["n_channels"] == 8]
    d_ch = cohens_d(high_ch, low_ch)
    print(f"环境容量效应（通道 8 vs 1）：d = {d_ch:.2f}")
    print(f"  通道1: {np.mean(low_ch):.1%}  通道8: {np.mean(high_ch):.1%}")

    # 个体复杂度主效应：比较复杂度 1 vs 3（跨所有通道）
    low_c = [r["efficiency"] for r in results if r["complexity"] == 1]
    high_c = [r["efficiency"] for r in results if r["complexity"] == 3]
    d_c = cohens_d(high_c, low_c)
    print(f"个体复杂度效应（复杂度 3 vs 1）：d = {d_c:.2f}")
    print(f"  复杂度1: {np.mean(low_c):.1%}  复杂度3: {np.mean(high_c):.1%}")

    print()
    print("=== H2 判定 ===")
    ratio = abs(d_ch) / max(abs(d_c), 1e-9)
    print(f"效应量比 |d_环境| / |d_个体| = {ratio:.2f}")
    if ratio > 3:
        print("  -> H2 成立：环境效应显著大于个体效应")
    elif ratio > 1:
        print("  -> H2 部分成立：环境效应更大但差距不足 3 倍")
    else:
        print("  -> H2 不成立：个体效应不小于环境效应")

    # 保存
    (out_dir / "h2_results.json").write_text(
        json.dumps({"grid": {f"{k[0]}_{k[1]}": v for k, v in grid.items()},
                    "raw": results,
                    "cohens_d_channel": d_ch,
                    "cohens_d_complexity": d_c}, indent=2),
        encoding="utf-8",
    )

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

        for comp in complexities:
            means = [grid[(n_ch, comp)]["mean"] for n_ch in channels_list]
            stds = [grid[(n_ch, comp)]["std"] for n_ch in channels_list]
            ax1.errorbar(channels_list, means, yerr=stds, marker="o", capsize=4,
                         label=f"complexity {comp}")
        ax1.set_xlabel("environment channels (information capacity)")
        ax1.set_ylabel("foraging efficiency")
        ax1.set_title("H2: Environment capacity vs agent complexity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for comp in complexities:
            means = [np.mean([r["error_ratio"] for r in results
                              if r["n_channels"] == n_ch and r["complexity"] == comp])
                     for n_ch in channels_list]
            ax2.plot(channels_list, means, marker="s", label=f"complexity {comp}")
        ax2.set_xlabel("environment channels")
        ax2.set_ylabel("error ratio (wrong/collected)")
        ax2.set_title("Information fidelity vs capacity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "h2_results.png")
        plt.close(fig)
        print(f"\n图已保存: {out_dir / 'h2_results.png'}")
    except ImportError:
        print("matplotlib 未安装")


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen d 效应量。"""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n_a, n_b = len(a_arr), len(b_arr)
    var_a = a_arr.var(ddof=1)
    var_b = b_arr.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled)


if __name__ == "__main__":
    main()
