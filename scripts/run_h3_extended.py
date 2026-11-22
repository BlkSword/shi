"""H3 扩展实验：sigma 到 5.0，并诊断高 sigma 的机制.

三个问题
----------
Q1: 总产出是否继续上升？是否最终出现下降（真正的倒 U 形）？
Q2: 高 sigma 的收益来自哪里？分工更细，还是阈值高的个体自动退出？
Q3: 活跃个体数随 sigma 如何变化？

用法
----
    python scripts/run_h3_extended.py
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
from shi.tasks.allocation import (
    AllocationTask,
    make_allocation_stepper,
    make_allocation_world,
)


def run_one(sigma: float, seed: int, ticks: int, n_agents: int, n_types: int) -> dict:
    """跑一次，记录详细的机制指标。"""
    world = make_allocation_world(60, 60, n_types)
    task = AllocationTask(
        world, n_types=n_types, sites_per_type=6, demand=60.0, regen=0.05,
        rng=np.random.default_rng(seed),
    )
    task.setup()

    agents = AgentState(n_agents)
    agents.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    agents.set_thresholds(
        sample_thresholds(
            n_agents, median=1.0, sigma=sigma,
            rng=np.random.default_rng(seed + 2), n_types=n_types,
        )
    )

    stepper = make_allocation_stepper(task)
    engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=0)

    # 记录活跃个体数的时间序列
    active_series = []
    for t in range(ticks):
        engine.step()
        if t % 100 == 0:
            active_series.append(int(np.count_nonzero(agents.state > 0)))

    # 稳态时的活跃个体数（最后 20%）
    tail = active_series[len(active_series) * 4 // 5 :]
    active_steady = float(np.mean(tail)) if tail else 0.0

    # 各类型工作量分布
    by_type = task.work_by_type.copy()
    total = by_type.sum()
    type_share = (by_type / total).tolist() if total > 0 else [0.0] * n_types

    # 阈值分布统计
    th = agents.threshold
    th_flat = th.ravel()

    return {
        "sigma": sigma,
        "seed": seed,
        "work_done": float(task.work_done),
        "balance": float(task.balance()),
        "active_steady": active_steady,
        "active_ratio": active_steady / n_agents,
        "type_share": type_share,
        "threshold_mean": float(th_flat.mean()),
        "threshold_std": float(th_flat.std()),
        "threshold_min": float(th_flat.min()),
        "threshold_max": float(th_flat.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--agents", type=int, default=150)
    ap.add_argument("--types", type=int, default=3)
    ap.add_argument("--out", type=str, default="experiments/h3")
    args = ap.parse_args()

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(sigmas) * args.repeats
    done = 0
    for sigma in sigmas:
        for rep in range(args.repeats):
            results.append(run_one(sigma, rep * 1000, args.ticks, args.agents, args.types))
            done += 1
            print(f"\r  进度: {done}/{total}", end="", flush=True)
    print()
    print()

    # 汇总
    print(f"{'sigma':>6} {'总产出':>16} {'均衡度':>16} {'活跃比例':>10} {'阈值std':>9}")
    print("-" * 70)
    stats = {}
    for sigma in sigmas:
        rs = [r for r in results if r["sigma"] == sigma]
        work = [r["work_done"] for r in rs]
        bal = [r["balance"] for r in rs]
        act = [r["active_ratio"] for r in rs]
        thstd = [r["threshold_std"] for r in rs]
        stats[sigma] = {
            "work_mean": float(np.mean(work)),
            "work_std": float(np.std(work, ddof=1)),
            "balance_mean": float(np.mean(bal)),
            "balance_std": float(np.std(bal, ddof=1)),
            "active_mean": float(np.mean(act)),
            "threshold_std_mean": float(np.mean(thstd)),
        }
        s = stats[sigma]
        print(
            f"{sigma:>6.1f} {s['work_mean']:>9.1f} +- {s['work_std']:>5.1f} "
            f"{s['balance_mean']:>9.3f} +- {s['balance_std']:>5.3f} "
            f"{s['active_mean']:>9.1%} {s['threshold_std_mean']:>9.3f}"
        )

    # 各类型工作量分布（看是否出现「某些类型被放弃」）
    print()
    print("各类型工作量占比（均值）:")
    print(f"{'sigma':>6}  " + "  ".join(f"type{k}" for k in range(args.types)))
    print("-" * 50)
    for sigma in sigmas:
        rs = [r for r in results if r["sigma"] == sigma]
        shares = np.array([r["type_share"] for r in rs])
        mean_share = shares.mean(axis=0)
        print(f"{sigma:>6.1f}  " + "  ".join(f"{v:>5.1%}" for v in mean_share))

    # 保存
    (out_dir / "e1_extended.json").write_text(
        json.dumps({"stats": {str(k): v for k, v in stats.items()}, "raw": results}, indent=2),
        encoding="utf-8",
    )

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)

        ks = sigmas
        ax = axes[0]
        ax.errorbar(ks, [stats[k]["work_mean"] for k in ks],
                    yerr=[stats[k]["work_std"] for k in ks],
                    marker="o", capsize=4, color="#b45309")
        ax.set_xlabel("threshold sigma")
        ax.set_ylabel("total work done")
        ax.set_title("Productivity vs heterogeneity")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.errorbar(ks, [stats[k]["balance_mean"] for k in ks],
                    yerr=[stats[k]["balance_std"] for k in ks],
                    marker="s", capsize=4, color="#0f766e")
        ax.set_xlabel("threshold sigma")
        ax.set_ylabel("task balance (0-1)")
        ax.set_title("Division of labor vs heterogeneity")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(ks, [stats[k]["active_mean"] for k in ks],
                marker="^", color="#6d28d9")
        ax.set_xlabel("threshold sigma")
        ax.set_ylabel("active agent ratio")
        ax.set_title("Active agents vs heterogeneity")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "e1_extended.png")
        plt.close(fig)
        print(f"\n图已保存: {out_dir / 'e1_extended.png'}")
    except ImportError:
        print("matplotlib 未安装，跳过出图")


if __name__ == "__main__":
    main()
