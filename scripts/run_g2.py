"""G2 正式实验：目标注入与不可能性边界.

三个实验
----------
E1: 三个塑造方法对比（基线 / 势能 / 边界 / 反馈）
E2: 不可能性边界 1——目标在环境外
E3: 不可能性边界 2——多个等强吸引子

用法
----
    python scripts/run_g2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask,
    make_aggregation_stepper,
    make_aggregation_world,
    shape_boundary,
    shape_feedback,
    shape_potential,
)


def run_method(method: str, seed: int, ticks: int, n_agents: int) -> dict:
    """E1：跑一个塑造方法。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)

    if method == "baseline":
        w.clear()
    elif method == "potential":
        shape_potential(t, amplitude=100.0, length_scale=15.0)
    elif method == "boundary":
        shape_boundary(t, funnel_width=8.0)
    elif method == "feedback":
        shape_feedback(t, base_level=1.0, gradient_strength=10.0)
    else:
        raise ValueError(f"未知方法 {method}")

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))

    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)

    d0 = t.mean_distance(a)
    e.run(ticks)
    d1 = t.mean_distance(a)

    return {
        "method": method,
        "seed": seed,
        "d_initial": d0,
        "d_final": d1,
        "score": t.aggregation_score(a),
        "improvement": (d0 - d1) / d0 if d0 > 0 else 0.0,
    }


def run_outside(target_x: float, target_y: float, seed: int, ticks: int, n_agents: int) -> dict:
    """E2：目标在环境外。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=target_x, target_y=target_y)
    shape_potential(t, amplitude=100.0, length_scale=15.0)

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return {
        "target_x": target_x,
        "target_y": target_y,
        "seed": seed,
        "d_final": t.mean_distance(a),
        "score": t.aggregation_score(a),
    }


def run_two_attractors(sep: float, seed: int, ticks: int, n_agents: int) -> dict:
    """E3：两个等强吸引子。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)
    w.clear()

    ys, xs = np.mgrid[0:60, 0:60]
    d1 = np.hypot(xs - (30 - sep / 2), ys - 30)
    d2 = np.hypot(xs - (30 + sep / 2), ys - 30)
    w.data[0] = 100.0 * np.exp(-(d1 ** 2) / (2 * 8 ** 2)) + 100.0 * np.exp(-(d2 ** 2) / (2 * 8 ** 2))

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    left = int(np.sum(a.x < 30))
    right = int(np.sum(a.x >= 30))
    # 分裂度：0 表示全在一侧，0.5 表示完全对半
    split = min(left, right) / max(left + right, 1)

    return {
        "separation": sep,
        "seed": seed,
        "d_final": t.mean_distance(a),
        "score": t.aggregation_score(a),
        "left": left,
        "right": right,
        "split": split,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/g2")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============ E1 ============
    print("=" * 60)
    print("E1: 三个塑造方法对比")
    print("=" * 60)

    methods = ["baseline", "potential", "boundary", "feedback"]
    e1_results = []
    for m in methods:
        for rep in range(args.repeats):
            e1_results.append(run_method(m, rep * 1000, args.ticks, args.agents))
        print(f"\r  {m} 完成", end="", flush=True)
    print()
    print()

    print(f"{'方法':>12} {'聚集度':>18} {'改善':>18}")
    print("-" * 52)
    e1_summary = {}
    for m in methods:
        rs = [r for r in e1_results if r["method"] == m]
        scores = [r["score"] for r in rs]
        imps = [r["improvement"] for r in rs]
        e1_summary[m] = {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)),
            "improvement_mean": float(np.mean(imps)),
            "improvement_std": float(np.std(imps, ddof=1)),
        }
        s = e1_summary[m]
        print(f"{m:>12} {s['score_mean']:>9.3f} ± {s['score_std']:>5.3f} "
              f"{s['improvement_mean']:>9.1%} ± {s['improvement_std']:>5.1%}")

    # ============ E2 ============
    print()
    print("=" * 60)
    print("E2: 不可能性边界 1——目标在环境外")
    print("=" * 60)

    targets = [
        ("inside (30,30)", 30.0, 30.0),
        ("edge (0,30)", 0.0, 30.0),
        ("outside (-5,30)", -5.0, 30.0),
        ("outside (-10,30)", -10.0, 30.0),
        ("outside (-20,30)", -20.0, 30.0),
        ("far (100,100)", 100.0, 100.0),
    ]

    e2_results = []
    for name, tx, ty in targets:
        for rep in range(args.repeats):
            r = run_outside(tx, ty, rep * 1000, args.ticks, args.agents)
            r["name"] = name
            e2_results.append(r)

    print(f"{'目标位置':>16} {'最终距离':>12} {'聚集度':>18}")
    print("-" * 50)
    e2_summary = {}
    for name, tx, ty in targets:
        rs = [r for r in e2_results if r["name"] == name]
        ds = [r["d_final"] for r in rs]
        ss = [r["score"] for r in rs]
        e2_summary[name] = {
            "d_mean": float(np.mean(ds)),
            "score_mean": float(np.mean(ss)),
            "score_std": float(np.std(ss, ddof=1)),
        }
        s = e2_summary[name]
        print(f"{name:>16} {s['d_mean']:>12.2f} {s['score_mean']:>9.3f} ± {s['score_std']:>5.3f}")

    # ============ E3 ============
    print()
    print("=" * 60)
    print("E3: 不可能性边界 2——多个等强吸引子")
    print("=" * 60)

    separations = [0, 5, 10, 15, 20, 25, 30, 40]
    e3_results = []
    for sep in separations:
        for rep in range(args.repeats):
            e3_results.append(run_two_attractors(float(sep), rep * 1000, args.ticks, args.agents))

    print(f"{'吸引子间距':>10} {'聚集度':>18} {'分裂度':>18} {'左/右':>12}")
    print("-" * 62)
    e3_summary = {}
    for sep in separations:
        rs = [r for r in e3_results if r["separation"] == float(sep)]
        ss = [r["score"] for r in rs]
        splits = [r["split"] for r in rs]
        lefts = [r["left"] for r in rs]
        rights = [r["right"] for r in rs]
        e3_summary[sep] = {
            "score_mean": float(np.mean(ss)),
            "score_std": float(np.std(ss, ddof=1)),
            "split_mean": float(np.mean(splits)),
            "split_std": float(np.std(splits, ddof=1)),
        }
        s = e3_summary[sep]
        print(f"{sep:>10} {s['score_mean']:>9.3f} ± {s['score_std']:>5.3f} "
              f"{s['split_mean']:>9.3f} ± {s['split_std']:>5.3f} "
              f"{np.mean(lefts):>5.0f}/{np.mean(rights):<5.0f}")

    # ============ 结论 ============
    print()
    print("=" * 60)
    print("G2 结论")
    print("=" * 60)

    best_method = max(e1_summary, key=lambda m: e1_summary[m]["score_mean"])
    print(f"1. 最佳塑造方法：{best_method}（聚集度 {e1_summary[best_method]['score_mean']:.3f}）")

    # 找 E2 的临界点
    inside = e2_summary["inside (30,30)"]["score_mean"]
    outside = e2_summary["outside (-10,30)"]["score_mean"]
    print(f"2. 目标在环境内：聚集度 {inside:.3f}；在环境外：{outside:.3f}")
    print(f"   -> 目标必须在环境内，这是硬约束")

    # 找 E3 的临界间距
    threshold_sep = None
    for sep in separations:
        if e3_summary[sep]["score_mean"] < 0.5:
            threshold_sep = sep
            break
    if threshold_sep is not None:
        print(f"3. 吸引子间距 >= {threshold_sep} 时聚集度跌破 0.5")
        print(f"   -> 存在「吸引子竞争」的不可能性边界")

    # 保存
    (out_dir / "g2_results.json").write_text(
        json.dumps({
            "e1": e1_summary,
            "e2": e2_summary,
            "e3": {str(k): v for k, v in e3_summary.items()},
            "raw_e1": e1_results,
            "raw_e2": e2_results,
            "raw_e3": e3_results,
        }, indent=2),
        encoding="utf-8",
    )

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)

        # E1
        ax = axes[0]
        names = methods
        means = [e1_summary[m]["score_mean"] for m in names]
        stds = [e1_summary[m]["score_std"] for m in names]
        colors = ["#9ca3af", "#0f766e", "#b45309", "#6d28d9"]
        ax.bar(names, means, yerr=stds, capsize=4, color=colors)
        ax.set_ylabel("aggregation score")
        ax.set_title("E1: Shaping methods")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        # E2
        ax = axes[1]
        xs = [e2_summary[n]["d_mean"] for n, _, _ in targets]
        ys = [e2_summary[n]["score_mean"] for n, _, _ in targets]
        ax.plot(range(len(targets)), ys, marker="o", color="#b45309")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([n for n, _, _ in targets], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("aggregation score")
        ax.set_title("E2: Target outside environment")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

        # E3
        ax = axes[2]
        seps = separations
        scores = [e3_summary[s]["score_mean"] for s in seps]
        splits = [e3_summary[s]["split_mean"] for s in seps]
        ax.plot(seps, scores, marker="o", label="aggregation score", color="#0f766e")
        ax.plot(seps, splits, marker="s", label="split ratio", color="#dc2626")
        ax.set_xlabel("attractor separation")
        ax.set_ylabel("value")
        ax.set_title("E3: Competing attractors")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "g2_results.png")
        plt.close(fig)
        print(f"\n图已保存: {out_dir / 'g2_results.png'}")
    except ImportError:
        print("matplotlib 未安装")


if __name__ == "__main__":
    main()
