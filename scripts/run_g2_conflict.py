"""G2 扩展实验：冲突约束 vs 对齐约束."""

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
    shape_aligned_boundary,
    shape_boundary,
    shape_potential,
)


def run_conflict(n_rings: int, gap_ratio: float, seed: int, ticks: int, n_agents: int) -> dict:
    """势能场 + 冲突边界（同心环）。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)
    shape_potential(t, amplitude=100.0, length_scale=15.0)
    saved = w.data[0].copy()

    if n_rings > 0:
        shape_boundary(t, n_rings=n_rings, ring_gap=6.0, gap_ratio=gap_ratio,
                       rng=np.random.default_rng(seed))
        w.data[0] = saved
    else:
        t.blocked = None

    density = float(t.blocked.mean()) if t.blocked is not None else 0.0

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return {"density": density, "score": t.aggregation_score(a), "kind": "conflict"}


def run_aligned(grad_threshold: float, seed: int, ticks: int, n_agents: int) -> dict:
    """势能场 + 对齐边界（只在梯度弱处）。"""
    w = make_aggregation_world(60, 60)
    t = AggregationTask(w, target_x=30.0, target_y=30.0)
    shape_potential(t, amplitude=100.0, length_scale=15.0)
    density = shape_aligned_boundary(t, grad_threshold=grad_threshold)

    a = AgentState(n_agents)
    a.place_random(60.0, 60.0, np.random.default_rng(seed))
    e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=0)
    e.run(ticks)

    return {"density": density, "score": t.aggregation_score(a), "kind": "aligned"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=800)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/g2_conflict")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== E1: 冲突边界的密度扫描 =====
    print("=" * 60)
    print("E1: 冲突边界（同心环）密度 vs 性能")
    print("=" * 60)

    conflict_results = []
    for n_rings in [0, 1, 2, 3, 4, 5, 6]:
        for rep in range(args.repeats):
            conflict_results.append(run_conflict(n_rings, 0.15, rep * 1000, args.ticks, args.agents))

    print(f"{'环数':>5} {'边界密度':>10} {'聚集度':>18}")
    print("-" * 38)
    conflict_summary = {}
    for n_rings in [0, 1, 2, 3, 4, 5, 6]:
        rs = conflict_results[n_rings * args.repeats : (n_rings + 1) * args.repeats]
        scores = [r["score"] for r in rs]
        density = np.mean([r["density"] for r in rs])
        conflict_summary[n_rings] = {
            "density": float(density),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)),
        }
        s = conflict_summary[n_rings]
        print(f"{n_rings:>5} {density:>9.1%} {s['score_mean']:>9.3f} ± {s['score_std']:>5.3f}")

    # ===== E2: 对齐边界的阈值扫描 =====
    print()
    print("=" * 60)
    print("E2: 对齐边界（只在梯度弱处）阈值 vs 性能")
    print("=" * 60)

    thresholds = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    aligned_results = []
    for th in thresholds:
        for rep in range(args.repeats):
            aligned_results.append(run_aligned(th, rep * 1000, args.ticks, args.agents))

    print(f"{'梯度阈值':>10} {'边界密度':>10} {'聚集度':>18}")
    print("-" * 42)
    aligned_summary = {}
    for i, th in enumerate(thresholds):
        rs = aligned_results[i * args.repeats : (i + 1) * args.repeats]
        scores = [r["score"] for r in rs]
        density = np.mean([r["density"] for r in rs])
        aligned_summary[th] = {
            "density": float(density),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)),
        }
        s = aligned_summary[th]
        print(f"{th:>10.1f} {density:>9.1%} {s['score_mean']:>9.3f} ± {s['score_std']:>5.3f}")

    # ===== E3: 同密度下的对比 =====
    print()
    print("=" * 60)
    print("E3: 同密度下冲突 vs 对齐")
    print("=" * 60)

    print(f"{'密度区间':>12} {'冲突':>10} {'对齐':>10} {'差异':>10}")
    print("-" * 46)

    # 找密度接近的配对
    pairs = []
    for n_rings, cs in conflict_summary.items():
        if n_rings == 0:
            continue
        for th, as_ in aligned_summary.items():
            if abs(cs["density"] - as_["density"]) < 0.03:
                pairs.append((cs["density"], cs["score_mean"], as_["score_mean"], n_rings, th))

    for density, c_score, a_score, n_rings, th in pairs:
        diff = a_score - c_score
        print(f"{density:>11.1%} {c_score:>10.3f} {a_score:>10.3f} {diff:>+10.3f}")

    # 保存
    (out_dir / "conflict_results.json").write_text(
        json.dumps({
            "conflict": {str(k): v for k, v in conflict_summary.items()},
            "aligned": {str(k): v for k, v in aligned_summary.items()},
            "pairs": [{"density": p[0], "conflict": p[1], "aligned": p[2]} for p in pairs],
        }, indent=2),
        encoding="utf-8",
    )

    # 出图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)

        # 左图：密度 vs 性能
        c_dens = [conflict_summary[k]["density"] for k in conflict_summary]
        c_scores = [conflict_summary[k]["score_mean"] for k in conflict_summary]
        a_dens = [aligned_summary[k]["density"] for k in aligned_summary]
        a_scores = [aligned_summary[k]["score_mean"] for k in aligned_summary]

        ax1.plot(c_dens, c_scores, marker="o", label="conflicting boundary", color="#dc2626")
        ax1.plot(a_dens, a_scores, marker="s", label="aligned boundary", color="#0f766e")
        ax1.set_xlabel("boundary density")
        ax1.set_ylabel("aggregation score")
        ax1.set_title("Conflict vs aligned constraints")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：环数 vs 性能
        rings = list(conflict_summary.keys())
        ax2.errorbar(rings, [conflict_summary[k]["score_mean"] for k in rings],
                     yerr=[conflict_summary[k]["score_std"] for k in rings],
                     marker="o", capsize=4, color="#dc2626")
        ax2.set_xlabel("number of rings")
        ax2.set_ylabel("aggregation score")
        ax2.set_title("Performance degradation by ring count")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "conflict_results.png")
        plt.close(fig)
        print(f"\n图已保存: {out_dir / 'conflict_results.png'}")
    except ImportError:
        print("matplotlib 未安装")


if __name__ == "__main__":
    main()
