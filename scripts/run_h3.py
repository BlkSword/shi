"""H3 正式实验：异质性对分工与适应性的影响.

两个实验
----------
E1: sigma 扫描（稳态性能）
    测：均衡度、总产出、活跃个体数
    预期：倒 U 形或单调上升后饱和

E2: 环境突变（适应时间）
    在 t=800 时把类型 0 的需求清零
    测：系统多久重新分配劳动力
    预期：异质群体适应更快

用法
----
    python scripts/run_h3.py --experiment e1
    python scripts/run_h3.py --experiment e2
    python scripts/run_h3.py --experiment both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from shi.experiments.runner import AllocationConfig, run_allocation


def summarize(results: list[dict], key: str) -> dict:
    """按配置里的 key 分组，算均值/标准差。"""
    groups: dict = {}
    for r in results:
        k = r["config"][key]
        groups.setdefault(k, []).append(r)

    out = {}
    for k, rs in groups.items():
        out[k] = {
            "balance_mean": float(np.mean([r["balance"] for r in rs])),
            "balance_std": float(np.std([r["balance"] for r in rs], ddof=1)),
            "work_mean": float(np.mean([r["work_done"] for r in rs])),
            "work_std": float(np.std([r["work_done"] for r in rs], ddof=1)),
            "n": len(rs),
        }
    return out


def run_e1(sigmas: list[float], repeats: int, ticks: int, agents: int) -> list[dict]:
    """E1: sigma 扫描。"""
    results = []
    total = len(sigmas) * repeats
    done = 0
    for sigma in sigmas:
        for rep in range(repeats):
            cfg = AllocationConfig(
                n_agents=agents,
                n_ticks=ticks,
                threshold_sigma=sigma,
                seed=rep * 1000,
                record_every=0,
                hash_every=0,
            )
            results.append(run_allocation(cfg))
            done += 1
            print(f"\r  E1 进度: {done}/{total}", end="", flush=True)
    print()
    return results


def run_e2(sigmas: list[float], repeats: int, ticks: int, agents: int, shift_at: int) -> list[dict]:
    """E2: 环境突变，测适应时间。"""
    results = []
    total = len(sigmas) * repeats
    done = 0
    for sigma in sigmas:
        for rep in range(repeats):
            cfg = AllocationConfig(
                n_agents=agents,
                n_ticks=ticks,
                threshold_sigma=sigma,
                seed=rep * 1000,
                shift_at=shift_at,
                shift_type=0,
                record_every=20,
                hash_every=0,
            )
            results.append(run_allocation(cfg))
            done += 1
            print(f"\r  E2 进度: {done}/{total}", end="", flush=True)
    print()
    return results


def adaptation_time(metrics: list[dict], shift_tick: int, window: int = 200) -> float:
    """适应时间：突变后均衡度曲线下面积的归一化度量。

    为什么不用「首次达到阈值」
    --------------------------
    早期版本用「均衡度首次恢复到突变前 90% 的 tick」。
    问题是：记录间隔是 20 tick，所以结果只能是 20 的倍数，
    所有快速适应的配置都得到 20，无法区分。

    改用「恢复曲线下面积」：
        adapt = sum((target - balance[t]) for t in after) / target
    单位是 tick，且是连续值，能区分不同速度。
    值越小表示恢复越快。
    """
    before = [m for m in metrics if m["tick"] <= shift_tick]
    after = [m for m in metrics if m["tick"] > shift_tick]
    if not before or not after:
        return float("nan")

    baseline = float(np.mean([m["balance"] for m in before[-5:]]))
    if baseline <= 0:
        return float("nan")

    # 恢复曲线下面积（归一化到 baseline）
    deficit = sum(max(0.0, baseline - m["balance"]) for m in after)
    return float(deficit / baseline)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["e1", "e2", "both"], default="both")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--agents", type=int, default=150)
    ap.add_argument("--shift-at", type=int, default=800)
    ap.add_argument("--out", type=str, default="experiments/h3")
    args = ap.parse_args()

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("e1", "both"):
        print("=" * 60)
        print("E1: sigma 扫描（稳态性能）")
        print("=" * 60)
        results = run_e1(sigmas, args.repeats, args.ticks, args.agents)
        stats = summarize(results, "threshold_sigma")

        print()
        print(f"{'sigma':>7} {'均衡度':>18} {'总产出':>18} {'样本':>6}")
        print("-" * 55)
        for k in sorted(stats):
            s = stats[k]
            print(
                f"{k:>7.1f} {s['balance_mean']:>9.3f} +- {s['balance_std']:>5.3f} "
                f"{s['work_mean']:>9.1f} +- {s['work_std']:>6.1f} {s['n']:>6}"
            )

        (out_dir / "e1_results.json").write_text(
            json.dumps({"stats": {str(k): v for k, v in stats.items()}}, indent=2),
            encoding="utf-8",
        )

        # 出图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ks = sorted(stats)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=120)

            ax1.errorbar(ks, [stats[k]["balance_mean"] for k in ks],
                         yerr=[stats[k]["balance_std"] for k in ks],
                         marker="o", capsize=4, color="#0f766e")
            ax1.set_xlabel("threshold sigma")
            ax1.set_ylabel("task balance (0-1)")
            ax1.set_title("Division of labor vs heterogeneity")
            ax1.grid(True, alpha=0.3)

            ax2.errorbar(ks, [stats[k]["work_mean"] for k in ks],
                         yerr=[stats[k]["work_std"] for k in ks],
                         marker="s", capsize=4, color="#b45309")
            ax2.set_xlabel("threshold sigma")
            ax2.set_ylabel("total work done")
            ax2.set_title("Productivity vs heterogeneity")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(out_dir / "e1_sigma.png")
            plt.close(fig)
            print(f"\n图已保存: {out_dir / 'e1_sigma.png'}")
        except ImportError:
            print("matplotlib 未安装，跳过出图")

    if args.experiment in ("e2", "both"):
        print()
        print("=" * 60)
        print(f"E2: 环境突变（t={args.shift_at} 时清零类型 0）")
        print("=" * 60)
        results = run_e2(sigmas, args.repeats, args.ticks, args.agents, args.shift_at)

        print()
        print(f"{'sigma':>7} {'适应时间(tick)':>16} {'突变后均衡度':>16}")
        print("-" * 45)
        for sigma in sigmas:
            rs = [r for r in results if r["config"]["threshold_sigma"] == sigma]
            times = [adaptation_time(r["metrics"], args.shift_at) for r in rs]
            valid = [t for t in times if not np.isnan(t)]
            post_bal = [
                np.mean([m["balance"] for m in r["metrics"] if m["tick"] > args.shift_at + 200])
                for r in rs
            ]
            t_str = f"{np.mean(valid):.0f}" if valid else "N/A"
            print(f"{sigma:>7.1f} {t_str:>16} {np.mean(post_bal):>16.3f}")

        (out_dir / "e2_results.json").write_text(
            json.dumps([{"sigma": r["config"]["threshold_sigma"],
                         "balance": r["balance"],
                         "work_done": r["work_done"],
                         "adapt_time": adaptation_time(r["metrics"], args.shift_at)}
                        for r in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
