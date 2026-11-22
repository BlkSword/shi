"""跑通基线：P0 的最后一个任务.

基线定义
--------
最朴素的版本：阈值响应 + 信息素跟随 + 随机游走。
这个基线是后面所有对比的锚点。没有它，无法知道新方法到底有没有用。

用法
----
    python scripts/run_baseline.py
    python scripts/run_baseline.py --agents 500 --ticks 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接运行而不安装
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shi.experiments.runner import ForagingConfig, run_foraging, save_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="跑觅食基线实验")
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--resources", type=int, default=80)
    ap.add_argument("--sigma", type=float, default=0.0, help="阈值异质性")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/baseline")
    args = ap.parse_args()

    cfg = ForagingConfig(
        n_agents=args.agents,
        n_ticks=args.ticks,
        n_resources=args.resources,
        threshold_sigma=args.sigma,
        seed=args.seed,
    )

    print(f"配置: {cfg.n_agents} 个体, {cfg.n_ticks} ticks, {cfg.n_resources} 资源点")
    print("运行中...")

    result = run_foraging(cfg)

    out = save_experiment(result, args.out, name="baseline")

    print()
    print("=" * 50)
    print(f"耗时:        {result['elapsed_sec']:.2f} 秒")
    print(f"最终覆盖率:  {result['final_coverage']:.2%}")
    print(f"采集总数:    {result['total_collected']}")
    print(f"哈希数量:    {len(result['hash_sequence'])}")
    print(f"首个哈希:    {result['hash_sequence'][0] if result['hash_sequence'] else 'N/A'}")
    print(f"末个哈希:    {result['hash_sequence'][-1] if result['hash_sequence'] else 'N/A'}")
    print(f"结果目录:    {out}")
    print("=" * 50)

    # 打印收敛曲线（文本版）
    rows = result["metrics"]
    if rows:
        print()
        print("收敛曲线（覆盖率）:")
        step = max(1, len(rows) // 20)
        for row in rows[::step]:
            bar = "#" * int(row["coverage"] * 50)
            print(f"  t={row['tick']:>6}  {row['coverage']:>6.2%}  {bar}")


if __name__ == "__main__":
    main()
