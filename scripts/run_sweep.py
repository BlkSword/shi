"""参数扫描脚本：跑一组配置，出图.

用法
----
    python scripts/run_sweep.py --param threshold_sigma --values 0 0.1 0.3 0.6 1.0
    python scripts/run_sweep.py --preset h3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shi.experiments.analyze import plot_sweep, summarize_sweep
from shi.experiments.runner import ForagingConfig
from shi.experiments.sweep import grid_sweep

PRESETS = {
    "h3": {
        "param": "threshold_sigma",
        "values": [0.0, 0.1, 0.2, 0.4, 0.7, 1.0],
        "desc": "H3 预实验：异质性对性能的影响",
    },
    "h2": {
        "param": "n_channels",
        "values": [1, 3, 8],
        "desc": "H2 预实验：环境信息容量对性能的影响",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description="参数扫描")
    ap.add_argument("--preset", choices=list(PRESETS), help="预设实验")
    ap.add_argument("--param", type=str, help="扫描的参数名")
    ap.add_argument("--values", type=str, nargs="+", help="参数取值")
    ap.add_argument("--repeats", type=int, default=5, help="每个配置重复次数")
    ap.add_argument("--agents", type=int, default=150)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=str, default="experiments/sweep")
    args = ap.parse_args()

    if args.preset:
        preset = PRESETS[args.preset]
        param = preset["param"]
        raw_values = preset["values"]
        print(f"预设: {preset['desc']}")
    elif args.param and args.values:
        param = args.param
        raw_values = args.values
    else:
        ap.error("必须指定 --preset 或 (--param 和 --values)")

    # 类型转换：根据基础配置里该字段的类型
    base = ForagingConfig(n_agents=args.agents, n_ticks=args.ticks)
    field_type = type(getattr(base, param))
    values = [field_type(v) for v in raw_values]

    print(f"扫描参数: {param} = {values}")
    print(f"重复次数: {args.repeats}, 个体数: {args.agents}, ticks: {args.ticks}")
    print()

    def progress(done: int, total: int) -> None:
        pct = done / total * 100
        print(f"\r  进度: {done}/{total} ({pct:.0f}%)", end="", flush=True)

    results = grid_sweep(
        base_config=base,
        param_grid={param: values},
        n_repeats=args.repeats,
        n_workers=args.workers,
        progress=progress,
    )
    print()
    print()

    # 汇总
    stats = summarize_sweep(results, param)
    print(f"{'值':<12} {'均值':>10} {'标准差':>10} {'样本数':>8}")
    print("-" * 44)
    for k in sorted(stats.keys(), key=lambda x: (x is None, x)):
        s = stats[k]
        print(f"{str(k):<12} {s['mean']:>9.2%} {s['std']:>10.2%} {s['n']:>8}")

    # 出图
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = plot_sweep(results, param, out_dir / f"{param}.png")
    print()
    print(f"图已保存: {png}")

    # 保存原始结果
    raw = out_dir / f"{param}.json"
    raw.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"原始数据: {raw}")


if __name__ == "__main__":
    main()
