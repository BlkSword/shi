"""P0 核心验收：同种子 100 次运行，哈希序列必须完全一致.

这是整个研究的地基。如果这条不通过，后面所有实验都不可信。

用法
----
    python scripts/verify_determinism.py
    python scripts/verify_determinism.py --runs 100 --ticks 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shi.experiments.runner import ForagingConfig, run_foraging


def main() -> None:
    ap = argparse.ArgumentParser(description="验证确定性")
    ap.add_argument("--runs", type=int, default=100, help="重复次数")
    ap.add_argument("--ticks", type=int, default=500)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = ForagingConfig(
        n_agents=args.agents,
        n_ticks=args.ticks,
        n_resources=40,
        seed=args.seed,
        hash_every=50,
        record_every=0,  # 不记录指标，只关心哈希
    )

    print(f"验证配置: {args.runs} 次运行, {args.ticks} ticks, {args.agents} 个体, seed={args.seed}")
    print()

    baseline: list[str] | None = None
    mismatches = 0
    for i in range(args.runs):
        result = run_foraging(cfg)
        hashes = result["hash_sequence"]

        if baseline is None:
            baseline = hashes
            print(f"  基准哈希序列: {len(hashes)} 个")
            print(f"  首个: {hashes[0]}")
            print(f"  末个: {hashes[-1]}")
            print()
        elif hashes != baseline:
            mismatches += 1
            print(f"  [FAIL] 第 {i + 1} 次运行哈希不一致")
            for j, (a, b) in enumerate(zip(baseline, hashes)):
                if a != b:
                    print(f"         首个分歧在索引 {j}: {a} != {b}")
                    break

        if (i + 1) % 10 == 0:
            print(f"  已完成 {i + 1}/{args.runs} 次...")

    print()
    print("=" * 60)
    if mismatches == 0:
        print(f"[PASS] {args.runs} 次运行，哈希序列 100% 一致")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"[FAIL] {mismatches}/{args.runs} 次运行哈希不一致")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
