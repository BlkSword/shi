"""单次实验执行器.

约定
----
每次实验在 experiments/ 下创建一个目录：

    experiments/2026-11-25_e1_rule_complexity/
      config.yaml
      results.csv
      notes.md

notes.md 必须手写：假设、观察、结论。
"""

from __future__ import annotations

__all__ = ["run_experiment"]


def run_experiment(config: dict, out_dir: str) -> dict:
    """按配置跑一次实验，返回指标摘要。"""
    raise NotImplementedError("P0 待实现")
