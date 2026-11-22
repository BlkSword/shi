"""结果分析：出图与统计.

设计要点（待实现）
------------------
- 出图用 matplotlib，保存为 PNG + SVG
- 统计用 scipy.stats（ANOVA、效应量）
- 不只看 p 值，必须报告效应量
"""

from __future__ import annotations

__all__ = ["plot_convergence", "two_way_anova"]


def plot_convergence(results, out_path: str) -> None:
    """画收敛曲线。"""
    raise NotImplementedError("P0 待实现")


def two_way_anova(df, dv: str, factor_a: str, factor_b: str) -> dict:
    """双因素方差分析 + 效应量。"""
    raise NotImplementedError("P0 待实现")
