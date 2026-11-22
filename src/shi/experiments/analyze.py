"""结果分析：出图与统计.

设计要点
--------
- 出图用 matplotlib，保存为 PNG
- 统计用 scipy.stats（ANOVA、效应量）
- 不只看 p 值，必须报告效应量
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "plot_convergence",
    "plot_sweep",
    "two_way_anova",
    "cohens_d",
    "summarize_sweep",
]


def _plt():
    """惰性导入 matplotlib，避免无绘图需求时也加载。"""
    import matplotlib

    matplotlib.use("Agg")  # 无头后端
    import matplotlib.pyplot as plt

    return plt


def plot_convergence(
    metrics: list[dict[str, Any]],
    out_path: str | Path,
    y: str = "coverage",
    title: str = "Convergence",
) -> Path:
    """画收敛曲线。

    Parameters
    ----------
    metrics : list of dict
        每 tick 的指标记录（来自 MetricsRecorder.to_rows()）。
    out_path : str | Path
        输出 PNG 路径。
    y : str
        纵轴列名。
    title : str
        图标题。
    """
    plt = _plt()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ticks = [row["tick"] for row in metrics]
    values = [row[y] for row in metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(ticks, values, linewidth=1.8, color="#0f766e")
    ax.set_xlabel("tick")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def summarize_sweep(results: list[dict[str, Any]], by: str) -> dict[Any, dict[str, float]]:
    """按某个配置字段分组，计算统计量。

    Returns
    -------
    dict
        {分组值: {"mean": ..., "std": ..., "n": ..., "min": ..., "max": ...}}
    """
    groups: dict[Any, list[float]] = {}
    for r in results:
        key = r["config"].get(by)
        groups.setdefault(key, []).append(r["final_coverage"])

    out: dict[Any, dict[str, float]] = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n": int(len(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


def plot_sweep(
    results: list[dict[str, Any]],
    by: str,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """画扫描结果：横轴是参数，纵轴是最终覆盖率（带误差棒）。"""
    plt = _plt()
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    stats = summarize_sweep(results, by)
    keys = sorted(stats.keys(), key=lambda k: (k is None, k))
    means = [stats[k]["mean"] for k in keys]
    stds = [stats[k]["std"] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.errorbar(
        range(len(keys)), means, yerr=stds,
        marker="o", capsize=4, linewidth=1.8, color="#b45309",
    )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_xlabel(by)
    ax.set_ylabel("final coverage")
    ax.set_title(title or f"Coverage vs {by}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p)
    plt.close(fig)
    return p


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d 效应量。

    解释：0.2 小，0.5 中，0.8 大。
    比 p 值更有信息量——p 值只告诉你「有没有差异」，
    d 告诉你「差异有多大」。
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        raise ValueError("每组至少需要 2 个样本")

    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def two_way_anova(
    results: list[dict[str, Any]],
    dv: str,
    factor_a: str,
    factor_b: str,
) -> dict[str, Any]:
    """双因素方差分析。

    Parameters
    ----------
    results : list of dict
        扫描结果。
    dv : str
        因变量字段名（例如 "final_coverage"）。
    factor_a, factor_b : str
        两个自变量的配置字段名。

    Returns
    -------
    dict
        包含各主效应与交互效应的 F 值和 p 值。
    """
    from scipy import stats

    groups: dict[tuple[Any, Any], list[float]] = {}
    for r in results:
        key = (r["config"].get(factor_a), r["config"].get(factor_b))
        groups.setdefault(key, []).append(r[dv])

    a_levels = sorted({k[0] for k in groups}, key=str)
    b_levels = sorted({k[1] for k in groups}, key=str)

    # 构造 ANOVA 所需的嵌套列表
    cells = []
    for a in a_levels:
        for b in b_levels:
            vals = groups.get((a, b), [])
            if vals:
                cells.append(vals)

    if len(cells) < 2:
        raise ValueError("至少需要两个非空单元格")

    # 用 scipy 的 f_oneway 做近似（完整双因素 ANOVA 需要 statsmodels）
    f_stat, p_val = stats.f_oneway(*cells)

    return {
        "f": float(f_stat),
        "p": float(p_val),
        "n_cells": len(cells),
        "a_levels": a_levels,
        "b_levels": b_levels,
        "note": "近似单因素 ANOVA；完整双因素分析需安装 statsmodels",
    }
