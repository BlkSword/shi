"""参数扫描：多进程并行跑多组配置.

设计要点
--------
- 用 multiprocessing.Pool 而不是线程（Python 有 GIL）
- 进程隔离天然保证确定性：每个进程独立跑，互不干扰
- 支持网格搜索和重复实验（每个配置跑 n_repeats 次，取统计量）

为什么每个配置要跑多次
----------------------
单次运行有随机性（种子不同）。要比较两个配置的差异，
必须看分布而不是单点。n_repeats 就是样本量。
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from dataclasses import replace
from typing import Any, Callable

from shi.experiments.runner import ForagingConfig, run_foraging

__all__ = ["grid_sweep", "expand_grid"]


def expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """把参数网格展开成配置列表。

    >>> expand_grid({"a": [1, 2], "b": [3, 4]})
    [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _run_one(args: tuple[ForagingConfig, int]) -> dict[str, Any]:
    """跑单个配置（供多进程调用）。"""
    cfg, repeat_idx = args
    # 每个重复用不同的种子，但种子是确定的（可复现）
    cfg_with_seed = replace(cfg, seed=cfg.seed + repeat_idx * 1000)
    result = run_foraging(cfg_with_seed)
    return {
        "config": cfg_with_seed.to_dict(),
        "repeat": repeat_idx,
        "final_coverage": result["final_coverage"],
        "total_collected": result["total_collected"],
        "elapsed_sec": result["elapsed_sec"],
        "last_hash": result["hash_sequence"][-1] if result["hash_sequence"] else None,
    }


def grid_sweep(
    base_config: ForagingConfig,
    param_grid: dict[str, list[Any]],
    n_repeats: int = 1,
    n_workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """网格扫描。

    Parameters
    ----------
    base_config : ForagingConfig
        基础配置，网格中的参数会覆盖它。
    param_grid : dict[str, list]
        参数网格，例如 {"threshold_sigma": [0.0, 0.1, 0.5]}。
    n_repeats : int
        每个配置重复次数。
    n_workers : int | None
        并行进程数。None 表示用 CPU 核心数。
    progress : Callable | None
        进度回调 (done, total)。

    Returns
    -------
    list of dict
        每个元素是一次运行的结果。
    """
    combos = expand_grid(param_grid)
    tasks: list[tuple[ForagingConfig, int]] = []
    for combo in combos:
        cfg = replace(base_config, **combo)
        for r in range(n_repeats):
            tasks.append((cfg, r))

    total = len(tasks)
    results: list[dict[str, Any]] = []

    if n_workers is None:
        n_workers = max(1, (mp.cpu_count() or 1) - 1)

    if n_workers == 1:
        for i, task in enumerate(tasks):
            results.append(_run_one(task))
            if progress:
                progress(i + 1, total)
    else:
        with mp.Pool(processes=n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_run_one, tasks)):
                results.append(res)
                if progress:
                    progress(i + 1, total)

    # 按配置排序，保证输出顺序确定
    results.sort(key=lambda r: (str(r["config"]), r["repeat"]))
    return results
