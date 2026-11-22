"""指标采集.

四个核心指标
------------
1. 收敛时间：从开始到稳定达成目标的 tick 数
2. 稳态性能：稳定后的目标达成度（0-1）
3. 鲁棒性：扰动后恢复到稳态的时间
4. 成本：每 tick 的通信量 / 计算量

设计要点
--------
- 每 tick 记录到内存，实验结束一次性落盘（避免频繁 IO 破坏确定性）
- 输出 CSV，便于 pandas 分析
- 记录器本身不参与仿真，不影响状态哈希
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = ["MetricsRecorder"]


class MetricsRecorder:
    """按 tick 记录指标。

    用法
    ----
    >>> rec = MetricsRecorder()
    >>> rec.record(tick=0, coverage=0.1, n_active=5)
    >>> rec.record(tick=1, coverage=0.3, n_active=12)
    >>> rec.to_csv("results.csv")
    """

    def __init__(self):
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] | None = None

    def record(self, tick: int, **kwargs: Any) -> None:
        """记录一个 tick 的指标。

        第一次调用时确定列集合；后续调用必须使用相同的键，
        否则抛错（防止不同实验的 CSV 混在一起）。
        """
        row = {"tick": int(tick), **kwargs}
        if self._columns is None:
            self._columns = list(row.keys())
        elif list(row.keys()) != self._columns:
            raise ValueError(
                f"指标列不一致。期望 {self._columns}，收到 {list(row.keys())}"
            )
        self._rows.append(row)

    @property
    def n_ticks(self) -> int:
        return len(self._rows)

    @property
    def columns(self) -> list[str]:
        return list(self._columns) if self._columns else []

    def series(self, column: str) -> list[float]:
        """取出某一列的全部值。"""
        if self._columns is None or column not in self._columns:
            raise KeyError(f"未知列 {column!r}，可用列: {self._columns}")
        return [row[column] for row in self._rows]

    def last(self, column: str) -> float:
        """取某一列最后一个值。"""
        if not self._rows:
            raise ValueError("还没有记录任何数据")
        return self._rows[-1][column]

    def summary(self) -> dict[str, float]:
        """每个数值列的最终值（用于快速比较实验）。"""
        if not self._rows:
            return {}
        return {
            k: v
            for k, v in self._rows[-1].items()
            if k != "tick" and isinstance(v, (int, float))
        }

    def to_csv(self, path: str | Path) -> None:
        """写入 CSV。"""
        if not self._rows:
            raise ValueError("没有数据可写")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self._rows)

    def to_rows(self) -> list[dict[str, Any]]:
        """返回原始行（用于 pandas DataFrame）。"""
        return list(self._rows)

    def clear(self) -> None:
        self._rows.clear()
        self._columns = None

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f"MetricsRecorder(ticks={len(self._rows)}, columns={self._columns})"
