"""指标采集.

四个核心指标
------------
1. 收敛时间：从开始到稳定达成目标的 tick 数
2. 稳态性能：稳定后的目标达成度（0-1）
3. 鲁棒性：扰动后恢复到稳态的时间
4. 成本：每 tick 的通信量 / 计算量

设计要点（待实现）
------------------
- 每 tick 记录到内存，实验结束一次性落盘
- 输出格式：CSV（便于 pandas 分析）
"""

from __future__ import annotations

__all__ = ["MetricsRecorder"]


class MetricsRecorder:
    """指标记录器。"""

    def __init__(self):
        raise NotImplementedError("P0 待实现")

    def record(self, tick: int, **kwargs) -> None:
        raise NotImplementedError("P0 待实现")

    def to_csv(self, path: str) -> None:
        raise NotImplementedError("P0 待实现")
