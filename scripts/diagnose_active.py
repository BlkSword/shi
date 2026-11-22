"""诊断：为什么活跃个体比例只有 0.4%-4%？"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.allocation import AllocationTask, make_allocation_world


print("=== 任务点处的刺激强度 vs 阈值 ===")
print()

for demand in [20.0, 50.0, 100.0, 200.0]:
    w = make_allocation_world(60, 60, 3)
    t = AllocationTask(w, n_types=3, sites_per_type=6, demand=demand,
                       rng=np.random.default_rng(0))
    t.setup()

    # 在任务点正上方测刺激
    x, y = t.sites[0][0]
    raw = w.read(x, y, "task_0")
    norm = w.sense(x, y, "task_0", 2, normalize=True)
    total = w.sense(x, y, "task_0", 2, normalize=False)

    # 阈值中位数 1.0 的个体在任务点处的响应概率
    p_norm = response_probability(norm, 1.0)
    p_raw = response_probability(raw, 1.0)

    print(f"demand={demand:>6.1f}")
    print(f"  单格浓度={raw:>8.2f}  邻域总和={total:>8.2f}  邻域均值={norm:>8.3f}")
    print(f"  p(用均值)={p_norm:.3f}   p(用单格)={p_raw:.3f}   p>0.5? {p_norm > 0.5}")
    print()

print("=== 结论 ===")
print("demand=20 时，邻域均值 = 20/25 = 0.8，阈值中位数 1.0")
print("p = 0.8^2 / (0.8^2 + 1^2) = 0.39 < 0.5  -> 大部分个体判定为闲置")
print()
print("修复方向：提高 demand，或改用单格浓度而非邻域均值")
