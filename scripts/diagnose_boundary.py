"""诊断：边界条件到底有没有阻挡个体？个体最终停在哪？"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask, make_aggregation_stepper, make_aggregation_world, shape_boundary,
)

w = make_aggregation_world(60, 60)
t = AggregationTask(w, target_x=30.0, target_y=30.0)
shape_boundary(t, n_rings=3, ring_gap=6.0, gap_ratio=0.15)

print("=== 边界条件统计 ===")
print(f"blocked 格子数: {t.blocked.sum()} / {t.blocked.size} ({t.blocked.mean():.1%})")
print(f"目标 (30,30) 是否被阻挡: {t.is_blocked(30.0, 30.0)}")

# 检查 blocked 的空间分布
ys, xs = np.mgrid[0:60, 0:60]
dist = np.hypot(xs-30, ys-30)
print(f"\nblocked 格子的距离分布:")
for lo, hi in [(0,10),(10,20),(20,30),(30,50)]:
    mask = (dist>=lo)&(dist<hi)
    if mask.sum()>0:
        print(f"  距离 {lo}-{hi}: {t.blocked[mask].mean():.1%} 被阻挡 ({mask.sum()} 格)")

a = AgentState(100)
a.place_random(60.0, 60.0, np.random.default_rng(0))
stepper = make_aggregation_stepper(t, exploration=0.5)
e = Engine(w, a, seed=0, stepper=stepper, hash_every=0)

print(f"\n=== 演化过程 ===")
print(f"{'tick':>6} {'平均距离':>10} {'在blocked上':>12} {'聚集度':>9}")
for tick in range(800):
    e.step()
    if tick % 100 == 0 or tick == 799:
        d = t.mean_distance(a)
        on_blocked = sum(1 for i in range(a.n) if t.is_blocked(a.x[i], a.y[i]))
        print(f"{tick:>6} {d:>10.2f} {on_blocked:>12} {t.aggregation_score(a):>9.3f}")

print(f"\n=== 最终位置分布 ===")
print(f"x 范围: {a.x.min():.1f} - {a.x.max():.1f}")
print(f"y 范围: {a.y.min():.1f} - {a.y.max():.1f}")
print(f"x 标准差: {a.x.std():.2f}, y 标准差: {a.y.std():.2f}")
