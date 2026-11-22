"""诊断：为什么修复搜索后异质性效应消失了？"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.allocation import AllocationTask, make_allocation_stepper, make_allocation_world


def run(sigma, seed=0, ticks=1500):
    w = make_allocation_world(60, 60, 3)
    t = AllocationTask(w, n_types=3, sites_per_type=6, demand=60.0, regen=0.05,
                       rng=np.random.default_rng(seed))
    t.setup()
    a = AgentState(150)
    a.place_random(60.0, 60.0, np.random.default_rng(seed + 1))
    a.set_thresholds(sample_thresholds(150, sigma=sigma,
                                      rng=np.random.default_rng(seed + 2), n_types=3))
    e = Engine(w, a, seed=seed, stepper=make_allocation_stepper(t), hash_every=0)

    # 记录每 200 tick 的各类型活跃数
    history = []
    for i in range(ticks):
        e.step()
        if i % 200 == 0:
            per_type = [int(np.count_nonzero(a.state == k+1)) for k in range(3)]
            history.append((i, per_type))

    return t.balance(), t.work_done, history, t.work_by_type.tolist()


print("=== 各 sigma 下的分工演化 ===")
for sigma in [0.0, 0.3, 1.0]:
    bal, work, hist, by_type = run(sigma)
    print(f"\nsigma={sigma}: balance={bal:.3f}, work={work:.1f}, by_type={by_type}")
    for tick, per_type in hist:
        print(f"  t={tick:>4}: {per_type}")

print()
print("=== 关键问题 ===")
print("如果 sigma=0 时各类型活跃数也均衡，说明分工不是由阈值驱动的，")
print("而是由「个体随机分布 + 就近工作」驱动的。")
print("这意味着：修复搜索行为后，空间位置取代了阈值成为分工的主因。")
