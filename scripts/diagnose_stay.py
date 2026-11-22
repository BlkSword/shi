"""诊断：个体到达任务点后发生了什么？"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from shi.core.agent import AgentState, response_probability, sample_thresholds
from shi.core.engine import Engine
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.allocation import AllocationTask, make_allocation_stepper, make_allocation_world


w = make_allocation_world(60, 60, 3)
t = AllocationTask(w, n_types=3, sites_per_type=6, demand=60.0, regen=0.05,
                   rng=np.random.default_rng(0))
t.setup()

a = AgentState(150)
a.place_random(60.0, 60.0, np.random.default_rng(1))
a.set_thresholds(sample_thresholds(150, sigma=0.5, rng=np.random.default_rng(2), n_types=3))

stepper = make_allocation_stepper(t)
e = Engine(w, a, seed=0, stepper=stepper, hash_every=0)

print("tick | 活跃 | 各类型活跃数 | 平均刺激 | 平均p | 到最近任务点距离")
print("-" * 95)

for tick in range(600):
    e.step()
    if tick % 100 == 0 or tick == 599:
        active = a.state > 0
        n_active = int(np.count_nonzero(active))

        # 各类型活跃数
        per_type = [int(np.count_nonzero(a.state == k+1)) for k in range(3)]

        # 计算每个个体的最大刺激和对应 p
        stim = np.zeros((150, 3))
        for k in range(3):
            for i in range(150):
                stim[i, k] = w.sense(a.x[i], a.y[i], f"task_{k}", 2, normalize=True)
        p = response_probability(stim, a.threshold)
        max_p = p.max(axis=1)

        # 到最近任务点的距离
        all_sites = [s for sites in t.sites for s in sites]
        dists = []
        for i in range(150):
            d = min(np.hypot(a.x[i]-sx, a.y[i]-sy) for sx, sy in all_sites)
            dists.append(d)

        print(f"{tick:>4} | {n_active:>4} | {str(per_type):>13} | "
              f"{stim.max(axis=1).mean():>8.3f} | {max_p.mean():>6.3f} | {np.mean(dists):>8.2f}")

print()
print("=== 关键检查 ===")
print(f"任务点处的刺激（demand=60，邻域均值）: {60/25:.2f}")
print(f"阈值中位数: {np.median(a.threshold):.2f}")
print(f"任务点处的 p: {response_probability(60/25, 1.0):.3f}")
print()
print("如果个体在任务点处 p>0.5 但活跃数很少，说明个体没有停留在任务点附近")
