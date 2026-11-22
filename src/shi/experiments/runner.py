"""单次实验执行器.

约定
----
每次实验在 experiments/ 下创建一个目录：

    experiments/2026-11-25_e1_rule_complexity/
      config.json
      results.csv
      summary.json

notes.md 必须手写：假设、观察、结论。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.core.world import ChannelSpec, PheromoneField
from shi.tasks.allocation import (
    AllocationTask,
    make_allocation_stepper,
    make_allocation_world,
)
from shi.tasks.foraging import ForagingTask, make_foraging_stepper

__all__ = [
    "ForagingConfig",
    "AllocationConfig",
    "run_foraging",
    "run_allocation",
    "save_experiment",
]


@dataclass
class ForagingConfig:
    """觅食实验配置。

    所有影响结果的参数都在这里，没有魔法数字散落在代码里。
    """

    # 世界
    width: int = 60
    height: int = 60
    n_resources: int = 80
    resource_amount: float = 10.0

    # 个体
    n_agents: int = 200
    threshold_median: float = 1.0
    threshold_sigma: float = 0.0   # 异质性参数（H3 的自变量）
    speed: float = 0.5
    noise: float = 0.3
    sense_radius: int = 2

    # 环境可写性（H2 的自变量）
    n_channels: int = 3            # 1 / 3 / 8
    supply_decay: float = 0.98
    diffusion: float = 0.0

    # 运行
    n_ticks: int = 2000
    record_every: int = 20
    hash_every: int = 100
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_channels(cfg: ForagingConfig) -> list[ChannelSpec]:
    """按 n_channels 构造通道配置。

    这是 H2 的自变量：环境的信息容量。

    前三个通道是觅食任务必需的（supply / recruit / order），
    超过 3 个时补充辅助通道，模拟更丰富的环境信息。
    """
    base = [
        ChannelSpec("supply", decay=1.0, diffusion=0.0),
        ChannelSpec("recruit", decay=0.97, diffusion=0.10),
        ChannelSpec("order", decay=0.99, diffusion=0.20),
    ]
    if cfg.n_channels <= 3:
        return base[: max(1, cfg.n_channels)]

    extra = [
        ChannelSpec(f"aux{i}", decay=0.95, diffusion=0.05)
        for i in range(cfg.n_channels - 3)
    ]
    return base + extra


def run_foraging(cfg: ForagingConfig) -> dict[str, Any]:
    """跑一次觅食实验，返回指标摘要。"""
    t_start = time.perf_counter()

    channels = _build_channels(cfg)
    world = PheromoneField(
        cfg.width, cfg.height, channels=channels, diffusion=cfg.diffusion
    )

    task = ForagingTask(
        world=world,
        nest_x=cfg.width / 2.0,
        nest_y=cfg.height / 2.0,
        n_resources=cfg.n_resources,
        resource_amount=cfg.resource_amount,
        rng=np.random.default_rng(cfg.seed),
    )
    task.setup()

    agents = AgentState(cfg.n_agents)
    agents.place_random(
        float(cfg.width), float(cfg.height), np.random.default_rng(cfg.seed + 1)
    )
    agents.set_thresholds(
        sample_thresholds(
            cfg.n_agents,
            median=cfg.threshold_median,
            sigma=cfg.threshold_sigma,
            rng=np.random.default_rng(cfg.seed + 2),
        )
    )

    stepper = make_foraging_stepper(
        task, speed=cfg.speed, noise=cfg.noise, sense_radius=cfg.sense_radius
    )

    engine = Engine(
        world=world,
        agents=agents,
        seed=cfg.seed,
        stepper=stepper,
        hash_every=cfg.hash_every,
    )

    # 主循环：每 record_every tick 记录一次（0 表示不记录）
    for _ in range(cfg.n_ticks):
        engine.step()
        if cfg.record_every > 0 and engine.tick % cfg.record_every == 0:
            engine.metrics.record(
                tick=engine.tick,
                coverage=task.coverage(),
                collected=task.collected,
                carrying=int(np.count_nonzero(agents.carrying)),
                supply_total=world.total("supply"),
            )

    elapsed = time.perf_counter() - t_start

    return {
        "config": cfg.to_dict(),
        "elapsed_sec": elapsed,
        "final_coverage": task.coverage(),
        "total_collected": task.collected,
        "hash_sequence": engine.hash_sequence(),
        "metrics": engine.metrics.to_rows(),
    }


def save_experiment(
    result: dict[str, Any],
    out_dir: str | Path,
    name: str = "run",
) -> Path:
    """把实验结果落盘。"""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    (p / f"{name}_config.json").write_text(
        json.dumps(result["config"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "elapsed_sec": result["elapsed_sec"],
        "final_coverage": result["final_coverage"],
        "total_collected": result["total_collected"],
        "n_hashes": len(result["hash_sequence"]),
        "first_hash": result["hash_sequence"][0] if result["hash_sequence"] else None,
        "last_hash": result["hash_sequence"][-1] if result["hash_sequence"] else None,
    }
    (p / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 指标 CSV
    rows = result["metrics"]
    if rows:
        cols = list(rows[0].keys())
        lines = [",".join(cols)]
        for row in rows:
            lines.append(",".join(str(row[c]) for c in cols))
        (p / f"{name}_results.csv").write_text("\n".join(lines), encoding="utf-8")

    return p


@dataclass
class AllocationConfig:
    """多任务分配实验配置。

    这是检验 H3（异质性）的主实验。与觅食任务不同，
    多任务设定下异质性有明确的作用机制：分工。
    """

    # 世界
    width: int = 60
    height: int = 60
    n_types: int = 3
    sites_per_type: int = 6
    demand: float = 20.0
    regen: float = 0.05

    # 个体
    n_agents: int = 150
    threshold_median: float = 1.0
    threshold_sigma: float = 0.0   # H3 的自变量
    speed: float = 0.6
    noise: float = 0.2
    sense_radius: int = 2
    work_amount: float = 1.0

    # 环境突变（测适应时间）
    shift_at: int = 0              # 0 表示不突变
    shift_type: int = 0            # 突变时把哪个类型的需求清零

    # 运行
    n_ticks: int = 1500
    record_every: int = 20
    hash_every: int = 100
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_allocation(cfg: AllocationConfig) -> dict[str, Any]:
    """跑一次多任务分配实验。

    如果 cfg.shift_at > 0，会在该 tick 触发环境突变：
    把 shift_type 类型的所有任务点需求清零，
    然后观察系统多久恢复（适应时间）。
    """
    t_start = time.perf_counter()

    world = make_allocation_world(cfg.width, cfg.height, cfg.n_types)
    task = AllocationTask(
        world,
        n_types=cfg.n_types,
        sites_per_type=cfg.sites_per_type,
        demand=cfg.demand,
        regen=cfg.regen,
        rng=np.random.default_rng(cfg.seed),
    )
    task.setup()

    agents = AgentState(cfg.n_agents)
    agents.place_random(
        float(cfg.width), float(cfg.height), np.random.default_rng(cfg.seed + 1)
    )
    agents.set_thresholds(
        sample_thresholds(
            cfg.n_agents,
            median=cfg.threshold_median,
            sigma=cfg.threshold_sigma,
            rng=np.random.default_rng(cfg.seed + 2),
            n_types=cfg.n_types,
        )
    )

    stepper = make_allocation_stepper(
        task,
        speed=cfg.speed,
        noise=cfg.noise,
        sense_radius=cfg.sense_radius,
        work_amount=cfg.work_amount,
    )

    engine = Engine(
        world=world,
        agents=agents,
        seed=cfg.seed,
        stepper=stepper,
        hash_every=cfg.hash_every,
    )

    shifted = False
    shift_tick = 0

    for _ in range(cfg.n_ticks):
        # 环境突变：把某个类型的需求清零
        if cfg.shift_at > 0 and not shifted and engine.tick >= cfg.shift_at:
            # 精确清零：读取当前值再减去它，避免产生负浓度
            ch = f"task_{cfg.shift_type}"
            for x, y in task.sites[cfg.shift_type]:
                cur = world.read(x, y, ch)
                if cur > 0:
                    world.deposit(x, y, ch, -cur)
            shifted = True
            shift_tick = engine.tick

        engine.step()

        if cfg.record_every > 0 and engine.tick % cfg.record_every == 0:
            engine.metrics.record(
                tick=engine.tick,
                work_done=task.work_done,
                completion=task.completion(),
                balance=task.balance(),
                active=int(np.count_nonzero(agents.state > 0)),
            )

    elapsed = time.perf_counter() - t_start

    return {
        "config": cfg.to_dict(),
        "elapsed_sec": elapsed,
        "work_done": task.work_done,
        "completion": task.completion(),
        "balance": task.balance(),
        "work_by_type": task.work_by_type.tolist(),
        "shifted": shifted,
        "shift_tick": shift_tick,
        "hash_sequence": engine.hash_sequence(),
        "metrics": engine.metrics.to_rows(),
    }
