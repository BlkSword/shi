"""多任务分配任务测试。"""

import numpy as np
import pytest

from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.allocation import (
    AllocationTask,
    make_allocation_stepper,
    make_allocation_world,
)


class TestWorld:
    def test_creates_channels(self):
        w = make_allocation_world(20, 20, 3)
        assert w.channel_names == ["task_0", "task_1", "task_2"]


class TestTask:
    def _make(self, n_types=3, seed=0):
        w = make_allocation_world(30, 30, n_types)
        t = AllocationTask(w, n_types=n_types, sites_per_type=4, rng=np.random.default_rng(seed))
        t.setup()
        return w, t

    def test_setup_places_sites(self):
        w, t = self._make()
        assert len(t.sites) == 3
        assert all(len(s) == 4 for s in t.sites)

    def test_setup_fills_demand(self):
        w, t = self._make()
        assert w.total("task_0") > 0

    def test_work_reduces_demand(self):
        w, t = self._make()
        x, y = t.sites[0][0]
        before = w.read(x, y, "task_0")
        done = t.work_at(x, y, 0, 5.0)
        assert done == 5.0
        assert w.read(x, y, "task_0") == pytest.approx(before - 5.0)

    def test_work_on_empty_returns_zero(self):
        w, t = self._make()
        assert t.work_at(0.0, 0.0, 0) == 0.0

    def test_balance_perfect_when_equal(self):
        w, t = self._make()
        t.work_by_type[:] = [10.0, 10.0, 10.0]
        assert t.balance() == pytest.approx(1.0)

    def test_balance_zero_when_single_type(self):
        w, t = self._make()
        t.work_by_type[:] = [30.0, 0.0, 0.0]
        assert t.balance() == pytest.approx(0.0)

    def test_regenerate_increases_demand(self):
        w, t = self._make()
        x, y = t.sites[0][0]
        t.work_at(x, y, 0, 10.0)
        before = w.read(x, y, "task_0")
        t.regenerate()
        assert w.read(x, y, "task_0") > before

    def test_missing_channel_raises(self):
        w = make_allocation_world(10, 10, 2)
        with pytest.raises(ValueError):
            AllocationTask(w, n_types=3)


class TestStepper:
    def _setup(self, sigma=0.5, seed=0):
        w = make_allocation_world(30, 30, 3)
        t = AllocationTask(w, n_types=3, sites_per_type=4, rng=np.random.default_rng(seed))
        t.setup()
        a = AgentState(30)
        a.place_random(30.0, 30.0, np.random.default_rng(seed + 1))
        a.set_thresholds(
            sample_thresholds(30, sigma=sigma, rng=np.random.default_rng(seed + 2), n_types=3)
        )
        return w, t, a

    def test_agents_move(self):
        w, t, a = self._setup()
        stepper = make_allocation_stepper(t)
        x0 = a.x.copy()
        stepper(a, w, 0, np.random.default_rng(0))
        assert not np.array_equal(a.x, x0)

    def test_agents_stay_in_bounds(self):
        w, t, a = self._setup()
        stepper = make_allocation_stepper(t, speed=5.0, noise=5.0)
        for i in range(50):
            stepper(a, w, i, np.random.default_rng(i))
        assert np.all(a.x >= 0) and np.all(a.x <= 29.0)

    def test_work_accumulates(self):
        w, t, a = self._setup()
        stepper = make_allocation_stepper(t)
        for i in range(100):
            stepper(a, w, i, np.random.default_rng(i))
        assert t.work_done > 0

    def test_empty_population_ok(self):
        w, t, _ = self._setup()
        a = AgentState(0)
        stepper = make_allocation_stepper(t)
        stepper(a, w, 0, np.random.default_rng(0))


def _run_balance(sigma: float, seed: int, ticks: int = 800) -> tuple[float, float]:
    """跑一次，返回 (均衡度, 总工作量)。"""
    w = make_allocation_world(40, 40, 3)
    t = AllocationTask(w, n_types=3, sites_per_type=5, rng=np.random.default_rng(seed))
    t.setup()

    a = AgentState(80)
    a.place_random(40.0, 40.0, np.random.default_rng(seed + 1))
    a.set_thresholds(
        sample_thresholds(80, sigma=sigma, rng=np.random.default_rng(seed + 2), n_types=3)
    )

    engine = Engine(w, a, seed=seed, stepper=make_allocation_stepper(t), hash_every=0)
    engine.run(ticks)
    return t.balance(), t.work_done


class TestHeterogeneityEffect:
    """H3 的核心机制：异质性带来分工，分工带来均衡与产出。

    注意 sigma=0 的行为：所有个体阈值相同，它们会**随机**扎堆到
    某一个类型（取决于初始位置和噪声），所以单次运行的均衡度
    方差极大——有时碰巧均衡，有时完全失衡。
    正确的判据是「多次运行的方差」，而不是「单次的均值」。
    """

    def test_homogeneous_has_high_variance(self):
        """sigma=0：不同种子下结果差异极大（随机扎堆）。"""
        results = [_run_balance(0.0, seed) for seed in range(6)]
        balances = [b for b, _ in results]
        assert np.std(balances) > 0.15, f"同质群体应有高方差，实际 std={np.std(balances):.3f}"

    def test_heterogeneous_is_stable(self):
        """sigma=0.5：不同种子下结果稳定（分工是结构性的）。"""
        results = [_run_balance(0.5, seed) for seed in range(6)]
        balances = [b for b, _ in results]
        assert np.std(balances) < 0.15, f"异质群体应稳定，实际 std={np.std(balances):.3f}"
        assert np.mean(balances) > 0.7, f"异质群体应均衡，实际 mean={np.mean(balances):.3f}"

    def test_heterogeneous_produces_more_work(self):
        """异质群体的总产出显著高于同质群体。"""
        homo = [_run_balance(0.0, seed)[1] for seed in range(6)]
        hetero = [_run_balance(0.5, seed)[1] for seed in range(6)]
        assert np.mean(hetero) > np.mean(homo) * 2, (
            f"异质产出应显著更高：homo={np.mean(homo):.1f}, hetero={np.mean(hetero):.1f}"
        )


class TestDeterminism:
    def _run(self, seed):
        w = make_allocation_world(30, 30, 3)
        t = AllocationTask(w, n_types=3, sites_per_type=4, rng=np.random.default_rng(seed))
        t.setup()
        a = AgentState(40)
        a.place_random(30.0, 30.0, np.random.default_rng(seed + 1))
        a.set_thresholds(
            sample_thresholds(40, sigma=0.5, rng=np.random.default_rng(seed + 2), n_types=3)
        )
        e = Engine(w, a, seed=seed, stepper=make_allocation_stepper(t), hash_every=10)
        e.run(200)
        return e.hash_sequence()

    def test_same_seed_same_result(self):
        assert self._run(3) == self._run(3)

    def test_different_seed_different_result(self):
        assert self._run(3) != self._run(4)
