"""延迟觅食任务测试（H2 实验用）。"""

import numpy as np
import pytest

from shi.core.agent import AgentState, sample_thresholds
from shi.core.engine import Engine
from shi.tasks.delayed_foraging import (
    DelayedForagingTask,
    make_delayed_foraging_stepper,
    make_delayed_foraging_world,
)


class TestWorld:
    def test_creates_channels(self):
        w = make_delayed_foraging_world(20, 20, n_channels=4)
        assert w.n_channels == 4
        assert w.channel_names == ["mem_0", "mem_1", "mem_2", "mem_3"]

    def test_precision_stored(self):
        w = make_delayed_foraging_world(20, 20, n_channels=2, precision_bits=16)
        assert w.precision_bits == 16


class TestTask:
    def _make(self, n_channels=4, n_types=4, seed=0):
        w = make_delayed_foraging_world(40, 40, n_channels=n_channels)
        t = DelayedForagingTask(
            w, n_resources=20, n_types=n_types, cycle=100, scout_ratio=0.3,
            rng=np.random.default_rng(seed),
        )
        t.setup()
        return w, t

    def test_setup_places_resources(self):
        w, t = self._make()
        assert len(t.resources) == 20
        assert all(len(r) == 3 for r in t.resources)  # (x, y, type)

    def test_phase_alternates(self):
        w, t = self._make()
        t.tick = 0
        assert t.is_scouting()
        t.tick = 50
        assert t.is_harvesting()

    def test_target_type_rotates(self):
        w, t = self._make(n_types=4)
        t.tick = 0
        assert t.target_type() == 0
        t.tick = 100
        assert t.target_type() == 1

    def test_cannot_collect_while_scouting(self):
        w, t = self._make()
        t.tick = 0  # scout phase
        x, y, rt = t.resources[0]
        assert t.collect_at(x, y) is False

    def test_can_collect_target_type_while_harvesting(self):
        w, t = self._make(n_types=4)
        t.tick = 50  # harvest phase, target=0
        # 找一个类型 0 的资源
        target_res = [r for r in t.resources if r[2] == 0]
        if target_res:
            x, y, _ = target_res[0]
            assert t.collect_at(x, y) is True

    def test_wrong_type_increments_counter(self):
        w, t = self._make(n_types=4)
        t.tick = 50  # harvest, target=0
        wrong = [r for r in t.resources if r[2] != 0]
        if wrong:
            x, y, _ = wrong[0]
            before = t.wrong_visits
            t.collect_at(x, y)
            assert t.wrong_visits == before + 1

    def test_write_memory_respects_budget(self):
        w = make_delayed_foraging_world(20, 20, n_channels=2)
        t = DelayedForagingTask(w, n_resources=5, write_budget=0.5)
        t.setup()
        actual = t.write_memory(5.0, 5.0, 0, 10.0)
        assert actual == 0.5

    def test_efficiency(self):
        w, t = self._make()
        assert t.efficiency() == 0.0
        t.collected = 10
        assert t.efficiency() == pytest.approx(0.5)


class TestStepper:
    def _setup(self, n_channels=4, complexity=2, seed=0):
        w = make_delayed_foraging_world(40, 40, n_channels=n_channels)
        t = DelayedForagingTask(
            w, n_resources=20, n_types=4, cycle=100, scout_ratio=0.3,
            rng=np.random.default_rng(seed),
        )
        t.setup()
        a = AgentState(30)
        a.place_random(40.0, 40.0, np.random.default_rng(seed + 1))
        a.set_thresholds(
            sample_thresholds(30, sigma=0.5, rng=np.random.default_rng(seed + 2))
        )
        return w, t, a

    def test_agents_move(self):
        w, t, a = self._setup()
        stepper = make_delayed_foraging_stepper(t)
        x0 = a.x.copy()
        stepper(a, w, 0, np.random.default_rng(0))
        assert not np.array_equal(a.x, x0)

    def test_agents_stay_in_bounds(self):
        w, t, a = self._setup()
        stepper = make_delayed_foraging_stepper(t, speed=5.0, search_speed=5.0)
        for i in range(100):
            stepper(a, w, i, np.random.default_rng(i))
        assert np.all(a.x >= 0) and np.all(a.x <= 39.0)

    def test_all_complexities_run(self):
        for comp in [1, 2, 3]:
            w, t, a = self._setup(complexity=comp)
            stepper = make_delayed_foraging_stepper(t, rule_complexity=comp)
            for i in range(50):
                stepper(a, w, i, np.random.default_rng(i))
            assert np.all(np.isfinite(a.x))

    def test_empty_population_ok(self):
        w, t, _ = self._setup()
        a = AgentState(0)
        stepper = make_delayed_foraging_stepper(t)
        stepper(a, w, 0, np.random.default_rng(0))


class TestDeterminism:
    def _run(self, seed):
        w = make_delayed_foraging_world(40, 40, n_channels=4)
        t = DelayedForagingTask(
            w, n_resources=20, n_types=4, cycle=100, scout_ratio=0.3,
            rng=np.random.default_rng(seed),
        )
        t.setup()
        a = AgentState(30)
        a.place_random(40.0, 40.0, np.random.default_rng(seed + 1))
        a.set_thresholds(
            sample_thresholds(30, sigma=0.5, rng=np.random.default_rng(seed + 2))
        )
        e = Engine(w, a, seed=seed, stepper=make_delayed_foraging_stepper(t), hash_every=10)
        e.run(200)
        return e.hash_sequence()

    def test_same_seed_same_result(self):
        assert self._run(3) == self._run(3)

    def test_different_seed_different_result(self):
        assert self._run(3) != self._run(4)
