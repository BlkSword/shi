"""目标聚集任务测试（G2 实验用）。"""

import numpy as np
import pytest

from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.tasks.aggregation import (
    AggregationTask,
    make_aggregation_stepper,
    make_aggregation_world,
    shape_boundary,
    shape_feedback,
    shape_potential,
)


class TestWorld:
    def test_creates_field_channel(self):
        w = make_aggregation_world(20, 20)
        assert w.channel_names == ["field"]


class TestTask:
    def _make(self):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        return w, t

    def test_requires_field_channel(self):
        from shi.core.world import PheromoneField
        w = PheromoneField(10, 10, channels={"a": 1.0})
        with pytest.raises(ValueError):
            AggregationTask(w, 5.0, 5.0)

    def test_mean_distance(self):
        w, t = self._make()
        a = AgentState(4)
        a.x[:] = [20.0, 20.0, 20.0, 20.0]
        a.y[:] = [20.0, 20.0, 20.0, 20.0]
        assert t.mean_distance(a) == pytest.approx(0.0)

    def test_aggregation_score_range(self):
        w, t = self._make()
        a = AgentState(10)
        a.place_random(40.0, 40.0, np.random.default_rng(0))
        s = t.aggregation_score(a)
        assert 0.0 <= s <= 1.0

    def test_is_blocked_false_by_default(self):
        w, t = self._make()
        assert t.is_blocked(10.0, 10.0) is False


class TestShaping:
    def _make(self):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        return w, t

    def test_potential_creates_peak_at_target(self):
        w, t = self._make()
        shape_potential(t, amplitude=100.0, length_scale=10.0)
        at_target = w.read(20.0, 20.0, "field")
        far = w.read(0.0, 0.0, "field")
        assert at_target > far
        assert at_target == pytest.approx(100.0, rel=0.01)

    def test_potential_with_repellers(self):
        w, t = self._make()
        shape_potential(t, amplitude=100.0, length_scale=10.0,
                        n_repellers=3, repeller_amplitude=-50.0,
                        rng=np.random.default_rng(0))
        # 目标处仍应是最高点
        assert w.read(20.0, 20.0, "field") > 0

    def test_feedback_creates_gradient(self):
        w, t = self._make()
        shape_feedback(t, base_level=1.0, gradient_strength=10.0)
        near = w.read(20.0, 20.0, "field")
        far = w.read(0.0, 0.0, "field")
        assert near > far

    def test_boundary_creates_blocked_mask(self):
        w, t = self._make()
        shape_boundary(t, funnel_width=5.0)
        assert t.blocked is not None
        assert t.blocked.shape == (40, 40)
        # 目标附近不应被阻挡
        assert not t.is_blocked(20.0, 20.0)


class TestStepper:
    def _setup(self, shape="potential"):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        if shape == "potential":
            shape_potential(t, amplitude=100.0, length_scale=10.0)
        elif shape == "feedback":
            shape_feedback(t)
        elif shape == "boundary":
            shape_boundary(t)
        a = AgentState(30)
        a.place_random(40.0, 40.0, np.random.default_rng(0))
        return w, t, a

    def test_agents_move(self):
        w, t, a = self._setup()
        stepper = make_aggregation_stepper(t)
        x0 = a.x.copy()
        stepper(a, w, 0, np.random.default_rng(0))
        assert not np.array_equal(a.x, x0)

    def test_agents_stay_in_bounds(self):
        w, t, a = self._setup()
        stepper = make_aggregation_stepper(t, speed=5.0)
        for i in range(100):
            stepper(a, w, i, np.random.default_rng(i))
        assert np.all(a.x >= 0) and np.all(a.x <= 39.0)

    def test_empty_population_ok(self):
        w, t, _ = self._setup()
        a = AgentState(0)
        stepper = make_aggregation_stepper(t)
        stepper(a, w, 0, np.random.default_rng(0))


class TestAggregationBehavior:
    """端到端：势能景观应该能把个体聚集到目标。"""

    def test_potential_aggregates(self):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        shape_potential(t, amplitude=100.0, length_scale=10.0)

        a = AgentState(50)
        a.place_random(40.0, 40.0, np.random.default_rng(0))
        d0 = t.mean_distance(a)

        e = Engine(w, a, seed=0, stepper=make_aggregation_stepper(t), hash_every=0)
        e.run(500)

        d1 = t.mean_distance(a)
        assert d1 < d0 * 0.2, f"应该显著聚集：{d0:.2f} -> {d1:.2f}"

    def test_baseline_does_not_aggregate(self):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        w.clear()

        a = AgentState(50)
        a.place_random(40.0, 40.0, np.random.default_rng(0))
        d0 = t.mean_distance(a)

        e = Engine(w, a, seed=0, stepper=make_aggregation_stepper(t), hash_every=0)
        e.run(500)

        d1 = t.mean_distance(a)
        assert d1 > d0 * 0.8, f"空环境不应聚集：{d0:.2f} -> {d1:.2f}"


class TestDeterminism:
    def _run(self, seed):
        w = make_aggregation_world(40, 40)
        t = AggregationTask(w, target_x=20.0, target_y=20.0)
        shape_potential(t, amplitude=100.0, length_scale=10.0)
        a = AgentState(30)
        a.place_random(40.0, 40.0, np.random.default_rng(seed))
        e = Engine(w, a, seed=seed, stepper=make_aggregation_stepper(t), hash_every=10)
        e.run(200)
        return e.hash_sequence()

    def test_same_seed_same_result(self):
        assert self._run(3) == self._run(3)

    def test_different_seed_different_result(self):
        assert self._run(3) != self._run(4)
