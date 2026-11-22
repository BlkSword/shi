"""觅食任务测试。"""

import numpy as np
import pytest

from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.core.world import PheromoneField
from shi.tasks.foraging import ForagingTask, make_foraging_stepper


def _make_task(seed=0, n_resources=20):
    world = PheromoneField(40, 40)
    task = ForagingTask(
        world=world,
        nest_x=20.0,
        nest_y=20.0,
        n_resources=n_resources,
        rng=np.random.default_rng(seed),
    )
    return world, task


class TestForagingTask:
    def test_requires_channels(self):
        world = PheromoneField(10, 10, channels={"a": 0.9})
        with pytest.raises(ValueError):
            ForagingTask(world, 5.0, 5.0)

    def test_setup_places_resources(self):
        world, task = _make_task()
        task.setup()
        assert len(task.resources) == 20
        assert world.total("supply") > 0

    def test_setup_marks_nest(self):
        world, task = _make_task()
        task.setup()
        assert world.read(20.0, 20.0, "order") > 0

    def test_collect_reduces_supply(self):
        world, task = _make_task()
        task.setup()
        x, y = task.resources[0]
        before = world.read(x, y, "supply")
        assert task.collect_at(x, y) is True
        assert world.read(x, y, "supply") == pytest.approx(before - 1.0)

    def test_collect_fails_on_empty(self):
        world, task = _make_task()
        task.setup()
        assert task.collect_at(0.0, 0.0) is False

    def test_deliver_at_nest(self):
        world, task = _make_task()
        task.setup()
        assert task.deliver_at(20.0, 20.0) is True
        assert task.deliver_at(0.0, 0.0) is False

    def test_coverage_starts_zero(self):
        world, task = _make_task()
        task.setup()
        assert task.coverage() == 0.0

    def test_coverage_increases_after_collect(self):
        world, task = _make_task()
        task.setup()
        x, y = task.resources[0]
        task.collect_at(x, y)
        assert task.coverage() > 0.0

    def test_maintain_nest_marker(self):
        world, task = _make_task()
        task.setup()
        before = world.read(20.0, 20.0, "order")
        world.step()
        task.maintain_nest_marker()
        assert world.read(20.0, 20.0, "order") > before * 0.9


class TestForagingStepper:
    def test_agents_move(self):
        world, task = _make_task()
        task.setup()
        agents = AgentState(20)
        agents.place_random(40.0, 40.0, np.random.default_rng(1))
        agents.set_thresholds(np.full(20, 1.0))

        stepper = make_foraging_stepper(task)
        x0 = agents.x.copy()
        stepper(agents, world, 0, np.random.default_rng(2))
        assert not np.array_equal(agents.x, x0)

    def test_agents_stay_in_bounds(self):
        world, task = _make_task()
        task.setup()
        agents = AgentState(30)
        agents.place_random(40.0, 40.0, np.random.default_rng(1))
        agents.set_thresholds(np.full(30, 1.0))

        stepper = make_foraging_stepper(task, speed=5.0, noise=5.0)
        for t in range(100):
            stepper(agents, world, t, np.random.default_rng(t))
        assert np.all(agents.x >= 0) and np.all(agents.x <= 39.0)
        assert np.all(agents.y >= 0) and np.all(agents.y <= 39.0)

    def test_empty_population_ok(self):
        world, task = _make_task()
        task.setup()
        agents = AgentState(0)
        stepper = make_foraging_stepper(task)
        stepper(agents, world, 0, np.random.default_rng(0))  # 不应抛错


class TestForagingDeterminism:
    def _run(self, seed: int, n_ticks: int = 100) -> list[str]:
        world = PheromoneField(40, 40)
        task = ForagingTask(
            world=world, nest_x=20.0, nest_y=20.0, n_resources=20,
            rng=np.random.default_rng(seed),
        )
        task.setup()

        agents = AgentState(50)
        agents.place_random(40.0, 40.0, np.random.default_rng(seed))
        agents.set_thresholds(np.full(50, 1.0))

        engine = Engine(
            world=world, agents=agents, seed=seed,
            stepper=make_foraging_stepper(task), hash_every=10,
        )
        engine.run(n_ticks)
        return engine.hash_sequence()

    def test_same_seed_same_result(self):
        assert self._run(5) == self._run(5)

    def test_different_seed_different_result(self):
        assert self._run(5) != self._run(6)
