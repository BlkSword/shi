"""主循环与指标测试。"""

import numpy as np
import pytest

from shi.core.agent import AgentState
from shi.core.engine import Engine
from shi.core.metrics import MetricsRecorder
from shi.core.world import PheromoneField


def _noop_stepper(agents, world, tick, rng):
    """什么都不做的 stepper，用于测试引擎本身。"""
    pass


def _drift_stepper(agents, world, tick, rng):
    """让所有个体向右漂移，用于测试状态变化。"""
    agents.x += 1.0


class TestMetricsRecorder:
    def test_record_and_series(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        rec.record(tick=1, a=2.0)
        assert rec.series("a") == [1.0, 2.0]
        assert rec.n_ticks == 2

    def test_columns_locked_after_first(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        with pytest.raises(ValueError):
            rec.record(tick=1, b=2.0)

    def test_last(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        rec.record(tick=1, a=5.0)
        assert rec.last("a") == 5.0

    def test_summary(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0, b=2.0)
        rec.record(tick=1, a=3.0, b=4.0)
        assert rec.summary() == {"a": 3.0, "b": 4.0}

    def test_to_csv(self, tmp_path):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        rec.record(tick=1, a=2.0)
        out = tmp_path / "sub" / "results.csv"
        rec.to_csv(out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "tick,a" in text
        assert "0,1.0" in text

    def test_empty_csv_raises(self, tmp_path):
        with pytest.raises(ValueError):
            MetricsRecorder().to_csv(tmp_path / "x.csv")

    def test_unknown_column_raises(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        with pytest.raises(KeyError):
            rec.series("nope")

    def test_clear(self):
        rec = MetricsRecorder()
        rec.record(tick=0, a=1.0)
        rec.clear()
        assert rec.n_ticks == 0
        assert rec.columns == []


class TestEngine:
    def _make(self, n=10, seed=42, hash_every=10, stepper=None):
        world = PheromoneField(20, 20)
        agents = AgentState(n)
        agents.place_random(20.0, 20.0, np.random.default_rng(0))
        return Engine(
            world=world,
            agents=agents,
            seed=seed,
            stepper=stepper or _noop_stepper,
            hash_every=hash_every,
        )

    def test_initial_state(self):
        e = self._make()
        assert e.tick == 0
        assert e.hashes == []

    def test_step_increments_tick(self):
        e = self._make()
        e.step()
        assert e.tick == 1

    def test_run_n_ticks(self):
        e = self._make()
        e.run(50)
        assert e.tick == 50

    def test_hash_recorded_at_interval(self):
        e = self._make(hash_every=10)
        e.run(50)
        assert len(e.hashes) == 5
        assert [t for t, _ in e.hashes] == [10, 20, 30, 40, 50]

    def test_hash_every_zero_disables(self):
        e = self._make(hash_every=0)
        e.run(50)
        assert e.hashes == []

    def test_stepper_is_called(self):
        e = self._make(stepper=_drift_stepper)
        x0 = e.agents.x.copy()
        e.run(5)
        np.testing.assert_array_equal(e.agents.x, x0 + 5.0)

    def test_metrics_recorded(self):
        e = self._make()
        e.run(100, record_every=20)
        assert e.metrics.n_ticks == 5
        assert e.metrics.series("tick") == [20, 40, 60, 80, 100]

    def test_metrics_with_extra_columns(self):
        e = self._make()
        e.run(20, record_every=10, config="baseline")
        assert "config" in e.metrics.columns

    def test_reset(self):
        e = self._make()
        e.run(30, record_every=10)
        e.reset()
        assert e.tick == 0
        assert e.metrics.n_ticks == 0
        assert e.hashes == []

    def test_current_hash_deterministic(self):
        e1 = self._make()
        e2 = self._make()
        assert e1.current_hash() == e2.current_hash()

    def test_hash_changes_after_movement(self):
        e = self._make(stepper=_drift_stepper)
        h0 = e.current_hash()
        e.step()
        assert e.current_hash() != h0


class TestEngineDeterminism:
    """引擎层面的确定性：这是 P0 的核心验收标准。"""

    def _run(self, seed: int, n_ticks: int = 200) -> list[str]:
        world = PheromoneField(30, 30)
        agents = AgentState(50)
        agents.place_random(30.0, 30.0, np.random.default_rng(seed))

        def stepper(ag, w, tick, rng):
            # 用 rng 驱动，确保随机性也参与确定性检验
            ag.x += rng.normal(0.0, 0.1, ag.n)
            ag.y += rng.normal(0.0, 0.1, ag.n)
            ag.clamp_to(29.0, 29.0)
            w.deposit_many(ag.x, ag.y, "supply", np.full(ag.n, 0.1))

        engine = Engine(world, agents, seed=seed, stepper=stepper, hash_every=10)
        engine.run(n_ticks)
        return engine.hash_sequence()

    def test_same_seed_same_hash_sequence(self):
        """同种子两次运行，哈希序列必须完全一致。"""
        a = self._run(seed=123)
        b = self._run(seed=123)
        assert a == b
        assert len(a) == 20

    def test_different_seed_different_hash_sequence(self):
        a = self._run(seed=1)
        b = self._run(seed=2)
        assert a != b

    def test_repeated_runs_stable(self):
        """连续跑 10 次，每次结果都相同。"""
        baseline = self._run(seed=7)
        for _ in range(10):
            assert self._run(seed=7) == baseline
