"""阈值个体测试。"""

import numpy as np
import pytest

from shi.core.agent import (
    STATE_CARRY,
    STATE_IDLE,
    AgentState,
    response_probability,
    sample_thresholds,
)


class TestResponseProbability:
    def test_half_at_threshold(self):
        """刺激等于阈值时，响应概率应为 0.5。"""
        assert response_probability(2.0, 2.0) == pytest.approx(0.5)

    def test_increases_with_stimulus(self):
        p_low = response_probability(1.0, 2.0)
        p_high = response_probability(4.0, 2.0)
        assert p_low < p_high

    def test_bounded(self):
        for s in [0.0, 0.1, 1.0, 10.0, 1000.0]:
            p = response_probability(s, 1.0)
            assert 0.0 <= p < 1.0

    def test_zero_stimulus_zero_probability(self):
        assert response_probability(0.0, 1.0) == 0.0

    def test_vectorized(self):
        s = np.array([1.0, 2.0, 3.0])
        theta = np.array([1.0, 1.0, 1.0])
        p = response_probability(s, theta)
        assert p.shape == (3,)
        assert p[0] == pytest.approx(0.5)

    def test_negative_stimulus_clamped_to_zero(self):
        """负刺激钳制为 0（环境场可能有数值误差产生的极小负值）。"""
        assert response_probability(-1.0, 1.0) == 0.0
        assert response_probability(-1e-12, 1.0) == 0.0

    def test_rejects_nonpositive_threshold(self):
        with pytest.raises(ValueError):
            response_probability(1.0, 0.0)


class TestSampleThresholds:
    def test_sigma_zero_is_homogeneous(self):
        t = sample_thresholds(100, median=2.0, sigma=0.0)
        assert np.all(t == 2.0)

    def test_sigma_positive_is_heterogeneous(self):
        t = sample_thresholds(1000, median=1.0, sigma=0.5)
        assert t.std() > 0.0
        assert np.all(t > 0)

    def test_median_approximately_correct(self):
        t = sample_thresholds(10000, median=3.0, sigma=0.5, rng=np.random.default_rng(1))
        assert abs(np.median(t) - 3.0) < 0.2

    def test_reproducible_with_same_rng_seed(self):
        a = sample_thresholds(100, sigma=0.5, rng=np.random.default_rng(7))
        b = sample_thresholds(100, sigma=0.5, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_larger_sigma_more_spread(self):
        a = sample_thresholds(5000, sigma=0.1, rng=np.random.default_rng(3))
        b = sample_thresholds(5000, sigma=1.0, rng=np.random.default_rng(3))
        assert b.std() > a.std()

    def test_rejects_bad_args(self):
        with pytest.raises(ValueError):
            sample_thresholds(-1)
        with pytest.raises(ValueError):
            sample_thresholds(10, median=0.0)
        with pytest.raises(ValueError):
            sample_thresholds(10, sigma=-1.0)


class TestAgentState:
    def test_initial_state(self):
        a = AgentState(10)
        assert a.n == 10
        assert np.all(a.state == STATE_IDLE)
        assert np.all(a.carrying == 0)
        assert np.all(a.threshold == 1.0)

    def test_place_random_in_bounds(self):
        a = AgentState(100)
        a.place_random(50.0, 30.0, np.random.default_rng(0))
        assert np.all(a.x >= 0) and np.all(a.x <= 50.0)
        assert np.all(a.y >= 0) and np.all(a.y <= 30.0)

    def test_place_at(self):
        a = AgentState(5)
        a.place_at(10.0, 20.0)
        assert np.all(a.x == 10.0)
        assert np.all(a.y == 20.0)

    def test_positions_shape(self):
        a = AgentState(7)
        assert a.positions.shape == (7, 2)

    def test_set_thresholds(self):
        a = AgentState(3)
        a.set_thresholds(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(a.threshold, [1.0, 2.0, 3.0])

    def test_set_thresholds_wrong_shape(self):
        a = AgentState(3)
        with pytest.raises(ValueError):
            a.set_thresholds(np.array([1.0, 2.0]))

    def test_count_by_state(self):
        a = AgentState(10)
        a.state[:3] = STATE_CARRY
        counts = a.count_by_state()
        assert counts["carry"] == 3
        assert counts["idle"] == 7

    def test_threshold_stats(self):
        a = AgentState(4)
        a.set_thresholds(np.array([1.0, 2.0, 3.0, 4.0]))
        stats = a.threshold_stats()
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0

    def test_clamp_to(self):
        a = AgentState(3)
        a.x[:] = [-5.0, 5.0, 100.0]
        a.y[:] = [-5.0, 5.0, 100.0]
        a.clamp_to(10.0, 10.0)
        np.testing.assert_array_equal(a.x, [0.0, 5.0, 10.0])

    def test_wrap_to(self):
        a = AgentState(3)
        a.x[:] = [-1.0, 5.0, 11.0]
        a.y[:] = [0.0, 0.0, 0.0]
        a.wrap_to(10.0, 10.0)
        np.testing.assert_array_equal(a.x, [9.0, 5.0, 1.0])

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError):
            AgentState(-1)
