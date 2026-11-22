"""确定性随机数测试.

核心要求：同种子必然产生相同序列，且不受调用顺序/并行度影响。
"""

import numpy as np
import pytest

from shi.core.rng import (
    RngRegistry,
    derive,
    derive_int,
    derive_normal,
    make_streams,
)


class TestMakeStreams:
    def test_same_seed_same_sequence(self):
        """同种子两次创建，序列必须逐位一致。"""
        a = make_streams(42, 5)
        b = make_streams(42, 5)
        for ga, gb in zip(a, b):
            np.testing.assert_array_equal(ga.random(100), gb.random(100))

    def test_different_seed_different_sequence(self):
        """不同种子应产生不同序列。"""
        a = make_streams(42, 3)
        b = make_streams(43, 3)
        assert not np.array_equal(a[0].random(100), b[0].random(100))

    def test_streams_are_independent(self):
        """不同个体的流互不相同。"""
        streams = make_streams(42, 10)
        firsts = [g.random() for g in streams]
        assert len(set(firsts)) == len(firsts)

    def test_stream_identity_stable_across_calls(self):
        """第 i 条流的内容只取决于 (seed, i)，与 n 无关。"""
        a = make_streams(42, 3)
        b = make_streams(42, 10)
        np.testing.assert_array_equal(a[2].random(50), b[2].random(50))

    def test_zero_streams(self):
        assert make_streams(42, 0) == []

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            make_streams(42, -1)


class TestDerive:
    def test_deterministic(self):
        """同参数多次调用结果相同。"""
        assert derive(1, 2, 3) == derive(1, 2, 3)

    def test_in_unit_interval(self):
        for i in range(1000):
            v = derive(7, i, i * 3)
            assert 0.0 <= v < 1.0

    def test_order_independent(self):
        """无状态：调用顺序不影响结果。"""
        expected = [derive(5, t, 9) for t in range(100)]
        shuffled = list(range(100))
        np.random.default_rng(0).shuffle(shuffled)
        actual = {t: derive(5, t, 9) for t in shuffled}
        for t, v in enumerate(expected):
            assert actual[t] == v

    def test_different_inputs_differ(self):
        """不同 (tick, entity) 组合应产生不同值。"""
        values = {derive(1, t, e) for t in range(20) for e in range(20)}
        assert len(values) == 400

    def test_salt_changes_result(self):
        assert derive(1, 2, 3, salt=0) != derive(1, 2, 3, salt=1)

    def test_roughly_uniform(self):
        """粗略均匀性检查：10000 个样本的均值应接近 0.5。"""
        samples = [derive(11, i, i % 7) for i in range(10000)]
        assert abs(np.mean(samples) - 0.5) < 0.02

    def test_rejects_non_int(self):
        with pytest.raises(TypeError):
            derive(1.5, 2, 3)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            derive(1 << 70, 0, 0)


class TestDeriveInt:
    def test_in_range(self):
        for i in range(500):
            v = derive_int(3, i, i, 10)
            assert 0 <= v < 10

    def test_deterministic(self):
        assert derive_int(3, 4, 5, 100) == derive_int(3, 4, 5, 100)

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            derive_int(1, 2, 3, 0)


class TestDeriveNormal:
    def test_roughly_standard_normal(self):
        samples = [derive_normal(13, i, i % 5) for i in range(20000)]
        assert abs(np.mean(samples)) < 0.05
        assert abs(np.std(samples) - 1.0) < 0.05


class TestRngRegistry:
    def test_same_entity_same_stream(self):
        reg = RngRegistry(42)
        a = reg.stream(7).random(10)
        b = reg.stream(7).random(10)
        # 同一条流，连续取值，所以 a 和 b 不同但可复现
        reg2 = RngRegistry(42)
        np.testing.assert_array_equal(a, reg2.stream(7).random(10))
        np.testing.assert_array_equal(b, reg2.stream(7).random(10))

    def test_different_entities_differ(self):
        reg = RngRegistry(42)
        assert reg.stream(1).random() != reg.stream(2).random()

    def test_lazy_creation(self):
        reg = RngRegistry(42)
        assert len(reg) == 0
        reg.stream(5)
        assert len(reg) == 1
        assert 5 in reg
        assert 6 not in reg

    def test_different_seeds_differ(self):
        a = RngRegistry(1).stream(0).random(10)
        b = RngRegistry(2).stream(0).random(10)
        assert not np.array_equal(a, b)
