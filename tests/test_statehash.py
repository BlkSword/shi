"""状态哈希测试.

核心要求：相同状态必然产生相同哈希；不同状态必然产生不同哈希。
"""

import numpy as np
import pytest

from shi.core.statehash import hash_arrays, hash_bytes, state_hash


class TestHashBytes:
    def test_deterministic(self):
        assert hash_bytes(b"abc", b"def") == hash_bytes(b"abc", b"def")

    def test_different_content_differs(self):
        assert hash_bytes(b"abc") != hash_bytes(b"abd")

    def test_chunk_boundary_matters(self):
        """分块方式不同应产生不同哈希（防止拼接歧义）。"""
        assert hash_bytes(b"ab", b"c") != hash_bytes(b"a", b"bc")

    def test_tick_participates(self):
        assert hash_bytes(b"x", tick=1) != hash_bytes(b"x", tick=2)

    def test_length_is_32_hex(self):
        h = hash_bytes(b"x")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


class TestHashArrays:
    def test_deterministic(self):
        a = np.arange(100, dtype=np.float64)
        assert hash_arrays([a]) == hash_arrays([a])

    def test_dtype_matters(self):
        """float32 和 float64 即使数值相同也应产生不同哈希。"""
        a32 = np.arange(10, dtype=np.float32)
        a64 = np.arange(10, dtype=np.float64)
        assert hash_arrays([a32]) != hash_arrays([a64])

    def test_shape_matters(self):
        a = np.arange(12, dtype=np.float64)
        b = a.reshape(3, 4)
        assert hash_arrays([a]) != hash_arrays([b])

    def test_non_contiguous_input(self):
        """非连续数组应先转连续再哈希，结果与连续版本一致。"""
        a = np.arange(20, dtype=np.float64).reshape(4, 5)
        view = a[:, ::2]
        assert hash_arrays([view]) == hash_arrays([np.ascontiguousarray(view)])

    def test_negative_zero_differs_from_zero(self):
        """-0.0 与 0.0 位模式不同，哈希应不同（我们要逐位一致）。"""
        a = np.array([0.0], dtype=np.float64)
        b = np.array([-0.0], dtype=np.float64)
        assert hash_arrays([a]) != hash_arrays([b])

    def test_order_of_arrays_matters(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert hash_arrays([a, b]) != hash_arrays([b, a])


class TestStateHash:
    def test_deterministic(self):
        pos = np.array([[1.0, 2.0], [3.0, 4.0]])
        env = np.zeros((10, 10), dtype=np.float64)
        assert state_hash(pos, env, 0) == state_hash(pos, env, 0)

    def test_position_change_detected(self):
        env = np.zeros((10, 10), dtype=np.float64)
        p1 = np.array([[1.0, 2.0]])
        p2 = np.array([[1.0, 2.0000001]])
        assert state_hash(p1, env, 0) != state_hash(p2, env, 0)

    def test_env_change_detected(self):
        pos = np.array([[1.0, 2.0]])
        e1 = np.zeros((10, 10), dtype=np.float64)
        e2 = e1.copy()
        e2[3, 4] = 1e-12
        assert state_hash(pos, e1, 0) != state_hash(pos, e2, 0)

    def test_tick_change_detected(self):
        pos = np.array([[1.0, 2.0]])
        env = np.zeros((5, 5), dtype=np.float64)
        assert state_hash(pos, env, 0) != state_hash(pos, env, 1)
