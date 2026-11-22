"""环境场测试。"""

import numpy as np
import pytest

from shi.core.world import ChannelSpec, PheromoneField


class TestConstruction:
    def test_default_channels(self):
        f = PheromoneField(10, 10)
        assert f.channel_names == ["threat", "supply", "order"]
        assert f.n_channels == 3
        assert f.shape == (10, 10)

    def test_custom_channels_dict(self):
        f = PheromoneField(5, 5, channels={"a": 0.9, "b": 0.5})
        assert f.channel_names == ["a", "b"]

    def test_custom_channel_specs(self):
        specs = [ChannelSpec("x", decay=0.9, diffusion=0.1)]
        f = PheromoneField(5, 5, channels=specs)
        assert f.channel_names == ["x"]

    def test_rejects_bad_size(self):
        with pytest.raises(ValueError):
            PheromoneField(0, 10)

    def test_rejects_empty_channels(self):
        with pytest.raises(ValueError):
            PheromoneField(5, 5, channels={})

    def test_rejects_bad_decay(self):
        with pytest.raises(ValueError):
            ChannelSpec("x", decay=1.5)

    def test_rejects_bad_diffusion(self):
        with pytest.raises(ValueError):
            ChannelSpec("x", diffusion=0.5)


class TestDepositRead:
    def test_deposit_then_read(self):
        f = PheromoneField(10, 10)
        f.deposit(3.5, 4.5, "supply", 2.0)
        assert f.read(3.5, 4.5, "supply") == 2.0

    def test_deposit_accumulates(self):
        f = PheromoneField(10, 10)
        f.deposit(3.0, 3.0, "supply", 1.0)
        f.deposit(3.0, 3.0, "supply", 2.0)
        assert f.read(3.0, 3.0, "supply") == 3.0

    def test_channels_are_independent(self):
        f = PheromoneField(10, 10)
        f.deposit(1.0, 1.0, "supply", 5.0)
        assert f.read(1.0, 1.0, "threat") == 0.0

    def test_negative_deposit_erases(self):
        f = PheromoneField(10, 10)
        f.deposit(2.0, 2.0, "supply", 5.0)
        f.deposit(2.0, 2.0, "supply", -3.0)
        assert f.read(2.0, 2.0, "supply") == 2.0

    def test_out_of_bounds_clamps(self):
        f = PheromoneField(10, 10)
        f.deposit(-5.0, -5.0, "supply", 1.0)
        assert f.read(0.0, 0.0, "supply") == 1.0
        f.deposit(100.0, 100.0, "supply", 1.0)
        assert f.read(9.0, 9.0, "supply") == 1.0

    def test_unknown_channel_raises(self):
        f = PheromoneField(10, 10)
        with pytest.raises(KeyError):
            f.deposit(1.0, 1.0, "nope", 1.0)

    def test_deposit_many_handles_duplicates(self):
        """同一格多次写入必须累加，不能覆盖。"""
        f = PheromoneField(10, 10)
        xs = np.array([1.0, 1.0, 1.0])
        ys = np.array([1.0, 1.0, 1.0])
        amounts = np.array([1.0, 2.0, 3.0])
        f.deposit_many(xs, ys, "supply", amounts)
        assert f.read(1.0, 1.0, "supply") == 6.0


class TestSense:
    def test_sense_single_cell(self):
        f = PheromoneField(10, 10)
        f.deposit(5.0, 5.0, "supply", 3.0)
        assert f.sense(5.0, 5.0, "supply", radius=0) == 3.0

    def test_sense_includes_neighbors(self):
        f = PheromoneField(10, 10)
        f.deposit(5.0, 5.0, "supply", 1.0)
        f.deposit(6.0, 5.0, "supply", 2.0)
        assert f.sense(5.0, 5.0, "supply", radius=1) == 3.0

    def test_sense_radius_excludes_far(self):
        f = PheromoneField(10, 10)
        f.deposit(5.0, 5.0, "supply", 1.0)
        f.deposit(9.0, 9.0, "supply", 100.0)
        assert f.sense(5.0, 5.0, "supply", radius=1) == 1.0

    def test_sense_clips_at_boundary(self):
        f = PheromoneField(10, 10)
        f.deposit(0.0, 0.0, "supply", 1.0)
        assert f.sense(0.0, 0.0, "supply", radius=5) == 1.0

    def test_negative_radius_raises(self):
        f = PheromoneField(10, 10)
        with pytest.raises(ValueError):
            f.sense(1.0, 1.0, "supply", radius=-1)


class TestGradient:
    def test_gradient_points_uphill(self):
        f = PheromoneField(10, 10)
        # 在 x 方向造一个递增的场
        for x in range(10):
            f.deposit(float(x), 5.0, "supply", float(x))
        gx, gy = f.gradient(5.0, 5.0, "supply")
        assert gx > 0
        assert gy == 0.0

    def test_gradient_zero_on_flat(self):
        f = PheromoneField(10, 10)
        gx, gy = f.gradient(5.0, 5.0, "supply")
        assert gx == 0.0 and gy == 0.0


class TestStep:
    def test_decay_reduces_concentration(self):
        f = PheromoneField(10, 10, channels={"a": 0.5})
        f.deposit(5.0, 5.0, "a", 8.0)
        f.step()
        assert f.read(5.0, 5.0, "a") == pytest.approx(4.0)

    def test_decay_is_per_channel(self):
        f = PheromoneField(10, 10, channels={"fast": 0.5, "slow": 1.0})
        f.deposit(5.0, 5.0, "fast", 8.0)
        f.deposit(5.0, 5.0, "slow", 8.0)
        f.step()
        assert f.read(5.0, 5.0, "fast") == pytest.approx(4.0)
        assert f.read(5.0, 5.0, "slow") == pytest.approx(8.0)

    def test_decay_converges_to_zero(self):
        f = PheromoneField(10, 10, channels={"a": 0.5})
        f.deposit(5.0, 5.0, "a", 1.0)
        for _ in range(100):
            f.step()
        assert f.read(5.0, 5.0, "a") < 1e-20

    def test_diffusion_spreads(self):
        f = PheromoneField(10, 10, channels=[ChannelSpec("a", decay=1.0, diffusion=0.2)])
        f.deposit(5.0, 5.0, "a", 100.0)
        f.step()
        # 中心变少，邻居变多
        assert f.read(5.0, 5.0, "a") < 100.0
        assert f.read(6.0, 5.0, "a") > 0.0

    def test_diffusion_conserves_total(self):
        """零通量边界下，扩散不改变总量。"""
        f = PheromoneField(10, 10, channels=[ChannelSpec("a", decay=1.0, diffusion=0.2)])
        f.deposit(5.0, 5.0, "a", 100.0)
        before = f.total("a")
        f.step()
        assert f.total("a") == pytest.approx(before, rel=1e-12)


class TestDeterminism:
    def test_same_operations_same_state(self):
        """同样的操作序列必须产生逐位相同的状态。"""
        def build():
            f = PheromoneField(20, 20)
            for i in range(50):
                f.deposit(float(i % 20), float(i // 20), "supply", float(i))
                f.step()
            return f

        a, b = build(), build()
        np.testing.assert_array_equal(a.data, b.data)

    def test_snapshot_is_copy(self):
        f = PheromoneField(5, 5)
        f.deposit(1.0, 1.0, "supply", 1.0)
        snap = f.snapshot()
        f.deposit(1.0, 1.0, "supply", 1.0)
        assert snap[1, 1, 1] == 1.0  # 快照不受后续修改影响


class TestMisc:
    def test_clear(self):
        f = PheromoneField(5, 5)
        f.deposit(1.0, 1.0, "supply", 5.0)
        f.clear()
        assert f.total("supply") == 0.0

    def test_total(self):
        f = PheromoneField(10, 10)
        f.deposit(1.0, 1.0, "supply", 2.0)
        f.deposit(2.0, 2.0, "supply", 3.0)
        assert f.total("supply") == 5.0
