"""确定性测试：P0 阶段的硬性验收标准.

验收条件：同一个种子跑 100 次，状态哈希序列 100% 一致。
"""

import pytest


@pytest.mark.skip(reason="P0 待实现")
def test_same_seed_same_hash():
    """同种子两次运行，哈希序列必须完全一致。"""
    raise NotImplementedError


@pytest.mark.skip(reason="P0 待实现")
def test_different_seed_different_hash():
    """不同种子应产生不同哈希（防止哈希实现退化为常量）。"""
    raise NotImplementedError
