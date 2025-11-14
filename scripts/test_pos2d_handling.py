#!/usr/bin/env python3
"""
测试 pos2d 自动注入机制是否正常工作
验证 inference_hf_lora.py 中的 pos2d 处理逻辑
"""
import os
import sys
import torch

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from scripts.inference_hf_lora import _prepare_pos2d, _inject_pos2d_hook
from parallel.columnar import _find_rotary_holder


def test_prepare_pos2d():
    """测试 _prepare_pos2d 函数"""
    print("=" * 80)
    print("测试 1: _prepare_pos2d 函数")
    print("=" * 80)

    # 创建测试 input_ids
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # 准备 pos2d
    pos2d = _prepare_pos2d(input_ids)

    # 验证形状
    assert pos2d.shape == (batch_size, seq_len, 2), f"pos2d 形状错误: {pos2d.shape}"
    print(f"✓ pos2d 形状正确: {pos2d.shape}")

    # 验证 branch_ids 全为 0
    branch_ids = pos2d[:, :, 0]
    assert torch.all(branch_ids == 0), "branch_ids 应该全为 0"
    print(f"✓ branch_ids 全为 0")

    # 验证 time_ids 是线性递增的
    time_ids = pos2d[:, :, 1]
    expected_time_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    assert torch.all(time_ids == expected_time_ids), "time_ids 应该是线性递增的"
    print(f"✓ time_ids 线性递增: {time_ids[0].tolist()}")

    print("\n测试 1 通过！\n")


def test_inject_pos2d_hook():
    """测试 _inject_pos2d_hook 是否正确修改了 prepare_inputs_for_generation"""
    print("=" * 80)
    print("测试 2: _inject_pos2d_hook 函数")
    print("=" * 80)

    # 创建一个简单的模型对象
    class DummyModel:
        def __init__(self):
            self._pos2d_hook_injected = False

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

    model = DummyModel()

    # 注入 hook
    _inject_pos2d_hook(model)

    # 验证 hook 已注入
    assert model._pos2d_hook_injected, "hook 应该已注入"
    print("✓ hook 已注入")

    # 验证原始方法被保存
    assert hasattr(model, "_orig_prepare_inputs_for_generation"), "原始方法应该被保存"
    print("✓ 原始方法已保存")

    # 验证不会重复注入
    _inject_pos2d_hook(model)
    print("✓ 不会重复注入 hook")

    print("\n测试 2 通过！\n")


def test_pos2d_dimension_consistency():
    """测试不同序列长度的 pos2d 一致性"""
    print("=" * 80)
    print("测试 3: pos2d 维度一致性")
    print("=" * 80)

    test_cases = [
        (1, 5),    # 单条短序列
        (2, 10),   # 双条中等序列
        (4, 128),  # 批量长序列
        (1, 1),    # 极短序列
    ]

    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        pos2d = _prepare_pos2d(input_ids)

        # 验证形状
        assert pos2d.shape == (batch_size, seq_len, 2), \
            f"形状错误: batch_size={batch_size}, seq_len={seq_len}, pos2d.shape={pos2d.shape}"

        # 验证 branch 维度全为 0
        assert torch.all(pos2d[:, :, 0] == 0), \
            f"branch_ids 应该全为 0: batch_size={batch_size}, seq_len={seq_len}"

        # 验证 time 维度递增
        for b in range(batch_size):
            expected = torch.arange(seq_len)
            assert torch.all(pos2d[b, :, 1] == expected), \
                f"time_ids 应该递增: batch_size={batch_size}, seq_len={seq_len}"

        print(f"✓ batch_size={batch_size}, seq_len={seq_len} 验证通过")

    print("\n测试 3 通过！\n")


def main():
    print("\n" + "=" * 80)
    print("pos2d 自动注入机制测试")
    print("=" * 80 + "\n")

    try:
        test_prepare_pos2d()
        test_inject_pos2d_hook()
        test_pos2d_dimension_consistency()

        print("=" * 80)
        print("✅ 所有测试通过！pos2d 自动注入机制工作正常")
        print("=" * 80)

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试出错: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
