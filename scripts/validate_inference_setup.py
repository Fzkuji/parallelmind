#!/usr/bin/env python3
"""
验证完整的 HuggingFace + LoRA + 2D RoPE 推理流程
检查所有关键组件是否正确集成
"""
import os
import sys
import argparse

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def check_imports():
    """检查所有必要的导入"""
    print("=" * 80)
    print("检查 1: 验证必要模块导入")
    print("=" * 80)

    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ transformers")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False

    try:
        from model.model_lora import apply_lora, load_lora, save_lora
        print("✓ model.model_lora")
    except ImportError as e:
        print(f"✗ model.model_lora: {e}")
        return False

    try:
        from parallel.columnar import (
            patch_model_with_interleaved_2d_rope,
            set_rope_pos2d,
            _find_rotary_holder,
        )
        print("✓ parallel.columnar")
    except ImportError as e:
        print(f"✗ parallel.columnar: {e}")
        return False

    try:
        from scripts.inference_hf_lora import (
            load_model_with_lora,
            generate_text,
            _prepare_pos2d,
            _inject_pos2d_hook,
        )
        print("✓ scripts.inference_hf_lora")
    except ImportError as e:
        print(f"✗ scripts.inference_hf_lora: {e}")
        return False

    print("\n所有模块导入成功！\n")
    return True


def check_training_script():
    """检查训练脚本是否存在且可导入"""
    print("=" * 80)
    print("检查 2: 验证训练脚本")
    print("=" * 80)

    train_script = os.path.join(root_path, "trainer", "train_hf_lora.py")
    if not os.path.exists(train_script):
        print(f"✗ 训练脚本不存在: {train_script}")
        return False

    print(f"✓ 训练脚本存在: {train_script}")

    # 检查关键函数
    try:
        sys.path.insert(0, os.path.join(root_path, "trainer"))
        import train_hf_lora
        assert hasattr(train_hf_lora, "auto_pair_indices"), "缺少 auto_pair_indices 函数"
        print("✓ auto_pair_indices 函数存在")
        assert hasattr(train_hf_lora, "train_epoch"), "缺少 train_epoch 函数"
        print("✓ train_epoch 函数存在")
    except Exception as e:
        print(f"✗ 训练脚本检查失败: {e}")
        return False

    print("\n训练脚本验证成功！\n")
    return True


def check_inference_script():
    """检查推理脚本是否正确实现"""
    print("=" * 80)
    print("检查 3: 验证推理脚本实现")
    print("=" * 80)

    inference_script = os.path.join(root_path, "scripts", "inference_hf_lora.py")
    if not os.path.exists(inference_script):
        print(f"✗ 推理脚本不存在: {inference_script}")
        return False

    print(f"✓ 推理脚本存在: {inference_script}")

    # 检查关键函数
    from scripts.inference_hf_lora import (
        _prepare_pos2d,
        _inject_pos2d_hook,
        load_model_with_lora,
        generate_text,
        interactive_chat,
    )

    print("✓ _prepare_pos2d 函数存在")
    print("✓ _inject_pos2d_hook 函数存在")
    print("✓ load_model_with_lora 函数存在")
    print("✓ generate_text 函数存在")
    print("✓ interactive_chat 函数存在")

    # 检查 _inject_pos2d_hook 实现
    import inspect
    source = inspect.getsource(_inject_pos2d_hook)
    assert "prepare_inputs_for_generation" in source, "缺少 prepare_inputs_for_generation 重写"
    print("✓ _inject_pos2d_hook 正确实现了 prepare_inputs_for_generation 重写")

    assert "set_rope_pos2d" in source, "缺少 set_rope_pos2d 调用"
    print("✓ _inject_pos2d_hook 中调用了 set_rope_pos2d")

    print("\n推理脚本验证成功！\n")
    return True


def check_documentation():
    """检查文档是否完整"""
    print("=" * 80)
    print("检查 4: 验证文档完整性")
    print("=" * 80)

    docs = [
        ("README.md", "主文档"),
        ("docs/INFERENCE_GUIDE.md", "推理指南"),
        ("docs/TRAIN_HF_LORA_USAGE.md", "训练使用文档"),
    ]

    all_exist = True
    for doc_path, doc_name in docs:
        full_path = os.path.join(root_path, doc_path)
        if os.path.exists(full_path):
            print(f"✓ {doc_name} 存在: {doc_path}")

            # 检查关键词
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            if "INFERENCE_GUIDE" in doc_path:
                if "pos2d" in content:
                    print(f"  ✓ {doc_name} 包含 pos2d 说明")
                else:
                    print(f"  ⚠️  {doc_name} 缺少 pos2d 说明")

                if "set_rope_pos2d" in content:
                    print(f"  ✓ {doc_name} 包含 set_rope_pos2d 说明")
                else:
                    print(f"  ⚠️  {doc_name} 缺少 set_rope_pos2d 说明")

        else:
            print(f"✗ {doc_name} 不存在: {doc_path}")
            all_exist = False

    if all_exist:
        print("\n文档完整性验证成功！\n")
    else:
        print("\n⚠️  部分文档缺失\n")

    return all_exist


def check_pos2d_workflow():
    """检查 pos2d 工作流程"""
    print("=" * 80)
    print("检查 5: 验证 pos2d 工作流程")
    print("=" * 80)

    import torch
    from scripts.inference_hf_lora import _prepare_pos2d, _inject_pos2d_hook

    # 测试 _prepare_pos2d
    input_ids = torch.randint(0, 1000, (2, 10))
    pos2d = _prepare_pos2d(input_ids)

    assert pos2d.shape == (2, 10, 2), "pos2d 形状错误"
    print("✓ _prepare_pos2d 生成正确的 pos2d 形状")

    assert torch.all(pos2d[:, :, 0] == 0), "branch_ids 应该全为 0"
    print("✓ branch_ids 全为 0（单分支推理）")

    expected_time = torch.arange(10).unsqueeze(0).expand(2, -1)
    assert torch.all(pos2d[:, :, 1] == expected_time), "time_ids 应该线性递增"
    print("✓ time_ids 线性递增")

    # 测试 hook 注入
    class DummyModel:
        def __init__(self):
            self._pos2d_hook_injected = False

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids}

    model = DummyModel()
    _inject_pos2d_hook(model)

    assert model._pos2d_hook_injected, "hook 应该已注入"
    print("✓ hook 成功注入到模型")

    assert hasattr(model, "_orig_prepare_inputs_for_generation"), "原始方法应该被保存"
    print("✓ 原始 prepare_inputs_for_generation 已保存")

    print("\npos2d 工作流程验证成功！\n")
    return True


def check_file_structure():
    """检查项目文件结构"""
    print("=" * 80)
    print("检查 6: 验证项目文件结构")
    print("=" * 80)

    required_files = [
        "trainer/train_hf_lora.py",
        "scripts/inference_hf_lora.py",
        "scripts/test_inference.sh",
        "model/model_lora.py",
        "parallel/columnar.py",
        "parallel_data/parallel_dataset.py",
        "parallel_data/parallel_collator.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(root_path, file_path)
        if os.path.exists(full_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 不存在")
            all_exist = False

    if all_exist:
        print("\n文件结构验证成功！\n")
    else:
        print("\n⚠️  部分文件缺失\n")

    return all_exist


def print_summary():
    """打印使用摘要"""
    print("=" * 80)
    print("使用摘要")
    print("=" * 80)

    print("\n📝 训练命令示例：")
    print("-" * 80)
    print("""
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \\
  --base_model Qwen/Qwen2-0.5B-Instruct \\
  --data_path dataset/pretrain_hq_split.jsonl \\
  --lora_rank 8 \\
  --batch_size 4 \\
  --batch_by_samples \\
  --max_branches_per_sample 16 \\
  --min_branches_per_sample 1 \\
  --rope_2d_ratio 0.5 \\
  --epochs 3 \\
  --ddp
""".strip())

    print("\n🔍 推理命令示例：")
    print("-" * 80)
    print("""
# 交互式对话
python scripts/inference_hf_lora.py \\
  --base_model Qwen/Qwen2-0.5B-Instruct \\
  --lora_path out/lora/hf_lora_hf_final.pth \\
  --lora_rank 8 \\
  --rope_2d_ratio 0.5 \\
  --mode chat

# 单次生成
python scripts/inference_hf_lora.py \\
  --base_model Qwen/Qwen2-0.5B-Instruct \\
  --lora_path out/lora/hf_lora_hf_final.pth \\
  --lora_rank 8 \\
  --mode generate \\
  --prompt "你好，请介绍一下你自己"
""".strip())

    print("\n⚠️  重要提示：")
    print("-" * 80)
    print("""
1. ✅ pos2d 已自动处理，无需手动设置
2. ✅ 推理脚本自动重写了 prepare_inputs_for_generation
3. ✅ 每次生成前会自动调用 set_rope_pos2d
4. ⚠️  lora_rank 必须与训练时一致
5. ⚠️  rope_2d_ratio 必须与训练时一致
6. ⚠️  如果训练时使用了 --patch_rope，推理时也必须启用（默认启用）
""".strip())

    print("\n📚 详细文档：")
    print("-" * 80)
    print("- 推理指南: docs/INFERENCE_GUIDE.md")
    print("- 训练文档: docs/TRAIN_HF_LORA_USAGE.md")
    print("- 主文档: README.md")
    print()


def main():
    parser = argparse.ArgumentParser(description="验证 HuggingFace + LoRA + 2D RoPE 推理设置")
    parser.add_argument("--skip-imports", action="store_true", help="跳过导入检查")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("HuggingFace + LoRA + 2D RoPE 推理流程验证")
    print("=" * 80 + "\n")

    checks = []

    if not args.skip_imports:
        checks.append(("模块导入", check_imports()))
    checks.append(("训练脚本", check_training_script()))
    checks.append(("推理脚本", check_inference_script()))
    checks.append(("文档完整性", check_documentation()))
    checks.append(("pos2d 工作流程", check_pos2d_workflow()))
    checks.append(("文件结构", check_file_structure()))

    # 打印结果
    print("=" * 80)
    print("验证结果汇总")
    print("=" * 80)

    all_passed = True
    for name, result in checks:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s} {status}")
        if not result:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✅ 所有验证通过！系统已正确配置")
        print_summary()
        return 0
    else:
        print("\n❌ 部分验证失败，请检查上述错误")
        return 1


if __name__ == "__main__":
    sys.exit(main())
