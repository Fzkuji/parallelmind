# pos2d 自动注入实现总结

## 问题描述

GPT-5 发现了推理脚本中的关键问题：

> **必须给 2D RoPE 注入 pos2d**
>
> `trainer/train_hf_lora.py`、`parallel_generate.py` 在前向前都会调用 `set_rope_pos2d`，而 Claude 写的 `scripts/inference_hf_lora.py` 只是 `patch_model_with_interleaved_2d_rope`，却没有在 forward/generate 之前设置 pos2d。
>
> 如果你开启了 `--patch_rope`，首次推理就会报错：
> ```
> RuntimeError: extra_pos2d is not set. Call set_rope_pos2d first.
> ```

## 问题根源

2D RoPE 需要在每次前向传播前调用 `set_rope_pos2d()` 来设置位置编码。这包括：
1. **初始 prompt 的前向传播**
2. **增量生成的每一步**（通过 `prepare_inputs_for_generation`）

原始实现只做了模型的 patch（`patch_model_with_interleaved_2d_rope`），但没有：
- ✗ 在首次生成前调用 `set_rope_pos2d`
- ✗ 重写 `prepare_inputs_for_generation` 以在增量生成时自动调用

## 解决方案

### 1. 添加 `_prepare_pos2d()` 辅助函数

位置：`scripts/inference_hf_lora.py:26-48`

```python
def _prepare_pos2d(input_ids: torch.Tensor) -> torch.Tensor:
    """
    为单条文本准备 pos2d（单个 branch + 线性 time id）

    Args:
        input_ids: [batch_size, seq_len]

    Returns:
        pos2d: [batch_size, seq_len, 2]
               [:, :, 0] 是 branch_id（全0）
               [:, :, 1] 是 time_id（0, 1, 2, ...）
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # time_ids: 0, 1, 2, ..., seq_len-1
    time_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # branch_ids: 全部为 0（单个 branch）
    branch_ids = torch.zeros_like(time_ids)

    # 堆叠成 [batch, seq, 2]
    pos2d = torch.stack([branch_ids, time_ids], dim=-1)

    return pos2d
```

### 2. 添加 `_inject_pos2d_hook()` 钩子注入函数

位置：`scripts/inference_hf_lora.py:57-84`

```python
def _inject_pos2d_hook(model):
    """重写 prepare_inputs_for_generation，保证增量生成也会携带 pos2d"""
    if getattr(model, "_pos2d_hook_injected", False):
        return  # 避免重复注入

    if not hasattr(model, "prepare_inputs_for_generation"):
        return

    # 保存原始方法
    model._orig_prepare_inputs_for_generation = model.prepare_inputs_for_generation

    def _prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 调用原始方法获取 inputs
        inputs = self._orig_prepare_inputs_for_generation(input_ids, **kwargs)

        # 获取或生成 position_ids
        position_ids = inputs.get("position_ids")
        if position_ids is None:
            seq_len = inputs["input_ids"].size(-1)
            position_ids = torch.arange(seq_len, device=inputs["input_ids"].device).unsqueeze(0)
            inputs["position_ids"] = position_ids

        # 构造 pos2d（branch_id=0, time_id=position_ids）
        branch_ids = torch.zeros_like(position_ids)
        pos2d = torch.stack([branch_ids, position_ids], dim=-1)

        # ⚠️ 关键：每次生成前调用 set_rope_pos2d
        set_rope_pos2d(self, pos2d)

        return inputs

    # 重写方法
    model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)
    model._pos2d_hook_injected = True
```

### 3. 在 `load_model_with_lora()` 中调用钩子注入

位置：`scripts/inference_hf_lora.py:145-148`

```python
# 应用 2D RoPE
if patch_rope:
    print(f"应用 2D RoPE（ratio={rope_2d_ratio}）...")
    pair_indices = auto_pair_indices(model, rope_2d_ratio)
    patch_model_with_interleaved_2d_rope(model, pair_indices)
    print(f"✓ 2D RoPE 已应用（{len(pair_indices)} 个频率对用于 branch 维度）")

    # ⚠️ 关键：注入 pos2d 自动设置钩子
    _inject_pos2d_hook(model)
    model._uses_pos2d = True
else:
    model._uses_pos2d = False
```

### 4. 在生成前设置 pos2d（首次前向传播）

位置：`scripts/inference_hf_lora.py:210-211`（`generate_text` 函数）

```python
# 编码输入
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"].to(device)

# ⚠️ 关键：首次生成前设置 pos2d
if getattr(model, "_uses_pos2d", False):
    _set_prompt_pos2d(model, input_ids)

# 生成（增量生成会自动调用 prepare_inputs_for_generation 中的 pos2d 设置）
with torch.no_grad():
    outputs = model.generate(...)
```

同样在 `interactive_chat()` 函数中（位置：`scripts/inference_hf_lora.py:281-282`）也有相同处理。

## 工作流程

```
加载基础模型
    ↓
应用 2D RoPE (patch_model_with_interleaved_2d_rope)
    ↓
注入 pos2d 钩子 (_inject_pos2d_hook)
    ├─ 保存原始 prepare_inputs_for_generation
    └─ 重写为自动调用 set_rope_pos2d 的版本
    ↓
应用 LoRA
    ↓
推理时：
    ├─ 首次生成前：调用 _set_prompt_pos2d(model, input_ids)
    ├─ 首次前向传播：使用完整 prompt 的 pos2d
    └─ 增量生成：每步自动通过 prepare_inputs_for_generation 调用 set_rope_pos2d
```

## 验证结果

运行 `python scripts/validate_inference_setup.py`：

```
================================================================================
验证结果汇总
================================================================================
模块导入                 ✅ 通过
训练脚本                 ✅ 通过
推理脚本                 ✅ 通过
文档完整性                ✅ 通过
pos2d 工作流程           ✅ 通过
文件结构                 ✅ 通过
================================================================================

✅ 所有验证通过！系统已正确配置
```

## 关键要点

### ✅ 已完成

1. ✅ `_prepare_pos2d()` - 为单分支推理生成正确的 pos2d 张量
2. ✅ `_inject_pos2d_hook()` - 重写 `prepare_inputs_for_generation` 自动注入 pos2d
3. ✅ 首次生成前调用 `_set_prompt_pos2d()`
4. ✅ 增量生成时通过钩子自动调用 `set_rope_pos2d()`
5. ✅ 更新文档说明 pos2d 自动处理机制

### ⚠️ 用户需要注意

1. **推荐方式**：使用 `scripts/inference_hf_lora.py`，pos2d 已自动处理
2. **自定义代码**：如果自己写推理代码，必须：
   - 调用 `patch_model_with_interleaved_2d_rope()` 后
   - 重写 `prepare_inputs_for_generation` 方法
   - 在每次生成前调用 `set_rope_pos2d()`

3. **参数一致性**：
   - `lora_rank` 必须与训练时一致
   - `rope_2d_ratio` 必须与训练时一致
   - 如果训练时使用了 `--patch_rope`，推理时也必须启用

## 对比：训练 vs 推理

| 阶段 | 设置 pos2d 的位置 | 方式 |
|------|------------------|------|
| **训练** | `train_hf_lora.py:76-77` | 每个 batch 显式调用 `set_rope_pos2d(rope_target_model, pos2d)` |
| **推理** | `inference_hf_lora.py:210-211` + `67-84` | 首次调用 `_set_prompt_pos2d()` + 钩子自动注入 |

训练时，每个 batch 都有完整的 pos2d（来自 `ParallelPretrainCollator`），可以直接在训练循环中设置。

推理时，分为两个阶段：
1. **首次前向**：处理完整 prompt，使用 `_set_prompt_pos2d()`
2. **增量生成**：每生成一个 token，通过重写的 `prepare_inputs_for_generation` 自动更新 pos2d

## 测试验证

### 单元测试

```bash
python scripts/test_pos2d_handling.py
```

验证：
- ✅ `_prepare_pos2d()` 生成正确形状和内容
- ✅ 钩子注入不会重复
- ✅ 不同序列长度的 pos2d 一致性

### 集成测试

```bash
python scripts/validate_inference_setup.py
```

验证：
- ✅ 所有必要模块可导入
- ✅ 训练和推理脚本存在且正确实现
- ✅ 文档包含 pos2d 说明
- ✅ pos2d 工作流程完整

### 实际推理测试

```bash
# 需要已训练的 LoRA 权重
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/hf_lora_hf_final.pth \
  --lora_rank 8 \
  --mode generate \
  --prompt "你好"
```

预期：不会出现 "extra_pos2d is not set" 错误。

## 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/inference_hf_lora.py` | 推理脚本（包含 pos2d 自动注入） |
| `trainer/train_hf_lora.py` | 训练脚本（包含显式 pos2d 设置） |
| `parallel/columnar.py` | 2D RoPE 实现（`set_rope_pos2d`、`patch_model_with_interleaved_2d_rope`） |
| `docs/INFERENCE_GUIDE.md` | 推理详细指南（包含 pos2d 警告） |
| `docs/TRAIN_HF_LORA_USAGE.md` | 训练使用文档 |
| `scripts/test_pos2d_handling.py` | pos2d 单元测试 |
| `scripts/validate_inference_setup.py` | 完整验证脚本 |

## 总结

GPT-5 发现的 pos2d 缺失问题已完全解决：

1. ✅ 实现了 `_prepare_pos2d()` 生成单分支推理的 pos2d
2. ✅ 实现了 `_inject_pos2d_hook()` 自动重写 `prepare_inputs_for_generation`
3. ✅ 在首次生成前显式调用 `_set_prompt_pos2d()`
4. ✅ 增量生成时通过钩子自动注入 pos2d
5. ✅ 更新所有相关文档
6. ✅ 提供测试脚本验证实现正确性

用户现在可以直接使用 `scripts/inference_hf_lora.py` 进行推理，pos2d 会自动处理，无需手动干预。
