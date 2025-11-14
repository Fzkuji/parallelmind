# HuggingFace + LoRA 推理指南

## ⚠️ 重要提示：2D RoPE 需要 pos2d

如果训练时使用了 `--patch_rope`（2D RoPE），推理时必须：
1. 同样启用 `--patch_rope`
2. 确保 `--rope_2d_ratio` 与训练时一致
3. **自动处理 pos2d**：我们的推理脚本已经自动重写了 `prepare_inputs_for_generation`，会在每次生成前调用 `set_rope_pos2d`

如果不设置 pos2d，会遇到错误：
```
RuntimeError: extra_pos2d is not set. Call set_rope_pos2d first.
```

我们的推理脚本已经处理了这个问题，无需手动干预。

## 快速开始

训练完成后，LoRA 权重保存在：
- `out/lora/<lora_name>_<model_tag>.pth`（训练中保存）
- `out/lora/<lora_name>_<model_tag>_final.pth`（最终权重）

## 推理方法

> ⚠️ 如果训练时启用了 2D RoPE（`--patch_rope`），推理也必须保持相同的 `--rope_2d_ratio`。  
> 本脚本会自动为 prompt 和增量生成注入 `pos2d`，无需额外手动处理。

### 方法 1：交互式对话（推荐）

```bash
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/qwen2_parallel_lora_hf_final.pth \
  --lora_rank 8 \
  --rope_2d_ratio 0.5 \
  --mode chat
```

进入交互式对话界面：
```
User: 你好，请介绍一下你自己
Assistant: 你好！我是一个AI助手...

User: quit
再见！
```

### 方法 2：单次生成

```bash
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/qwen2_parallel_lora_hf_final.pth \
  --lora_rank 8 \
  --mode generate \
  --prompt "请写一首关于春天的诗："
```

### 方法 3：Python 代码调用（推荐使用封装函数）

**简单方式（推荐）**：
```python
from scripts.inference_hf_lora import load_model_with_lora, generate_text

# 加载模型（自动处理 2D RoPE 的 pos2d）
model, tokenizer, patch_rope = load_model_with_lora(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_path="out/lora/qwen2_parallel_lora_hf_final.pth",
    lora_rank=8,
    rope_2d_ratio=0.5,
    patch_rope=True,
)

# 生成（自动处理 pos2d）
response = generate_text(model, tokenizer, "你好")
```

**手动方式（需要处理 pos2d）**：
```python
import torch
import types
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_lora import apply_lora, load_lora
from parallel.columnar import (
    patch_model_with_interleaved_2d_rope,
    set_rope_pos2d,
)

# 1. 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    trust_remote_code=True
)

# 2. 应用 2D RoPE（如果训练时使用了）
from scripts.inference_hf_lora import auto_pair_indices
pair_indices = auto_pair_indices(model, ratio=0.5)
patch_model_with_interleaved_2d_rope(model, pair_indices)

# ⚠️ 关键：重写 prepare_inputs_for_generation 以自动设置 pos2d
def _prepare_inputs_for_generation(self, input_ids, **kwargs):
    inputs = self._orig_prepare_inputs_for_generation(input_ids, **kwargs)
    pos_ids = inputs.get("position_ids")
    if pos_ids is None:
        seq_len = inputs["input_ids"].size(-1)
        pos_ids = torch.arange(seq_len, device=inputs["input_ids"].device).unsqueeze(0)
        inputs["position_ids"] = pos_ids
    branch_ids = torch.zeros_like(pos_ids)
    pos2d = torch.stack([branch_ids, pos_ids], dim=-1)
    set_rope_pos2d(self, pos2d)
    return inputs

model._orig_prepare_inputs_for_generation = model.prepare_inputs_for_generation
model.prepare_inputs_for_generation = types.MethodType(_prepare_inputs_for_generation, model)

# 3. 应用 LoRA
apply_lora(model, rank=8)
load_lora(model, "out/lora/qwen2_parallel_lora_hf_final.pth")

# 4. 生成（现在会自动处理 pos2d）
model.eval()
prompt = "你好，请介绍一下你自己"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 参数说明

### 必填参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--base_model` | HuggingFace 基础模型 | Qwen/Qwen2-0.5B-Instruct |
| `--lora_path` | LoRA 权重文件路径 | out/lora/xxx_final.pth |

### 模型相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora_rank` | 8 | 必须与训练时一致 |
| `--rope_2d_ratio` | 0.5 | 必须与训练时一致 |
| `--no_patch_rope` | False | 如果训练时没用 2D RoPE，需设置此项 |
| `--device` | cuda | 运行设备 |
| `--dtype` | bfloat16 | 数据类型 |

### 推理模式

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `--mode` | chat / generate | chat=交互式对话，generate=单次生成 |
| `--prompt` | - | generate 模式的输入提示 |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_new_tokens` | 512 | 最大生成 token 数 |
| `--temperature` | 0.7 | 温度（0.1-1.0，越小越确定） |
| `--top_p` | 0.9 | Nucleus sampling |
| `--top_k` | 50 | Top-K sampling |
| `--repetition_penalty` | 1.1 | 重复惩罚 |

## 完整示例

### 示例 1：医疗问答模型

```bash
# 训练（假设已完成）
torchrun --nproc_per_node 8 trainer/train_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --data_path dataset/medical_data.jsonl \
  --lora_name medical_lora \
  --lora_rank 8

# 推理
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2-0.5B-Instruct \
  --lora_path out/lora/medical_lora_hf_final.pth \
  --lora_rank 8 \
  --mode chat \
  --temperature 0.3
```

### 示例 2：中文写作助手

```bash
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --lora_path out/lora/writer_lora_hf_final.pth \
  --lora_rank 16 \
  --mode generate \
  --prompt "请写一篇关于人工智能发展的文章" \
  --max_new_tokens 1024 \
  --temperature 0.8
```

### 示例 3：代码生成

```bash
python scripts/inference_hf_lora.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path out/lora/code_lora_hf_final.pth \
  --lora_rank 32 \
  --mode generate \
  --prompt "写一个 Python 函数计算斐波那契数列" \
  --temperature 0.2
```

## 批量推理脚本

如果需要批量处理，可以创建批量推理脚本：

```python
# batch_inference.py
import torch
from scripts.inference_hf_lora import load_model_with_lora, generate_text

# 加载模型（只需加载一次）
model, tokenizer = load_model_with_lora(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_path="out/lora/qwen2_parallel_lora_hf_final.pth",
    lora_rank=8,
)

# 批量推理
test_prompts = [
    "请介绍一下你自己",
    "什么是人工智能？",
    "请写一首诗",
]

for prompt in test_prompts:
    print(f"\n{'=' * 80}")
    print(f"Prompt: {prompt}")
    response = generate_text(model, tokenizer, prompt)
    print(f"Response: {response}")
```

运行：
```bash
python batch_inference.py
```

## 性能优化

### 1. 使用 Flash Attention

确保安装了 Flash Attention：
```bash
pip install flash-attn --no-build-isolation
```

### 2. 使用 8-bit 量化（节省显存）

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
)
```

### 3. 多 GPU 推理

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",  # 自动分配到多个 GPU
    torch_dtype=torch.bfloat16,
)
```

## 常见问题

### Q1: 出现 "extra_pos2d is not set" 错误

**错误信息**：
```
RuntimeError: extra_pos2d is not set. Call set_rope_pos2d first.
```

**原因**：使用了 2D RoPE（`--patch_rope`）但没有设置 pos2d

**解决**：
1. **推荐**：使用我们提供的推理脚本 `scripts/inference_hf_lora.py`，它已经自动处理了 pos2d
2. 如果自己写代码，需要重写 `prepare_inputs_for_generation` 方法（见上面的手动方式示例）

### Q2: 出现 "未找到 rotary_emb" 错误

**原因**：某些模型的 RoPE 结构不同

**解决**：添加 `--no_patch_rope` 跳过 2D RoPE 应用
```bash
python scripts/inference_hf_lora.py \
  --base_model xxx \
  --lora_path xxx \
  --no_patch_rope
```

### Q3: 生成结果质量不好

**可能原因**：
1. 训练不充分
2. 生成参数不合适

**解决方案**：
```bash
# 调整 temperature（降低随机性）
--temperature 0.3

# 调整 top_p（减少候选词）
--top_p 0.8

# 增加重复惩罚
--repetition_penalty 1.2
```

### Q4: 显存不足

**解决方案**：
1. 使用较小的模型
2. 使用 8-bit 量化
3. 减小 `--max_new_tokens`
4. 使用 CPU 推理（慢但可用）

```bash
python scripts/inference_hf_lora.py \
  --device cpu \
  --dtype float32
```

### Q5: LoRA 权重不匹配

**错误信息**：`size mismatch for ...`

**原因**：训练时的 `lora_rank` 与推理时不一致

**解决**：确保 `--lora_rank` 与训练时完全一致

### Q6: 如何验证 LoRA 是否正确加载？

在加载后检查模型参数：
```python
# 检查是否有 LoRA 层
for name, module in model.named_modules():
    if hasattr(module, 'lora'):
        print(f"✓ {name} 有 LoRA 层")
        print(f"  A shape: {module.lora.A.weight.shape}")
        print(f"  B shape: {module.lora.B.weight.shape}")
```

## 部署建议

### 本地 API 服务

使用 FastAPI 部署：
```python
# serve_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from scripts.inference_hf_lora import load_model_with_lora

app = FastAPI()

# 加载模型（启动时加载一次）
model, tokenizer = load_model_with_lora(
    base_model="Qwen/Qwen2-0.5B-Instruct",
    lora_path="out/lora/xxx_final.pth",
    lora_rank=8,
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7

@app.post("/generate")
def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}
```

启动服务：
```bash
uvicorn serve_api:app --host 0.0.0.0 --port 8000
```

调用 API：
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_new_tokens": 100}'
```

## 参数配置表

### 不同场景的推荐参数

| 场景 | temperature | top_p | top_k | repetition_penalty |
|------|-------------|-------|-------|--------------------|
| 事实问答 | 0.1-0.3 | 0.7-0.8 | 20-30 | 1.1 |
| 创意写作 | 0.7-0.9 | 0.9-0.95 | 50-100 | 1.0 |
| 代码生成 | 0.1-0.2 | 0.8 | 30 | 1.1 |
| 闲聊对话 | 0.6-0.8 | 0.9 | 50 | 1.1 |
| 翻译 | 0.1 | 0.8 | 20 | 1.0 |

## 总结

推理流程：
1. ✅ 加载基础模型
2. ✅ 应用 2D RoPE（如果训练时使用）
3. ✅ 应用 LoRA 层结构
4. ✅ 加载 LoRA 权重
5. ✅ 生成文本

关键点：
- `lora_rank` 必须与训练时一致
- `rope_2d_ratio` 必须与训练时一致
- 如果训练时使用了 `--patch_rope`，推理时也必须使用
- 调整生成参数以获得最佳效果
