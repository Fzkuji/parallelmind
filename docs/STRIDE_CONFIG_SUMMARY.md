# Stride配置总结

## 修改概述

根据用户需求，完成了以下修改：

1. ✅ **RoPE 2D默认stride**：从4096改回**128**
2. ✅ **FPE默认stride**：使用**1**（直接的branch索引）
3. ✅ **fpe_max_positions默认值**：从32768改为**512**

---

## 核心区别

### RoPE 2D（原方案）

```python
# 数据生成
branch_stride = 128  # 默认值
branch_positions = [0*128, 1*128, 2*128, ...] = [0, 128, 256, 384, ...]

# RoPE计算
θ = branch_position / 10000^(2i/d)
# stride越大，不同branch的旋转角度差越大

# 为什么需要stride？
# - 增大不同branch的角度差，提高区分度
# - stride=128 vs stride=4096 的区分度都不够（93-99%相似）
```

**配置**：
```bash
python trainer/train_pretrain.py --pe rope
# 数据层自动使用 branch_stride=128
```

---

### FPE（新方案）

```python
# 数据生成
branch_stride = 1  # FPE模式下的默认值
branch_positions = [0*1, 1*1, 2*1, ...] = [0, 1, 2, 3, ...]

# FPE查表
PE[0] = 预计算的向量0
PE[1] = 预计算的向量1
PE[2] = 预计算的向量2
...

# 为什么不需要大的stride？
# - 直接查表，每个索引对应不同向量
# - PE[0] ≠ PE[1] ≠ PE[2] ≠ ...
# - 区分度由Fourier公式保证（31%相似度）
```

**配置**：
```bash
python trainer/train_pretrain.py --pe fpe
# 数据层自动使用 branch_stride=1
```

---

## 实现细节

### 1. 数据生成层（自动）

在训练脚本中根据`pe_type`自动设置：

```python
# trainer/train_pretrain.py
branch_stride = 1 if args.pe == 'fpe' else 128

collator = ParallelPretrainCollator(
    tokenizer,
    ...,
    branch_stride=branch_stride,
)
```

**结果**：
- `--pe rope`：自动使用stride=128
- `--pe fpe`：自动使用stride=1

### 2. 模型层（直接使用）

```python
# model/model_minimind.py
if self.config.pe_type == 'fpe':
    branch_ids = pos2d[:, :, 0]  # [0, 1, 2, 3, ...] (FPE)
    # 或 [0, 128, 256, ...] (如果数据层传入的是RoPE格式)

    # 直接查表
    branch_pe = self.fourier_pe(branch_ids.long())
    hidden_states = hidden_states + branch_pe
```

**关键**：FPE直接使用branch_id查表，不需要除以stride

---

## 参数对比

| 参数 | RoPE 2D | FPE |
|-----|---------|-----|
| **branch_stride**（数据层） | 128 | 1 |
| **branch_positions** | [0, 128, 256, ...] | [0, 1, 2, 3, ...] |
| **max_positions**（模型层） | N/A | 512 |
| **区分度** | 93-99%相似 | 31%相似 |

---

## 默认值修改

### 1. DEFAULT_BRANCH_POSITION_STRIDE

```python
# parallel/columnar.py
DEFAULT_BRANCH_POSITION_STRIDE = 128  # 从4096改为128
```

**原因**：
- stride=4096太大，内存浪费
- stride=128是RoPE 2D的推荐值
- FPE模式会自动覆盖为1

### 2. fpe_max_positions

```python
# model/model_minimind.py
fpe_max_positions: int = 512  # 从32768改为512
```

**原因**：
- Branch数量通常很小（<100）
- 512足够容纳常见场景
- 减少内存占用：512×512 vs 32768×512

### 3. Collator的branch_stride参数

```python
# parallel_data/parallel_collator.py
class ParallelPretrainCollator:
    def __init__(
        self,
        ...,
        branch_stride: int = 128,  # 新增参数，默认128
    ):
```

**作用**：
- 允许训练脚本根据pe_type动态设置
- RoPE模式：128
- FPE模式：1

---

## 使用示例

### RoPE 2D模式

```bash
python trainer/train_pretrain.py \
  --pe rope \
  --epochs 2 \
  --batch_size 4 \
  --max_branches_per_sample 2

# 数据层自动使用 branch_stride=128
# Branch positions: [0, 128, 256, 384, ...]
```

### FPE模式

```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 2 \
  --batch_size 4 \
  --max_branches_per_sample 2

# 数据层自动使用 branch_stride=1
# Branch positions: [0, 1, 2, 3, ...]
```

---

## 测试结果

```bash
$ python test_fpe_integration.py
```

**FPE模式**：
```
✓ FourierPositionEncoding: max_pos=512
✓ Branch区分成功！
  相同input_ids，不同branch的logits差异: 0.531860
```

**说明**：
- Branch 0: PE[0]
- Branch 1: PE[1]（而不是PE[128]）
- 区分度：0.532（非常好）

---

## 常见问题

### Q1: 为什么RoPE 2D需要stride=128？

**A**: RoPE通过旋转角度差来区分位置：
```
θ_0 = 0 / 10000^k = 0
θ_128 = 128 / 10000^k
θ_256 = 256 / 10000^k

# stride越大，角度差越大，区分度越好
# 但即使stride=4096，相似度仍有93%+
```

### Q2: 为什么FPE只需要stride=1？

**A**: FPE直接查表，不依赖位置距离：
```
PE[0] = [sin(0/θ^0), cos(0/θ^0), sin(0/θ^1), ...]
PE[1] = [sin(1/θ^0), cos(1/θ^0), sin(1/θ^1), ...]
PE[2] = [sin(2/θ^0), cos(2/θ^0), sin(2/θ^1), ...]

# 每个索引对应不同向量，不需要"拉开距离"
```

### Q3: fpe_max_positions为什么设为512？

**A**:
- Branch数量通常<100
- 预留5倍空间（512）足够安全
- 参数量：512×512 = 262K（很小）
- 如果branch_id超过512会报错（可增大此值）

### Q4: 如果我的branch数量>512怎么办？

**A**: 增大`fpe_max_positions`：
```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --fpe_max_positions 1024  # 或更大
```

但通常不需要，因为：
- Branch数量一般<100
- 即使有100个branch，512已经够用

---

## 总结

| 方面 | RoPE 2D | FPE |
|-----|---------|-----|
| **Branch stride** | 128（需要拉大距离） | 1（直接索引） |
| **Branch positions** | [0, 128, 256, ...] | [0, 1, 2, 3, ...] |
| **Max positions** | N/A | 512 |
| **区分方式** | 旋转角度差 | 查表（不同向量） |
| **区分度** | 93-99%相似 ❌ | 31%相似 ✅ |
| **内存占用** | 小（无额外参数） | 小（512×512） |

**推荐配置**：
- ✅ RoPE 2D：使用默认值（stride=128）
- ✅ FPE：使用默认值（stride=1, max_pos=512）
- ✅ 不需要手动设置stride，训练脚本自动处理

---

## 修改的文件

1. ✅ [parallel/columnar.py](parallel/columnar.py:131) - DEFAULT_BRANCH_POSITION_STRIDE = 128
2. ✅ [parallel_data/parallel_collator.py](parallel_data/parallel_collator.py:20) - 添加branch_stride参数
3. ✅ [trainer/train_pretrain.py](trainer/train_pretrain.py:310-323) - 根据pe_type自动设置stride
4. ✅ [model/model_minimind.py](model/model_minimind.py:33) - fpe_max_positions = 512
5. ✅ [test_fpe_integration.py](test_fpe_integration.py) - 更新测试和说明

**用户无需关心**：
- stride会根据`--pe`参数自动设置
- 只需选择`--pe rope`或`--pe fpe`即可
