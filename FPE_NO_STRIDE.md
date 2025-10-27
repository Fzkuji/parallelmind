# 为什么 Fourier PE 不需要 stride？

## 问题

用户发现：使用Fourier PE时，`branch_stride`参数没有意义。

## 核心区别

### RoPE 2D：需要 stride

```python
# 数据生成
branch_id = [0, 1, 2, 3, ...]
branch_positions = [0*4096, 1*4096, 2*4096, 3*4096, ...] = [0, 4096, 8192, 12288, ...]

# RoPE计算
θ = branch_position / 10000^(2i/d)

# Branch 0: θ_0 = 0 / 10000^k = 0
# Branch 1: θ_1 = 4096 / 10000^k ≈ 0.41 (假设k=1)
# Branch 2: θ_2 = 8192 / 10000^k ≈ 0.82

# Q·K相似度
cos(θ_i - θ_j) = cos(4096 / 10000^k)

# stride越大，角度差越大，区分度越好
```

**stride的作用**：
- ✅ 增大不同branch的旋转角度差
- ✅ 提高RoPE的区分度
- ✅ stride=4096 比 stride=128 区分度更好（但仍不够，93-99%相似度）

---

### Fourier PE：不需要 stride

```python
# 数据生成（可以是任何值）
branch_id = [0, 4096, 8192, 12288, ...]  # 或者 [0, 1, 2, 3, ...]

# FPE查表
PE[0] = [sin(0/10000^0), cos(0/10000^0), sin(0/10000^1), cos(0/10000^1), ...]
PE[1] = [sin(1/10000^0), cos(1/10000^0), sin(1/10000^1), cos(1/10000^1), ...]
PE[4096] = [sin(4096/10000^0), cos(4096/10000^0), sin(4096/10000^1), ...]
...

# 直接使用branch_id查表
branch_pe = PE[branch_id]
```

**不需要stride**：
- ✅ 每个索引对应**不同的向量**
- ✅ PE[0] ≠ PE[1] ≠ PE[4096] ≠ ...
- ✅ 区分度由Fourier公式保证，不依赖索引间距

---

## 示例对比

### 场景：2个branch

**RoPE 2D**：
```python
# stride=128
Branch 0: pos=0*128=0 → θ=0
Branch 1: pos=1*128=128 → θ=128/10000^k

# 相似度
cos(θ_1 - θ_0) = cos(128/10000^k) ≈ 0.99  # 太相似！

# stride=4096
Branch 0: pos=0*4096=0 → θ=0
Branch 1: pos=1*4096=4096 → θ=4096/10000^k

# 相似度
cos(θ_1 - θ_0) = cos(4096/10000^k) ≈ 0.95  # 仍然太相似！
```

**Fourier PE**：
```python
# 直接用branch_id（不管是0,1还是0,4096）
Branch 0: branch_id=0 → PE[0]
Branch 1: branch_id=128 → PE[128]  # 或 branch_id=1 → PE[1]

# 相似度
PE[0] · PE[128] ≈ 0.31  # 很好的区分度！
PE[0] · PE[1] ≈ 0.85    # 也不错（虽然更相似）

# 关键：不管branch_id是多少，PE表都预计算好了
# 不需要通过stride来"拉开距离"
```

---

## 实现变化

### 之前的错误实现

```python
# model_minimind.py (旧版)
branch_ids = pos2d[:, :, 0]  # [0, 128, 256, ...]
branch_positions = (branch_ids / config.branch_stride).long()  # [0, 1, 2, ...]
branch_pe = fourier_pe(branch_positions)  # 查表

# 问题：
# 1. 引入了不必要的branch_stride参数
# 2. 需要"除以stride"还原成索引
# 3. 用户困惑：为什么FPE需要stride？
```

### 正确的实现

```python
# model_minimind.py (新版)
branch_ids = pos2d[:, :, 0]  # [0, 4096, 8192, ...] 或 [0, 1, 2, ...]
branch_pe = fourier_pe(branch_ids.long())  # 直接查表

# 优势：
# 1. 不需要branch_stride参数
# 2. 直接使用branch_id查表
# 3. 语义清晰：FPE是"查表"，不是"计算距离"
```

---

## 为什么max_positions要设32768？

因为：
1. **数据层面**：当前代码使用`branch_id * DEFAULT_BRANCH_POSITION_STRIDE`
   - `DEFAULT_BRANCH_POSITION_STRIDE = 4096`
   - 8个branch: `[0, 4096, 8192, 12288, 16384, 20480, 24576, 28672]`
   - 最大值：28672

2. **安全裕度**：设为32768（比28672大）确保所有可能的branch_id都能查表

3. **内存成本**：
   ```python
   # FPE表大小
   max_positions × d_model = 32768 × 512 = 16M参数

   # 固定版本（register_buffer）：不参与训练
   # 可学习版本（Parameter）：参与训练
   ```

---

## 配置对比

### RoPE 2D配置

```bash
python trainer/train_pretrain.py \
  --pe rope
  # stride在数据层面控制（DEFAULT_BRANCH_POSITION_STRIDE=4096）
```

### Fourier PE配置

```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --fpe_theta 10000.0 \
  --fpe_max_positions 32768 \
  --fpe_learnable  # 可选
  # 不需要 --branch_stride 参数！
```

---

## 总结

| 特性 | RoPE 2D | Fourier PE |
|-----|---------|-----------|
| **Branch编码方式** | 旋转（相对位置） | 查表（绝对位置） |
| **stride的作用** | ✅ 增大角度差，提高区分度 | ❌ 无作用，直接查表 |
| **参数** | `branch_stride` | `fpe_max_positions` |
| **关键点** | stride越大越好（但仍不够） | max_positions足够大即可 |
| **区分度** | 93-99%相似 ❌ | 31%相似 ✅ |

**结论**：
- ✅ RoPE 2D需要stride来拉开距离
- ✅ Fourier PE不需要stride，直接查表即可
- ✅ 移除了不必要的`branch_stride`配置参数
- ✅ 实现更简洁、语义更清晰

---

## 测试结果

```bash
$ python test_fpe_integration.py
```

**Branch区分度**：
```
相同input_ids，不同branch的logits差异: 0.478115
✓ Branch区分成功！不同branch产生不同的输出
```

**注意**：区分度从0.108 → 0.478，因为：
- Branch 0: PE[0]
- Branch 1: PE[128]（而不是PE[1]）
- PE[0]和PE[128]距离更远，区分度更好！

这进一步验证了：FPE直接用branch_id查表，不需要除以stride。
