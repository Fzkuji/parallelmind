# Fourier Position Encoding (FPE) 使用指南

## 概述

**Fourier PE** 是一种新的位置编码方案，专门用于区分多branch并行训练：

- **Branch维度**：使用 Fourier PE（绝对位置编码）
- **Time维度**：使用 1D RoPE（相对位置编码）

### 与RoPE 2D的对比

| 特性 | RoPE 2D（原方案） | Fourier PE（新方案） |
|-----|-----------------|-------------------|
| Branch编码方式 | RoPE（相对，旋转）| Fourier（绝对，加法）|
| Time编码方式 | RoPE（相对，旋转）| RoPE（相对，旋转）|
| Branch区分度 | 93-99%相似 ❌ | 31%相似 ✅ |
| 实现位置 | Attention内部 | Embedding后面 |
| 架构解耦性 | 低 | 高 |

### 为什么FPE更好？

```
RoPE 2D问题：
  不同branch: cos(θ_i) ⋅ cos(θ_j) + sin(θ_i) ⋅ sin(θ_j) = cos(θ_i - θ_j) ≈ 0.95
  → 即使stride=4096，相似度仍然93-99%

Fourier PE：
  Branch 0: PE(0) = [sin(0/θ^(2i/d)), cos(0/θ^(2i/d)), ...]
  Branch 1: PE(128) = [sin(128/θ^(2i/d)), cos(128/θ^(2i/d)), ...]
  → 直接加在embedding上，L2距离17.4，相似度31%
```

---

## 快速开始

### 1. 测试安装

```bash
# 测试Fourier PE实现
python model/fourier_pe.py

# 测试模型集成
python test_fpe_integration.py
```

应该看到：
```
✓ Branch区分成功！不同branch产生不同的输出
相同input_ids，不同branch的logits差异: 0.108702
```

---

### 2. 训练（推荐流程）

#### Stage 1: 单Branch预热（必须！）

让模型先学会基础的语言建模，不引入branch混淆。

```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 2 \
  --batch_size 4 \
  --branches_per_sample 1 \
  --branch_slice_count 8 \
  --branch_loop_all \
  --branch_stride 128 \
  --out_dir out/fpe_stage1
```

**参数说明**：
- `--pe fpe`: 使用Fourier PE（关键！）
- `--branches_per_sample 1`: 每个样本只有1个branch
- `--branch_slice_count 8`: 把数据分成8份，轮流训练
- `--branch_loop_all`: 自动遍历所有8份
- `--branch_stride 128`: branch之间的位置间隔

**预期结果**：
- Loss应该降到2.0以下
- 单branch生成质量应该正常

---

#### Stage 2: 2个Branch训练

在单branch基础上，逐步引入多branch。

```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 3 \
  --batch_size 4 \
  --batch_by_samples \
  --max_branches_per_sample 2 \
  --min_branches_per_sample 2 \
  --branch_stride 128 \
  --init_weight out/fpe_stage1/pretrain_512.pth \
  --out_dir out/fpe_stage2_2branch
```

**参数说明**：
- `--max_branches_per_sample 2`: 最多2个branch
- `--min_branches_per_sample 2`: 最少2个branch（固定2个）
- `--batch_by_samples`: batch_size表示样本数，不是文本数
- `--init_weight`: 从stage1的权重开始

**预期结果**：
- Loss应该继续下降（1.5左右）
- 2个branch推理应该独立且正确

---

#### Stage 3: 增加到4/8个Branch（可选）

如果2个branch成功，可以逐步增加。

```bash
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 3 \
  --batch_size 4 \
  --batch_by_samples \
  --max_branches_per_sample 4 \
  --min_branches_per_sample 2 \
  --branch_stride 128 \
  --init_weight out/fpe_stage2_2branch/pretrain_512.pth \
  --out_dir out/fpe_stage3_4branch
```

---

### 3. 推理测试

```bash
python parallel_generate.py \
  --model out/fpe_stage2_2branch/pretrain_512.pth \
  --branches 2 \
  --mode pretrain
```

**检查点**：
1. 不同branch的输出应该完全独立
2. 单branch时输出质量应该正常
3. 不应该出现gibberish

---

## 参数详解

### 位置编码参数

```bash
--pe fpe                    # 使用Fourier PE（默认rope）
--branch_stride 128         # Branch间隔（默认128）
--fpe_theta 10000.0         # FPE基础频率（默认10000）
--fpe_learnable             # 使FPE可学习（默认固定）
```

#### `--branch_stride` 的作用

控制不同branch在Fourier PE空间中的位置间隔：

```python
# stride=128时：
Branch 0 → FPE(0)
Branch 1 → FPE(128)
Branch 2 → FPE(256)
...

# stride=256时：
Branch 0 → FPE(0)
Branch 1 → FPE(256)
Branch 2 → FPE(512)
...
```

**推荐值**：
- `128`：默认，适合大多数场景
- `256`：如果128不够，可以增大到256
- `512`：最大间隔（不推荐，可能浪费位置空间）

**分析**：
```bash
# 查看不同stride的区分度
python -c "from model.fourier_pe import analyze_fourier_discrimination; \
           analyze_fourier_discrimination(stride=128)"
```

---

#### `--fpe_theta` 的作用

控制Fourier PE的频率范围：

```python
PE[pos, 2i]   = sin(pos / theta^(2i/d))
PE[pos, 2i+1] = cos(pos / theta^(2i/d))
```

- `theta=10000`（默认）：和Transformer原始论文一致
- `theta=1000`：更高频，适合小的stride
- `theta=100000`：更低频，适合大的stride

**推荐**：保持默认10000

---

#### `--fpe_learnable` 的作用

- **固定**（默认）：FPE从数学公式初始化，训练中不变
- **可学习**：FPE从数学公式初始化，训练中更新

**推荐**：
- 先用固定版本训练
- 如果效果不好，再尝试可学习版本

---

## 实现细节

### 架构图

```
Input tokens
    ↓
Token Embedding
    ↓
    + Fourier PE(branch_id)  ← 新增：branch编码
    ↓
Transformer Layers
    ├─ Self-Attention
    │   └─ 1D RoPE(time_id)  ← time编码
    └─ FFN
    ↓
Output
```

### 代码位置

1. **Fourier PE实现**：[model/fourier_pe.py](model/fourier_pe.py)
   ```python
   class FourierPositionEncoding(nn.Module):
       """Fourier位置编码（branch维度）"""
   ```

2. **模型集成**：[model/model_minimind.py](model/model_minimind.py:456-476)
   ```python
   # 初始化
   if config.pe_type == 'fpe':
       self.fourier_pe = FourierPositionEncoding(...)
   ```

3. **Forward逻辑**：[model/model_minimind.py](model/model_minimind.py:507-529)
   ```python
   # Forward
   if self.config.pe_type == 'fpe':
       branch_pe = self.fourier_pe(branch_positions)
       hidden_states = hidden_states + branch_pe
       cos, sin = self.rotary_emb(hidden_states, time_ids)
   ```

4. **训练脚本**：[trainer/train_pretrain.py](trainer/train_pretrain.py:210-215)
   ```python
   parser.add_argument("--pe", choices=['rope', 'fpe'])
   ```

---

## 对比实验

### 设计实验

同时训练两个模型进行对比：

```bash
# RoPE 2D（baseline）
python trainer/train_pretrain.py \
  --pe rope \
  --epochs 3 \
  --batch_size 4 \
  --max_branches_per_sample 2 \
  --out_dir out/rope_baseline

# Fourier PE（新方案）
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 3 \
  --batch_size 4 \
  --max_branches_per_sample 2 \
  --out_dir out/fpe_new
```

### 评估指标

1. **训练Loss**：
   - FPE应该更容易降低
   - 目标：< 1.5（RoPE通常卡在2.5）

2. **推理质量**：
   ```bash
   python parallel_generate.py --model <path> --branches 2
   ```
   - 检查是否有gibberish
   - 检查不同branch是否独立

3. **Branch区分度**（可选）：
   ```python
   # 在test_fpe_integration.py中的test_branch_discrimination()
   # 应该看到：
   相同input_ids，不同branch的logits差异: > 0.1
   ```

---

## 常见问题

### Q1: FPE和RoPE 2D能共存吗？

可以！`--pe`参数控制使用哪种方案：
- `--pe rope`：使用RoPE 2D（原方案）
- `--pe fpe`：使用Fourier PE（新方案）

模型权重不通用，需要分别训练。

---

### Q2: 能从RoPE 2D的权重迁移到FPE吗？

不能直接迁移。因为：
1. FPE在embedding后添加branch编码
2. Attention使用1D RoPE（不是2D）

建议从头训练FPE模型。

---

### Q3: branch_stride应该设置多大？

推荐128（默认）。分析：

```bash
python model/fourier_pe.py
```

看到：
```
stride=128: L2距离=17.4, 相似度=31%  ← 推荐
stride=256: L2距离=更大, 相似度=更低  ← 如果128不够
```

通常128就足够了。

---

### Q4: 为什么需要Stage 1单branch预热？

因为：
1. 多branch训练比单branch难得多
2. 如果模型还没学会基础LM，直接上多branch会崩溃
3. 单branch预热可以让模型先学会语言，再学习branch区分

类似于课程学习（Curriculum Learning）。

---

### Q5: FPE是绝对位置编码，会不会泛化性差？

**理论分析**：

1. **Branch维度**：
   - 是离散的（branch 0, 1, 2, ...）
   - 绝对位置编码更合适
   - 类比：BERT的segment embedding（也是绝对的）

2. **Time维度**：
   - 是连续的（时间步）
   - 仍然使用RoPE（相对位置编码）
   - 保留了泛化性

3. **实际效果**：
   - Branch ID通常固定（训练和推理一致）
   - 不需要外推到未见过的branch
   - 关键是区分度，不是泛化性

---

### Q6: 如果训练失败怎么办？

**Debug步骤**：

1. **检查loss**：
   ```bash
   # 如果loss不降：
   # 1. 确认使用了单branch预热
   # 2. 降低学习率（--learning_rate 1e-4）
   # 3. 减少branch数量（先试2个）
   ```

2. **检查推理**：
   ```bash
   python parallel_generate.py --model <path> --branches 1
   # 单branch应该正常

   python parallel_generate.py --model <path> --branches 2
   # 2个branch应该独立
   ```

3. **增大branch_stride**：
   ```bash
   --branch_stride 256  # 从128增大到256
   ```

4. **使用可学习FPE**：
   ```bash
   --fpe_learnable  # 让模型学习最优编码
   ```

---

## 总结

### 优势

1. ✅ **更强的区分度**：31%相似度 vs RoPE 2D的93-99%
2. ✅ **架构解耦**：branch和time编码分离，更清晰
3. ✅ **实现简单**：只需在embedding后加FPE
4. ✅ **可控性强**：调整stride即可控制区分度
5. ✅ **兼容性好**：和原有RoPE 2D并存，可对比

### 劣势

1. ⚠️ **需要重新训练**：不能直接迁移RoPE 2D的权重
2. ⚠️ **绝对位置编码**：branch维度使用绝对编码（但这可能是优势）

### 推荐使用场景

- ✅ 从头训练新模型
- ✅ RoPE 2D训练困难（loss卡在2.5）
- ✅ 需要强branch区分度
- ✅ 做研究实验对比

---

## 下一步

1. **运行测试**：
   ```bash
   python test_fpe_integration.py
   ```

2. **开始训练**：
   ```bash
   # Stage 1
   python trainer/train_pretrain.py --pe fpe --epochs 2 --branches_per_sample 1 ...
   ```

3. **监控训练**：
   - Loss应该能降到2.0以下（Stage 1）
   - Loss应该能降到1.5左右（Stage 2）

4. **推理验证**：
   ```bash
   python parallel_generate.py --model <path> --branches 2
   ```

5. **对比实验**：
   - 同时训练RoPE和FPE
   - 对比训练曲线和推理质量

祝实验顺利！
