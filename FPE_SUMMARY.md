# Fourier PE 实现总结

## 完成的工作

### 1. 核心实现

✅ **[model/fourier_pe.py](model/fourier_pe.py)**
- `FourierPositionEncoding`: Fourier位置编码（branch维度）
- `LearnableBranchEncoding`: 可学习的branch embedding（备选）
- `analyze_fourier_discrimination()`: 区分度分析函数

✅ **[model/model_minimind.py](model/model_minimind.py)**
- 添加`pe_type`配置参数（'rope' or 'fpe'）
- `MiniMindModel.__init__`: 根据pe_type初始化不同的位置编码
- `MiniMindModel.forward`: 根据pe_type使用不同的编码方式

✅ **[trainer/train_pretrain.py](trainer/train_pretrain.py)**
- 添加`--pe`参数（choices=['rope', 'fpe']）
- 添加FPE相关参数：`--fpe_theta`, `--branch_stride`, `--fpe_learnable`

### 2. 测试和文档

✅ **[test_fpe_integration.py](test_fpe_integration.py)**
- 测试RoPE 2D模式
- 测试Fourier PE模式
- 测试branch区分能力

✅ **[FOURIER_PE_GUIDE.md](FOURIER_PE_GUIDE.md)**
- 详细的使用指南
- 训练流程（Stage 1/2/3）
- 参数说明
- 常见问题

---

## 设计架构

### RoPE 2D（原方案，保留）

```
Input tokens → Embedding → Transformer
                              ↓
                         RoPE 2D(branch, time)
                              ↓
                         Attention
```

**问题**：不同branch的编码相似度93-99%，区分度不足

---

### Fourier PE（新方案）

```
Input tokens → Embedding
                  ↓
              + Fourier PE(branch)  ← branch编码（绝对）
                  ↓
             Transformer
                  ↓
             1D RoPE(time)  ← time编码（相对）
                  ↓
             Attention
```

**优势**：
- Branch区分度：31%相似度（vs RoPE的93-99%）
- L2距离：17.4（一致且稳定）
- 架构解耦：branch和time分离

---

## 测试结果

```bash
$ python model/fourier_pe.py
```

**Fourier PE区分度分析**：
```
Branch positions: [0, 128, 256, 384, 512, 640, 768, 896]
L2距离: 17.4 (相邻branch)
平均相似度: 31.36%
最大相似度: 40.59%
最小相似度: 20.01%

对比：
  RoPE 2D (stride=128): 93-99%相似度 ❌
  Fourier PE: 31.36%平均相似度 ✅
```

```bash
$ python test_fpe_integration.py
```

**模型集成测试**：
```
✓ RoPE 2D 模式测试通过
✓ Fourier PE 模式测试通过
✓ Branch区分成功！
  相同input_ids，不同branch的logits差异: 0.108702
```

---

## 使用方法

### 方式1: RoPE 2D（原方案）

```bash
python trainer/train_pretrain.py \
  --pe rope \
  --epochs 2 \
  --batch_size 4 \
  --max_branches_per_sample 2
```

### 方式2: Fourier PE（新方案）

```bash
# Stage 1: 单branch预热
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 2 \
  --batch_size 4 \
  --branches_per_sample 1 \
  --branch_slice_count 8 \
  --branch_loop_all \
  --out_dir out/fpe_stage1

# Stage 2: 2个branch
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 3 \
  --batch_size 4 \
  --max_branches_per_sample 2 \
  --min_branches_per_sample 2 \
  --batch_by_samples \
  --init_weight out/fpe_stage1/pretrain_512.pth \
  --out_dir out/fpe_stage2
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--pe` | `rope` | 位置编码类型：'rope' 或 'fpe' |
| `--branch_stride` | `128` | Branch间隔（用于FPE） |
| `--fpe_theta` | `10000.0` | FPE基础频率 |
| `--fpe_learnable` | `False` | 是否让FPE可学习 |

---

## 关键改动点

### 1. Config

```python
class MiniMindConfig:
    def __init__(
        self,
        pe_type: str = 'rope',           # 新增
        fpe_theta: float = 10000.0,      # 新增
        fpe_max_branches: int = 512,     # 新增
        fpe_learnable: bool = False,     # 新增
        branch_stride: int = 128,        # 新增
        ...
    ):
```

### 2. Model Init

```python
class MiniMindModel:
    def __init__(self, config):
        if config.pe_type == 'rope':
            # 原有方式：RoPE 2D
            patch_model_with_interleaved_2d_rope(self, pair_indices)
            self.fourier_pe = None
        elif config.pe_type == 'fpe':
            # 新方式：Fourier PE
            self.fourier_pe = FourierPositionEncoding(...)
```

### 3. Model Forward

```python
def forward(self, input_ids, pos2d, ...):
    hidden_states = self.embed_tokens(input_ids)

    if self.config.pe_type == 'rope':
        # RoPE 2D: branch + time都在RoPE中
        set_rope_pos2d(self, pos2d)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

    elif self.config.pe_type == 'fpe':
        # FPE: branch用Fourier，time用1D RoPE
        branch_ids = pos2d[:, :, 0]
        time_ids = pos2d[:, :, 1]

        branch_positions = (branch_ids / self.config.branch_stride).long()
        branch_pe = self.fourier_pe(branch_positions)
        hidden_states = hidden_states + branch_pe

        cos, sin = self.rotary_emb(hidden_states, time_ids)
```

---

## 对比实验建议

同时训练两个模型，对比效果：

```bash
# Terminal 1: RoPE 2D baseline
python trainer/train_pretrain.py \
  --pe rope \
  --epochs 5 \
  --out_dir out/rope_baseline

# Terminal 2: Fourier PE
python trainer/train_pretrain.py \
  --pe fpe \
  --epochs 5 \
  --out_dir out/fpe_new
```

**预期结果**：
- FPE的loss应该更容易降低（< 1.5）
- RoPE可能卡在2.5左右
- FPE的推理应该更独立、更少gibberish

---

## 文件清单

```
parallelmind/
├── model/
│   ├── fourier_pe.py              ← 新增：FPE实现
│   └── model_minimind.py          ← 修改：添加pe_type支持
├── trainer/
│   └── train_pretrain.py          ← 修改：添加--pe参数
├── test_fpe_integration.py        ← 新增：集成测试
├── FOURIER_PE_GUIDE.md            ← 新增：使用指南
└── FPE_SUMMARY.md                 ← 新增：本文件
```

---

## 下一步

1. **运行测试**：
   ```bash
   python test_fpe_integration.py
   ```

2. **开始训练**（推荐从小模型开始）：
   ```bash
   python trainer/train_pretrain.py \
     --pe fpe \
     --epochs 2 \
     --batch_size 4 \
     --branches_per_sample 1 \
     --branch_loop_all \
     --out_dir out/fpe_test
   ```

3. **对比实验**：
   - 同时跑rope和fpe
   - 对比loss曲线
   - 对比推理质量

4. **调优**（如果需要）：
   - 调整`branch_stride`（128 → 256）
   - 尝试`--fpe_learnable`
   - 调整训练策略（更多epochs、更大batch）

---

## 理论支持

### Fourier PE vs RoPE 2D

| 维度 | Fourier PE | RoPE 2D |
|-----|-----------|---------|
| **Branch** | 绝对位置（离散） | 相对位置（连续） |
| **适用性** | ✅ 离散的branch ID | ❌ branch ID不连续 |
| **区分度** | ✅ 31%相似度 | ❌ 93-99%相似度 |
| **实现** | ✅ 简单（加法） | ⚠️ 复杂（旋转） |
| **解耦性** | ✅ 独立于Attention | ❌ 耦合在Attention中 |

### 为什么RoPE 2D不适合branch？

RoPE设计用于**连续位置**的相对关系：
```
pos_i和pos_j的距离: Δ = pos_i - pos_j
旋转角度差: θ_i - θ_j = ω * Δ

问题：当Δ很大时（如stride=4096），cos(ω*Δ)仍然≈0.95
```

Fourier PE设计用于**绝对位置**的独立编码：
```
pos_i的编码: PE(pos_i) = [sin(pos_i/θ^k), cos(pos_i/θ^k), ...]
pos_j的编码: PE(pos_j) = [sin(pos_j/θ^k), cos(pos_j/θ^k), ...]

当pos_i ≠ pos_j时，PE(pos_i) ⊥ PE(pos_j)（正交性更好）
```

---

## 总结

**核心贡献**：
1. ✅ 实现了Fourier PE，解决RoPE 2D区分度不足的问题
2. ✅ 保留了原有RoPE 2D方案，可对比实验
3. ✅ 架构解耦，branch和time编码分离
4. ✅ 完整的测试和文档

**理论优势**：
- Branch区分度提升：31% vs 93-99%
- 更适合离散的branch ID
- 架构更清晰、更可控

**实践建议**：
- 先用FPE训练小模型，验证效果
- 如果成功，再扩展到大模型
- 保留RoPE作为baseline对比

祝实验顺利！🚀
