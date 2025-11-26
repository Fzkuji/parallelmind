# ParallelMind - ICML Submission Plan

## Core Contribution

**One-sentence**: 2D RoPE enables native parallel inference for LLMs, processing multiple queries in a single forward pass.

## Overall Performance: Table 1

**Task**: Answer N independent questions simultaneously
**Metric**: Throughput (tok/s) + Accuracy (%)

| Model | Method | #Branch | MMLU | | C-Eval | | GSM8K | | HumanEval | |
|-------|--------|---------|------|-----|--------|-----|-------|-----|-----------|-----|
| | | | tok/s | Acc | tok/s | Acc | tok/s | Acc | tok/s | Acc |
| **Qwen2.5-1.5B** | Sequential | 1 | 100 | 58.2 | 100 | 55.4 | 100 | 42.1 | 100 | 35.0 |
| | Batch | 4 | 320 | 58.2 | 320 | 55.4 | 320 | 42.1 | 320 | 35.0 |
| | Batch | 8 | 500 | 58.2 | 500 | 55.4 | 500 | 42.1 | 500 | 35.0 |
| | **2D RoPE** | 2 | ? | ? | ? | ? | ? | ? | ? | ? |
| | **2D RoPE** | 4 | ? | ? | ? | ? | ? | ? | ? | ? |
| | **2D RoPE** | 8 | ? | ? | ? | ? | ? | ? | ? | ? |
| **Qwen2.5-7B** | Sequential | 1 | 50 | 68.5 | 50 | 65.2 | 50 | 58.3 | 50 | 45.0 |
| | Batch | 4 | 160 | 68.5 | 160 | 65.2 | 160 | 58.3 | 160 | 45.0 |
| | **2D RoPE** | 4 | ? | ? | ? | ? | ? | ? | ? | ? |
| | **2D RoPE** | 8 | ? | ? | ? | ? | ? | ? | ? | ? |
| **LLaMA-3-8B** | Sequential | 1 | - | - | - | - | - | - | - | - |
| | Batch | 4 | - | - | - | - | - | - | - | - |
| | **2D RoPE** | 4 | - | - | - | - | - | - | - | - |

**Expected Conclusions**:
1. **Throughput**: 2D RoPE ≈ Batch, scales with #Branch
2. **Quality**: 2D RoPE ≈ Sequential/Batch, no degradation
3. **Generalization**: Works across multiple models

## Ablation Experiments

### Table 2: rope_2d_ratio Ablation

**Setting**: #Branch=4

| Training Mode | rope_2d_ratio | Loss ↓ | MMLU Acc | C-Eval Acc |
|---------------|---------------|--------|----------|------------|
| **Pre-train (MiniMind)** | 0.25 | ? | ? | ? |
| | 0.50 | ? | ? | ? |
| | 0.75 | ? | ? | ? |
| **Fine-tune (Qwen2.5-1.5B)** | 0.25 | ? | ? | ? |
| | 0.50 | ? | ? | ? |
| | 0.75 | ? | ? | ? |

> Note: Throughput not included as rope_2d_ratio doesn't affect inference speed.

### Table 3: Branch Scaling

**Setting**: rope_2d_ratio=0.5

| Training Mode | #Branch | Throughput (tok/s) | Speedup | Loss ↓ | MMLU Acc |
|---------------|---------|-------------------|---------|--------|----------|
| **Pre-train (MiniMind)** | 1 | ? | 1.0× | ? | ? |
| | 2 | ? | ?× | ? | ? |
| | 4 | ? | ?× | ? | ? |
| | 8 | ? | ?× | ? | ? |
| | 16 | ? | ?× | ? | ? |
| **Fine-tune (Qwen2.5-1.5B)** | 1 | 100 | 1.0× | ? | 58.2 |
| | 2 | ? | ?× | ? | ? |
| | 4 | ? | ?× | ? | ? |
| | 8 | ? | ?× | ? | ? |
| | 16 | ? | ?× | ? | ? |

> Note: Could also vary model size, #heads, head_dim, #layers - saved for extension.

## 2D Attention Pattern Analysis

### Table 4: Attention Visualization

**Goal**: Visualize how attention works in 2D position space (time × branch)

| Analysis | Description |
|----------|-------------|
| 2D Attention Heatmap | Visualize attention distribution in (time, branch) space |
| Head Specialization | Analyze if different heads focus on time vs branch dimensions |
| Cross-branch Attention | Observe information interaction patterns between branches |
| Layer-wise Pattern | Compare attention patterns across different layers |

**Visualization Plan**:
- Heatmap showing full 2D attention pattern
- Compare patterns at different layers / heads
- Potentially discover interesting specialization (some heads focus on time, others on branch)

## Case Study: Multi-Question Reading Comprehension

### Table 5: SQuAD Parallel QA

**Scenario**: Answer multiple questions about the same passage simultaneously

**Setup**:
- Dataset: SQuAD v1.1 / v2.0
- Task: Given one context passage, answer N questions in parallel
- Input format: `[Context] Q1: ... Q2: ... Q3: ... Q4: ...`
- Output: N answers generated in one forward pass

| Method | #Questions | Throughput (Q/s) | Speedup | F1 Score | Exact Match |
|--------|------------|------------------|---------|----------|-------------|
| Sequential | 1 | ? | 1.0× | ? | ? |
| Batch | 4 | ? | ?× | ? | ? |
| **2D RoPE** | 4 | ? | ?× | ? | ? |
| **2D RoPE** | 8 | ? | ?× | ? | ? |

**Why This Scenario**:
- Natural fit: Same context + multiple questions is common in real applications
- Shared context: All branches attend to the same passage, questions differ
- Clear metrics: F1/EM are standard QA metrics
- Demonstrates practical value: Document QA, customer support, education

**Analysis Points**:
- Compare answer quality vs sequential processing
- Measure throughput improvement with shared context
- Analyze attention patterns: how branches interact with shared context

**Alternative Datasets** (for future consideration):
| Dataset | Description | Notes |
|---------|-------------|-------|
| **RACE** | English exam questions, 3-5 questions per passage | ⭐ Very suitable |
| **MultiRC** | Multiple questions per paragraph | ⭐ Very suitable |
| **QuALITY** | Long documents with multiple questions | ⭐ Very suitable |
| **HotpotQA** | Multi-hop reasoning | Requires cross-doc reasoning |
| **DROP** | Numerical reasoning | More challenging |
| **C3** | Chinese multiple choice reading comprehension | Chinese |
| **CMRC** | Chinese Machine Reading Comprehension | Chinese |

