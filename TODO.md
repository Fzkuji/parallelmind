# ParallelMind Research TODO

## Overview

Research plan for validating 2D RoPE parallel inference.

**Core Capability**: Multiple queries → Multiple answers (in one forward pass)

## 1. Overall Performance

**Goal**: Prove 2D RoPE parallel inference is effective compared to alternatives.

### 1.1 Baseline Methods Comparison

| Method | Description |
|--------|-------------|
| Sequential | Process queries one by one |
| Traditional Batch | Multiple sequences in batch dimension |
| vLLM / TGI | Existing inference framework batching |

### 1.2 Metrics

| Dimension | Metrics | Benchmark |
|-----------|---------|-----------|
| **Throughput** | tokens/sec, queries/sec | Custom load test |
| **Latency** | first token latency, total latency | Custom load test |
| **Quality** | accuracy, perplexity | MMLU, C-Eval, GSM8K |
| **Memory** | peak memory, KV cache size | Profiling |

### 1.3 Test Models
- [ ] MiniMind-Small (26M) - Fast validation
- [ ] MiniMind (104M) - Medium scale
- [ ] Qwen2.5-1.5B - Practical scale
- [ ] Qwen2.5-7B - Large scale (if resources allow)

### 1.4 Experiments
- [ ] Throughput comparison: Sequential vs Parallel (2, 4, 8, 16 branches)
- [ ] Latency comparison: Single query vs Parallel queries
- [ ] Quality evaluation: MMLU/C-Eval accuracy with different branch counts
- [ ] Memory profiling: Peak memory at different branch counts

## 2. Ablation Studies

### 2.1 rope_2d_ratio Impact
- [ ] Test different ratios: 0.125, 0.25, 0.5, 0.75
- [ ] Measure quality degradation at different ratios
- [ ] Find optimal ratio for different model sizes

### 2.2 branch_stride Analysis
- [ ] Test different stride values: 128, 256, 512, 1024
- [ ] Analyze cross-branch interference at different strides
- [ ] Measure memory efficiency vs quality trade-off

### 2.3 Number of Parallel Branches
- [ ] Test scaling from 2 to 32 branches
- [ ] Identify the point where quality starts to degrade
- [ ] Analyze attention patterns across branches

## 3. Cross-Branch Independence Study

### 3.1 Attention Analysis
- [ ] Visualize attention patterns between different branches
- [ ] Measure attention leakage across branches
- [ ] Verify branches are truly independent

### 3.2 Output Quality Analysis
- [ ] Check for interference between branches
- [ ] Test with similar vs different prompts
- [ ] Compare parallel vs sequential output quality

## 4. Model Scaling Experiments

### 4.1 Different Model Sizes
- [ ] Test on MiniMind (26M, 104M)
- [ ] Test on Qwen2.5 (0.5B, 1.5B, 3B, 7B)
- [ ] Test on LLaMA series if applicable

### 4.2 Training Data Scale
- [ ] Compare models trained with different data amounts
- [ ] Analyze if more training data improves parallel capability

## 5. Training Strategy Experiments

### 5.1 LoRA vs Full Fine-tuning
- [ ] Compare quality between LoRA and full fine-tuning
- [ ] Analyze training efficiency (time, memory, data needed)
- [ ] Test different LoRA ranks (4, 8, 16, 32)

### 5.2 Training Data Format
- [ ] Compare training with different branch numbers
- [ ] Test with fixed vs dynamic branch numbers
- [ ] Analyze impact of padding strategies

### 5.3 Pre-training vs Fine-tuning Only
- [ ] Compare pre-training from scratch with 2D RoPE
- [ ] Compare fine-tuning existing models with 2D RoPE
- [ ] Analyze which approach is more efficient

## 6. Application Scenarios

**Goal**: Demonstrate practical value of multi-query parallel inference.

### 6.1 High-Throughput QA Service
- **Scenario**: API service handling multiple user requests simultaneously
- **Demo**: One inference call answers N independent questions
- **Metrics**: QPS, latency reduction
- [ ] Implement demo server
- [ ] Benchmark against sequential serving

### 6.2 Batch Document Processing
- **Scenario**: Summarize/translate multiple documents at once
- **Demo**: 10 articles → 10 summaries (single call)
- **Metrics**: Throughput, ROUGE scores
- [ ] Prepare multi-document dataset
- [ ] Compare with sequential processing

### 6.3 Multi-Task Evaluation Acceleration
- **Scenario**: Speed up model evaluation on benchmarks
- **Demo**: Parallel evaluation on MMLU/C-Eval
- **Metrics**: Evaluation time reduction
- [ ] Implement parallel evaluation script
- [ ] Measure speedup ratio

### 6.4 Parallel Information Extraction
- **Scenario**: Extract information from multiple sources simultaneously
- **Demo**: Multiple reviews → Parallel sentiment analysis
- **Metrics**: Processing speed, accuracy
- [ ] Prepare extraction dataset
- [ ] Benchmark extraction task

## 7. Efficiency Analysis

### 7.1 Memory Usage
- [ ] Profile memory consumption at different branch counts
- [ ] Compare with traditional batching memory usage
- [ ] Analyze KV cache efficiency

### 7.2 Computational Cost
- [ ] Measure FLOPs for parallel vs sequential
- [ ] Analyze attention computation overhead
- [ ] Test on different hardware (A100, 4090, 3090)

### 7.3 Real-world Throughput
- [ ] Benchmark end-to-end latency
- [ ] Test under different load conditions
- [ ] Compare with vLLM, TGI, etc.

## 8. Theoretical Analysis

### 8.1 Position Encoding Theory
- [ ] Analyze why 2D RoPE enables parallel inference
- [ ] Study the mathematical properties of branch separation
- [ ] Compare with other position encoding schemes

### 8.2 Attention Mechanism Analysis
- [ ] Study how attention behaves in 2D position space
- [ ] Analyze the role of different attention heads
- [ ] Investigate if some heads specialize for branch separation

## Priority Order

1. **High Priority** (Start immediately)
   - 1.4 Throughput/latency comparison experiments
   - 2.1 rope_2d_ratio impact study
   - 3.1 Attention pattern visualization

2. **Medium Priority** (After initial results)
   - 1.4 Quality evaluation on benchmarks
   - 2.2 branch_stride analysis
   - 4.1 Different model sizes

3. **Lower Priority** (For comprehensive study)
   - 6.x Application scenarios
   - 7.x Efficiency analysis
   - 8.x Theoretical analysis

## Notes

- All experiments should be reproducible with fixed random seeds
- Document hardware configuration for each experiment
- Save checkpoints and logs for analysis
- Consider writing experiment scripts in `scripts/experiments/`
