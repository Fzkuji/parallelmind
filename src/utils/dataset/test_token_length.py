"""
测试 GPT-2 tokenizer 的单词数 vs Token数关系

用于确定处理 FineWeb-Edu 数据集时的最佳切分长度
"""

import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from statistics import mean, median


def test_word_token_ratio(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    num_samples: int = 1000,
    tokenizer_name: str = "gpt2"
):
    """测试单词数和token数的关系"""

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading dataset: {dataset_name}/{subset}")
    dataset = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True
    )

    # 收集统计数据
    ratios = []  # token数/单词数的比例
    word_counts = []
    token_counts = []

    print(f"\nAnalyzing {num_samples} samples...")

    for i, item in enumerate(dataset):
        if i >= num_samples:
            break

        text = item.get('text', '').strip()
        if not text:
            continue

        # 计算单词数
        words = text.split()
        num_words = len(words)

        # 计算token数
        tokens = tokenizer.encode(text, add_special_tokens=False)
        num_tokens = len(tokens)

        if num_words > 0:
            ratio = num_tokens / num_words
            ratios.append(ratio)
            word_counts.append(num_words)
            token_counts.append(num_tokens)

    # 计算统计信息
    avg_ratio = mean(ratios)
    median_ratio = median(ratios)

    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"样本数: {len(ratios)}")
    print(f"平均 Token/Word 比例: {avg_ratio:.3f}")
    print(f"中位数 Token/Word 比例: {median_ratio:.3f}")
    print(f"平均单词数/样本: {mean(word_counts):.1f}")
    print(f"平均Token数/样本: {mean(token_counts):.1f}")

    print("\n" + "="*60)
    print("推荐配置（目标: 512 tokens/chunk）")
    print("="*60)

    # 基于平均比例计算
    target_tokens = 512
    recommended_words_avg = int(target_tokens / avg_ratio)
    recommended_words_median = int(target_tokens / median_ratio)

    print(f"基于平均比例: {recommended_words_avg} 单词 ≈ {target_tokens} tokens")
    print(f"基于中位数比例: {recommended_words_median} 单词 ≈ {target_tokens} tokens")

    # 测试几个具体的单词数
    print("\n" + "="*60)
    print("不同单词数对应的预期Token数")
    print("="*60)

    test_word_counts = [200, 250, 300, 350, 400, 450, 500]
    for words in test_word_counts:
        expected_tokens = int(words * avg_ratio)
        print(f"{words:3d} 单词 ≈ {expected_tokens:3d} tokens")

    # 实际测试一些样本
    print("\n" + "="*60)
    print("实际样本示例（前10个）")
    print("="*60)

    dataset = load_dataset(
        dataset_name,
        subset,
        split="train",
        streaming=True
    )

    for i, item in enumerate(dataset):
        if i >= 10:
            break

        text = item.get('text', '').strip()
        if not text:
            continue

        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ratio = len(tokens) / len(words) if len(words) > 0 else 0

        print(f"样本 {i+1}: {len(words):4d} 单词 → {len(tokens):4d} tokens (比例: {ratio:.3f})")

    print("\n" + "="*60)
    print("建议")
    print("="*60)
    print(f"使用 --max-words-per-chunk {recommended_words_avg} 来获得约512 tokens/chunk")
    print(f"或者直接使用 --max-tokens-per-chunk 512 让脚本自动控制")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Test word to token ratio for dataset processing"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='HuggingFaceFW/fineweb-edu',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--subset',
        type=str,
        default='sample-10BT',
        help='Dataset subset'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to analyze'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='Tokenizer to use'
    )

    args = parser.parse_args()

    test_word_token_ratio(
        dataset_name=args.dataset,
        subset=args.subset,
        num_samples=args.num_samples,
        tokenizer_name=args.tokenizer
    )


if __name__ == "__main__":
    main()
