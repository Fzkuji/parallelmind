"""
处理 FineWeb-Edu 10BT 数据集

功能：
1. 加载 HuggingFace FineWeb-Edu 数据集
2. 去重：移除完全相同的文本
3. 质量过滤：移除包含乱码或重复句子的文本
4. 智能切分：按句子切分，使用 tokenizer 精确控制每段 token 数
5. 保存为 JSON Lines 格式，每行包含 <|im_start|> 和 <|im_end|> 标记

使用方法：
    # 处理全部数据，每段精确控制在512 tokens
    python scripts/dataset/process_fineweb_edu_10bt.py \
        --output-dir dataset/fineweb-edu-10BT \
        --tokenizer gpt2

    # 测试处理（10万样本）
    python scripts/dataset/process_fineweb_edu_10bt.py \
        --max-samples 100000 \
        --output-dir dataset/fineweb-edu-10BT \
        --tokenizer gpt2
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Optional
from tqdm import tqdm
from transformers import AutoTokenizer


def compute_hash(text: str) -> str:
    """计算文本的 hash 值用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def is_garbage_text(text: str) -> bool:
    """
    检测文本是否包含乱码或质量问题

    检测规则：
    1. 非 ASCII 可打印字符占比过高（可能是乱码）
    2. 数字占比过高
    3. 特殊字符占比过高
    4. 文本过短
    """
    if len(text.strip()) < 50:  # 文本太短
        return True

    # 统计字符类型
    total_chars = len(text)
    ascii_printable = sum(1 for c in text if 32 <= ord(c) <= 126 or c in '\n\t')
    digits = sum(1 for c in text if c.isdigit())
    special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,;:!?\'"()-')

    # 非 ASCII 可打印字符占比过高
    if ascii_printable / total_chars < 0.7:
        return True

    # 数字占比过高
    if digits / total_chars > 0.3:
        return True

    # 特殊字符占比过高
    if special_chars / total_chars > 0.2:
        return True

    return False


def has_repetitive_sentences(text: str, threshold: float = 0.3) -> bool:
    """
    检测文本是否包含过多重复句子

    Args:
        text: 输入文本
        threshold: 重复句子占比阈值（超过此比例认为是重复文本）

    Returns:
        True 如果文本包含过多重复句子
    """
    # 按句子切分
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if len(sentences) < 3:
        return False  # 句子太少，无法判断

    # 统计每个句子出现的次数
    sentence_counts = Counter(sentences)

    # 计算重复句子的数量
    repeated_sentences = sum(count - 1 for count in sentence_counts.values() if count > 1)

    # 如果重复句子占比超过阈值
    if repeated_sentences / len(sentences) > threshold:
        return True

    # 检测连续重复的短语（3个词以上）
    words = text.lower().split()
    if len(words) > 20:
        # 检查是否有连续重复的3-gram
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        max_trigram_count = max(trigram_counts.values()) if trigram_counts else 0

        # 如果某个3-gram出现次数过多
        if max_trigram_count > len(trigrams) * 0.1:  # 超过10%
            return True

    return False


def split_into_sentences(text: str) -> List[str]:
    """
    将文本按句子切分

    使用正则表达式识别句子边界（. ! ? 后跟空格或换行）
    """
    # 先按段落分割
    paragraphs = text.split('\n')

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 按标点切分句子
        # 匹配 .!? 后面跟着空格、引号或结尾
        parts = re.split(r'([.!?]+[\s\'")\]]*)', para)

        current_sentence = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current_sentence += part
            else:
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""

        # 如果有剩余的内容（没有标点结尾的）
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

    return sentences


def create_chunks(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int = 512
) -> List[str]:
    """
    将文本切分成多个 chunk，每个 chunk 不超过 max_tokens 个 tokens

    切分策略：
    1. 按句子切分
    2. 使用 tokenizer 精确计算 token 数
    3. 合并句子直到接近 max_tokens
    4. 避免切断句子
    """
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        # 使用 tokenizer 计算实际 token 数
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))

        # 如果单个句子就超过 max_tokens，需要进一步切分
        if sentence_tokens > max_tokens * 1.3:  # 允许一定的超出
            # 如果当前chunk有内容，先保存
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = []
                current_tokens = 0

            # 对超长句子按逗号切分
            parts = re.split(r'([,;:][\s]*)', sentence)
            sub_chunk = ""

            for part in parts:
                test_text = sub_chunk + part
                test_tokens = len(tokenizer.encode(test_text, add_special_tokens=False))

                if test_tokens > max_tokens and sub_chunk:
                    chunks.append(sub_chunk.strip())
                    sub_chunk = part
                else:
                    sub_chunk = test_text

            if sub_chunk.strip():
                chunks.append(sub_chunk.strip())

            continue

        # 计算加入这个句子后的 token 数
        test_chunk = current_chunk + [sentence]
        test_text = ' '.join(test_chunk)
        test_tokens = len(tokenizer.encode(test_text, add_special_tokens=False))

        # 如果添加这个句子会超过 max_tokens，先保存当前chunk
        if test_tokens > max_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens = test_tokens

    # 保存最后一个chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def process_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_tokens_per_chunk: int = 512,
    tokenizer_name: str = "gpt2",
    output_dir: str = "dataset/fineweb-edu-10BT",
    offline: bool = False,
):
    """
    处理数据集的主函数
    """
    from datasets import load_dataset

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"Loading dataset: {dataset_name}/{subset}")
    if offline:
        print("Using offline mode (cached dataset)")

    # 加载数据集（streaming模式）
    dataset = load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True
    )

    # 统计信息
    stats = {
        'total_processed': 0,
        'duplicates': 0,
        'garbage': 0,
        'repetitive': 0,
        'chunks_created': 0,
    }

    # 去重用的 hash set
    seen_hashes: Set[str] = set()

    # 输出文件
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "train.jsonl"

    print(f"Processing dataset...")
    print(f"Output file: {output_file}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max tokens per chunk: {max_tokens_per_chunk}")
    print(f"Max samples: {max_samples if max_samples else 'ALL (10BT)'}")
    print()

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Processing", total=max_samples):
            stats['total_processed'] += 1

            # 达到最大样本数
            if max_samples and stats['total_processed'] > max_samples:
                break

            text = item.get('text', '').strip()
            if not text:
                continue

            # 去重检查
            text_hash = compute_hash(text)
            if text_hash in seen_hashes:
                stats['duplicates'] += 1
                continue
            seen_hashes.add(text_hash)

            # 质量过滤：检测乱码
            if is_garbage_text(text):
                stats['garbage'] += 1
                continue

            # 质量过滤：检测重复句子
            if has_repetitive_sentences(text):
                stats['repetitive'] += 1
                continue

            # 切分文本（使用 tokenizer 精确控制 token 数）
            chunks = create_chunks(text, tokenizer=tokenizer, max_tokens=max_tokens_per_chunk)

            # 保存每个 chunk
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # 跳过太短的chunk
                    continue

                # 格式化为 JSON Line，添加特殊标记
                formatted_text = f"<|im_start|>{chunk.strip()}<|im_end|>"
                json_line = json.dumps({"text": formatted_text}, ensure_ascii=False)
                f.write(json_line + '\n')
                stats['chunks_created'] += 1

    # 打印统计信息
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    print(f"Total samples processed: {stats['total_processed']:,}")
    print(f"Duplicates removed: {stats['duplicates']:,}")
    print(f"Garbage texts removed: {stats['garbage']:,}")
    print(f"Repetitive texts removed: {stats['repetitive']:,}")
    print(f"Chunks created: {stats['chunks_created']:,}")
    print(f"Output file: {output_file}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Process FineWeb-Edu dataset with deduplication and quality filtering"
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
        '--split',
        type=str,
        default='train',
        help='Dataset split'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all - processes entire 10BT dataset)'
    )
    parser.add_argument(
        '--max-tokens-per-chunk',
        type=int,
        default=512,
        help='Maximum tokens per chunk (default: 512, controlled by tokenizer)'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='gpt2',
        help='Tokenizer to use for calculating token lengths (default: gpt2)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset/fineweb-edu-10BT',
        help='Output directory'
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Use offline mode (use cached dataset)'
    )

    args = parser.parse_args()

    # 如果启用离线模式，立即设置环境变量（在导入datasets之前）
    if args.offline:
        import os
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        # 禁用镜像站，使用本地缓存
        if 'HF_ENDPOINT' in os.environ:
            del os.environ['HF_ENDPOINT']
        print("Offline mode enabled: using cached datasets only")

    process_dataset(
        dataset_name=args.dataset,
        subset=args.subset,
        split=args.split,
        max_samples=args.max_samples,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
