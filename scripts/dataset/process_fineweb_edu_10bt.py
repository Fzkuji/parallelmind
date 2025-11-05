"""
处理 FineWeb-Edu 10BT 数据集

功能：
1. 加载 HuggingFace FineWeb-Edu 数据集
2. 去重：使用 MinHash 算法进行精确+近似去重
   - 精确去重：移除完全相同的文本（MD5 hash）
   - 近似去重：移除高度相似的文本（MinHash + Jaccard similarity）
3. 质量过滤：移除包含乱码或重复句子的文本
4. 智能切分：按句子切分，使用 tokenizer 精确控制每段 token 数
5. 保存为 JSON Lines 格式，每行包含 <|im_start|> 和 <|im_end|> 标记

使用方法：
    # 处理全部数据，使用96核并行处理，启用近似去重
    python scripts/dataset/process_fineweb_edu_10bt.py \
        --output-dir dataset/fineweb-edu-10BT \
        --tokenizer gpt2 \
        --num-workers 96 \
        --dedup-threshold 0.85

    # 测试处理（10万样本），只使用精确去重
    python scripts/dataset/process_fineweb_edu_10bt.py \
        --max-samples 100000 \
        --output-dir dataset/fineweb-edu-10BT \
        --tokenizer gpt2 \
        --num-workers 96 \
        --dedup-threshold 1.0
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
from functools import partial


def compute_hash(text: str) -> str:
    """计算文本的 MD5 hash 值用于精确去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def compute_minhash_signature(text: str, num_perm: int = 128) -> Tuple[int, ...]:
    """
    计算文本的 MinHash 签名用于近似去重

    MinHash 可以高效地估算两个文本的 Jaccard 相似度
    相似度高的文本会有相似的 MinHash 签名

    Args:
        text: 输入文本
        num_perm: hash 函数数量（越大越精确，但计算和存储开销越大）

    Returns:
        MinHash 签名（tuple of integers）
    """
    # 使用 3-gram 作为特征
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        """提取 n-gram 特征"""
        text = text.lower()
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams = get_ngrams(text)
    if not ngrams:
        return tuple([0] * num_perm)

    # MinHash 算法
    signature = [float('inf')] * num_perm

    for ngram in ngrams:
        # 对每个 n-gram 使用多个 hash 函数
        for i in range(num_perm):
            # 使用不同的 seed 来模拟不同的 hash 函数
            hash_value = hash((ngram, i)) & 0xffffffff  # 32-bit hash
            signature[i] = min(signature[i], hash_value)

    return tuple(signature)


def estimate_jaccard_similarity(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
    """
    估算两个 MinHash 签名的 Jaccard 相似度

    Returns:
        相似度 (0.0 到 1.0)
    """
    if len(sig1) != len(sig2):
        return 0.0

    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


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


def process_single_item(
    item: Dict,
    tokenizer_name: str,
    max_tokens_per_chunk: int,
    num_perm: int = 128,
) -> Tuple[Optional[str], Optional[Tuple[int, ...]], str, Dict]:
    """
    处理单个数据样本（用于并行处理）

    返回：
        (text_hash, minhash_sig, status, chunks_data)
        - text_hash: 文本的MD5 hash值，用于精确去重
        - minhash_sig: 文本的MinHash签名，用于近似去重
        - status: 处理状态 ('success', 'empty', 'garbage', 'repetitive')
        - chunks_data: 处理后的chunks列表
    """
    # 加载tokenizer（每个进程独立加载）
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text = item.get('text', '').strip()
    if not text:
        return None, None, 'empty', {}

    # 计算hash和MinHash签名用于去重
    text_hash = compute_hash(text)
    minhash_sig = compute_minhash_signature(text, num_perm=num_perm)

    # 质量过滤：检测乱码
    if is_garbage_text(text):
        return text_hash, minhash_sig, 'garbage', {}

    # 质量过滤：检测重复句子
    if has_repetitive_sentences(text):
        return text_hash, minhash_sig, 'repetitive', {}

    # 切分文本
    chunks = create_chunks(text, tokenizer=tokenizer, max_tokens=max_tokens_per_chunk)

    # 过滤太短的chunk
    valid_chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 50]

    return text_hash, minhash_sig, 'success', {'chunks': valid_chunks}


def process_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_tokens_per_chunk: int = 512,
    tokenizer_name: str = "gpt2",
    output_dir: str = "dataset/fineweb-edu-10BT",
    offline: bool = False,
    num_workers: int = 1,
    batch_size: int = 1000,
    dedup_threshold: float = 0.85,
    num_perm: int = 128,
):
    """
    处理数据集的主函数

    Args:
        num_workers: 并行处理的进程数
        batch_size: 每批处理的样本数
        dedup_threshold: 近似去重的相似度阈值 (0.0-1.0)，默认0.85
                         设置为1.0则只使用精确去重（MD5 hash）
                         设置为0.85表示相似度>=85%的文本会被认为是重复
        num_perm: MinHash签名的hash函数数量，越大越精确但计算开销越大
    """
    from datasets import load_dataset

    print(f"Loading tokenizer: {tokenizer_name}")
    # 主进程只需要验证tokenizer可用
    _ = AutoTokenizer.from_pretrained(tokenizer_name)

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
        'exact_duplicates': 0,
        'near_duplicates': 0,
        'garbage': 0,
        'repetitive': 0,
        'chunks_created': 0,
    }

    # 去重用的数据结构
    seen_hashes: Set[str] = set()  # 精确去重
    seen_signatures: List[Tuple[Tuple[int, ...], str]] = []  # 近似去重: (minhash_sig, text_hash)

    # 输出文件
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "train.jsonl"

    print(f"Processing dataset...")
    print(f"Output file: {output_file}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Max tokens per chunk: {max_tokens_per_chunk}")
    print(f"Max samples: {max_samples if max_samples else 'ALL (10BT)'}")
    print(f"Parallel workers: {num_workers}")
    print(f"Batch size: {batch_size}")

    # 显示去重模式
    if dedup_threshold >= 1.0:
        print(f"Deduplication: Exact matching only (MD5 hash)")
    else:
        print(f"Deduplication: Exact + Near-duplicate detection")
        print(f"  - Similarity threshold: {dedup_threshold:.2f}")
        print(f"  - MinHash permutations: {num_perm}")
    print()

    # 创建处理函数（固定tokenizer_name、max_tokens和num_perm参数）
    process_func = partial(
        process_single_item,
        tokenizer_name=tokenizer_name,
        max_tokens_per_chunk=max_tokens_per_chunk,
        num_perm=num_perm
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用多进程池处理数据
        with Pool(processes=num_workers) as pool:
            batch = []
            iterator = iter(dataset)

            # 使用tqdm显示进度
            pbar = tqdm(desc="Processing", total=max_samples, unit="samples")

            while True:
                # 收集一批数据
                try:
                    for _ in range(batch_size):
                        if max_samples and stats['total_processed'] >= max_samples:
                            break
                        item = next(iterator)
                        batch.append(item)
                        stats['total_processed'] += 1
                except StopIteration:
                    # 数据集遍历完成
                    pass

                if not batch:
                    break

                # 并行处理这一批数据
                results = pool.map(process_func, batch)

                # 处理结果
                for text_hash, minhash_sig, status, chunks_data in results:
                    if status == 'empty':
                        continue
                    elif status == 'garbage':
                        stats['garbage'] += 1
                        continue
                    elif status == 'repetitive':
                        stats['repetitive'] += 1
                        continue
                    elif status == 'success':
                        # 精确去重检查（MD5 hash）
                        if text_hash in seen_hashes:
                            stats['exact_duplicates'] += 1
                            continue

                        # 近似去重检查（MinHash similarity）
                        is_near_duplicate = False
                        if dedup_threshold < 1.0 and minhash_sig is not None:
                            for stored_sig, stored_hash in seen_signatures:
                                similarity = estimate_jaccard_similarity(minhash_sig, stored_sig)
                                if similarity >= dedup_threshold:
                                    is_near_duplicate = True
                                    stats['near_duplicates'] += 1
                                    break

                        if is_near_duplicate:
                            continue

                        # 记录已处理的文本
                        seen_hashes.add(text_hash)
                        if dedup_threshold < 1.0 and minhash_sig is not None:
                            seen_signatures.append((minhash_sig, text_hash))

                        # 保存chunks
                        for chunk in chunks_data.get('chunks', []):
                            formatted_text = f"<|im_start|>{chunk.strip()}<|im_end|>"
                            json_line = json.dumps({"text": formatted_text}, ensure_ascii=False)
                            f.write(json_line + '\n')
                            stats['chunks_created'] += 1

                # 更新进度条
                pbar.update(len(batch))

                # 清空batch，准备下一轮
                batch = []

                # 检查是否达到最大样本数
                if max_samples and stats['total_processed'] >= max_samples:
                    break

            pbar.close()

    # 打印统计信息
    print("\n" + "="*60)
    print("Processing completed!")
    print("="*60)
    print(f"Total samples processed: {stats['total_processed']:,}")
    print(f"Exact duplicates removed: {stats['exact_duplicates']:,}")
    if dedup_threshold < 1.0:
        print(f"Near-duplicates removed (>={dedup_threshold:.0%} similar): {stats['near_duplicates']:,}")
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
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers for processing (default: 1, set to 96 for 128-core server)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for parallel processing (default: 1000)'
    )
    parser.add_argument(
        '--dedup-threshold',
        type=float,
        default=0.85,
        help='Similarity threshold for near-duplicate detection (0.0-1.0). '
             'Set to 1.0 for exact matching only (MD5 hash). '
             'Set to 0.85 to detect texts with >=85%% similarity as duplicates. '
             '(default: 0.85)'
    )
    parser.add_argument(
        '--num-perm',
        type=int,
        default=128,
        help='Number of hash functions for MinHash signature '
             '(more = more accurate but slower, default: 128)'
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
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        dedup_threshold=args.dedup_threshold,
        num_perm=args.num_perm,
    )


if __name__ == "__main__":
    main()
