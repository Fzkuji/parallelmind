import json
from typing import Dict, List, Optional, Tuple, Iterator

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer


class ParallelPretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str = None,
        slice_count: Optional[int] = None,
        slice_index: int = 0,
        # Hugging Face 数据集参数
        hf_dataset: Optional[str] = None,
        hf_subset: Optional[str] = None,
        hf_split: str = "train",
        text_column: Optional[str] = None,
        max_samples: Optional[int] = None,
        # 文本切分参数
        chunk_length: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.offsets: List[int] = []
        self._file_handle = None
        self.slice_count = slice_count
        self.slice_index = slice_index
        self.use_hf = hf_dataset is not None
        self.hf_data: List[Dict[str, str]] = []
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer

        if self.use_hf:
            # 从 Hugging Face 加载数据
            self._load_from_huggingface(hf_dataset, hf_subset, hf_split, text_column, max_samples)
            total_samples = len(self.hf_data)
        else:
            # 从本地文件加载
            if not data_path:
                raise ValueError("必须指定 data_path 或 hf_dataset")

            # 只构建索引，不加载数据
            with open(data_path, "r", encoding="utf-8") as handle:
                offset = 0
                for line in handle:
                    if line.strip():
                        # 如果设置了 max_samples，限制加载的样本数量
                        if max_samples and len(self.offsets) >= max_samples:
                            break
                        self.offsets.append(offset)
                    offset += len(line.encode('utf-8'))

            total_samples = len(self.offsets)

        if self.slice_count is not None:
            if self.slice_count <= 0:
                raise ValueError(f"slice_count must be positive, got {self.slice_count}")
            if not (0 <= self.slice_index < self.slice_count):
                raise ValueError(
                    f"slice_index must be in [0, {self.slice_count - 1}], got {self.slice_index}"
                )

            if self.use_hf:
                # 对HF数据应用切分
                original_data = self.hf_data
                self.hf_data = [
                    sample
                    for sample_idx, sample in enumerate(original_data)
                    if sample_idx % self.slice_count == self.slice_index
                ]
            else:
                # 对文件偏移应用切分
                original_offsets = self.offsets
                self.offsets = [
                    offset_value
                    for sample_idx, offset_value in enumerate(original_offsets)
                    if sample_idx % self.slice_count == self.slice_index
                ]

            remaining = len(self.hf_data) if self.use_hf else len(self.offsets)
            print(
                f"数据集初始化完成: {total_samples:,} 个样本 | 应用 branch 切分 {self.slice_index + 1}/{self.slice_count} 后剩余 {remaining:,} 个样本"
            )
        else:
            print(f"数据集初始化完成: {total_samples:,} 个样本")

    def _load_from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        text_column: Optional[str],
        max_samples: Optional[int]
    ) -> None:
        """从 Hugging Face 加载数据集"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("需要安装 datasets 库: pip install datasets")

        print(f"正在从 Hugging Face 加载数据集: {dataset_name}")
        if subset:
            print(f"  子集: {subset}")
        print(f"  分割: {split}")

        # 加载数据集
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            raise RuntimeError(f"加载 Hugging Face 数据集失败: {e}")

        # 检测文本列
        first_item = next(iter(dataset))
        available_columns = list(first_item.keys())
        print(f"  可用列: {available_columns}")

        if text_column is None:
            possible_columns = ['text', 'content', 'document', 'sentence', 'data']
            text_column = next((col for col in possible_columns if col in available_columns), available_columns[0])
            print(f"  自动检测文本列: {text_column}")

        # 重新创建迭代器并加载数据
        if subset:
            dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=True)

        # 提取数据
        count = 0
        total_chunks = 0
        use_chunking = self.chunk_length is not None and self.tokenizer is not None

        if use_chunking:
            print(f"  启用文本切分: 每个片段最大 {self.chunk_length} tokens")

        for item in dataset:
            if max_samples and count >= max_samples:
                break

            text_content = item.get(text_column, "")
            if isinstance(text_content, str) and text_content.strip():
                text = text_content.strip()

                if use_chunking:
                    # 切分长文本
                    chunks = self._chunk_text(text)
                    self.hf_data.extend([{"text": chunk} for chunk in chunks])
                    total_chunks += len(chunks)
                else:
                    # 不切分，直接存储
                    self.hf_data.append({"text": text})

                count += 1

                if count % 1000 == 0:
                    if use_chunking:
                        print(f"  已处理 {count:,} 个原始样本 -> {total_chunks:,} 个片段...")
                    else:
                        print(f"  已加载 {count:,} 个样本...")

        if use_chunking:
            print(f"✓ 从 Hugging Face 加载完成: {count:,} 个原始样本 -> {len(self.hf_data):,} 个切分片段")
        else:
            print(f"✓ 从 Hugging Face 加载完成: {len(self.hf_data):,} 个样本")

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本切分成多个固定长度的片段"""
        if not self.tokenizer or not self.chunk_length:
            return [text]

        # Tokenize 整个文本（临时抑制超长警告，因为我们会切分）
        import logging
        transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
        old_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        finally:
            transformers_logger.setLevel(old_level)

        # 如果文本短于 chunk_length，直接返回
        if len(tokens) <= self.chunk_length:
            return [text]

        # 切分成多个 chunks
        chunks = []
        for i in range(0, len(tokens), self.chunk_length):
            chunk_tokens = tokens[i:i + self.chunk_length]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():  # 只保留非空片段
                chunks.append(chunk_text.strip())

        return chunks

    def __len__(self) -> int:
        return len(self.hf_data) if self.use_hf else len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, str]:
        if self.use_hf:
            # 从缓存的HF数据读取
            return self.hf_data[index]
        else:
            # 懒加载：只在需要时读取
            # 保持文件句柄打开以提高性能
            if self._file_handle is None:
                self._file_handle = open(self.data_path, "r", encoding="utf-8")

            self._file_handle.seek(self.offsets[index])
            line = self._file_handle.readline()
            data = json.loads(line.strip())
            text = str(data.get("text", ""))
            return {"text": text}

    def __del__(self):
        # 清理文件句柄
        if self._file_handle is not None:
            self._file_handle.close()


class ParallelSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        # Hugging Face 数据集参数
        hf_dataset: Optional[str] = None,
        hf_subset: Optional[str] = None,
        hf_split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_tokens = tokenizer("<|im_start|>assistant", add_special_tokens=False).input_ids
        self.conversations: List[List[Dict[str, str]]] = []
        self.mapping: List[Tuple[int, int]] = []
        self.use_hf = hf_dataset is not None

        if self.use_hf:
            self._load_from_huggingface(hf_dataset, hf_subset, hf_split, max_samples)
        else:
            if not jsonl_path:
                raise ValueError("必须指定 jsonl_path 或 hf_dataset")
            self._index_dataset(jsonl_path)

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        conv_idx, turn_idx = self.mapping[index]
        history = self.conversations[conv_idx][: turn_idx + 1]
        prompt_text = self.tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        ).input_ids[: self.max_length]
        answer_offset = self._locate_answer_start(input_ids)
        if len(input_ids) < self.max_length:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_id] * (self.max_length - len(input_ids))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "answer_offset": torch.tensor(answer_offset, dtype=torch.long),
        }

    def _load_from_huggingface(
        self,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        max_samples: Optional[int]
    ) -> None:
        """从 Hugging Face 加载SFT数据集"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("需要安装 datasets 库: pip install datasets")

        print(f"正在从 Hugging Face 加载SFT数据集: {dataset_name}")
        if subset:
            print(f"  子集: {subset}")
        print(f"  分割: {split}")

        # 加载数据集
        try:
            if subset:
                dataset = load_dataset(dataset_name, subset, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            raise RuntimeError(f"加载 Hugging Face 数据集失败: {e}")

        # 提取对话数据
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break

            # 尝试提取对话
            convo = self._extract_conversations(item)
            if convo:
                conv_idx = len(self.conversations)
                self.conversations.append(convo)
                for idx, turn in enumerate(convo):
                    if turn.get("role") == "assistant":
                        self.mapping.append((conv_idx, idx))
                count += 1

                if count % 1000 == 0:
                    print(f"  已加载 {count:,} 个对话...")

        print(f"✓ 从 Hugging Face 加载完成: {len(self.conversations):,} 个对话, {len(self.mapping):,} 个训练样本")

    def _extract_conversations(self, item: Dict) -> Optional[List[Dict[str, str]]]:
        """从不同格式中提取对话"""
        # 标准格式: {"conversations": [{"role": "user", "content": "..."}]}
        if "conversations" in item:
            convs = item["conversations"]
            if isinstance(convs, list) and len(convs) > 0:
                # 检查是否是标准格式
                if all(isinstance(turn, dict) and "role" in turn and "content" in turn for turn in convs):
                    return [{"role": turn["role"], "content": turn["content"]} for turn in convs]

                # ShareGPT格式: {"from": "human", "value": "..."}
                if all(isinstance(turn, dict) and "from" in turn and "value" in turn for turn in convs):
                    result = []
                    for turn in convs:
                        from_role = turn.get("from", "")
                        role = "user" if from_role == "human" else "assistant" if from_role == "gpt" else from_role
                        result.append({"role": role, "content": turn.get("value", "")})
                    return result if result else None

        # OpenAI messages格式: {"messages": [{"role": "user", "content": "..."}]}
        if "messages" in item:
            msgs = item["messages"]
            if isinstance(msgs, list) and len(msgs) > 0:
                if all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in msgs):
                    return [{"role": msg["role"], "content": msg["content"]} for msg in msgs]

        # Alpaca格式: {"instruction": "...", "input": "...", "output": "..."}
        if "instruction" in item and "output" in item:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")

            if instruction or output:
                user_content = f"{instruction}\n{input_text}" if input_text else instruction
                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]

        # 简单格式: {"prompt": "...", "response": "..."}
        if "prompt" in item and "response" in item:
            return [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]}
            ]

        return None

    def _index_dataset(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                convo = [{"role": turn["role"], "content": turn["content"]} for turn in data.get("conversations", [])]
                conv_idx = len(self.conversations)
                self.conversations.append(convo)
                for idx, turn in enumerate(convo):
                    if turn.get("role") == "assistant":
                        self.mapping.append((conv_idx, idx))

    def _locate_answer_start(self, input_ids: List[int]) -> int:
        bos = self.bos_tokens
        for idx in range(len(input_ids)):
            if input_ids[idx : idx + len(bos)] == bos:
                start = idx + len(bos)
                return start
        return 0


class ParallelPretrainIterableDataset(IterableDataset):
    """
    真正的 streaming 数据集，用于 Hugging Face 数据集
    - 不预加载所有数据到内存
    - 训练时按需加载和切分文本
    - 适合大规模数据集（如 FineWeb-Edu）
    """
    def __init__(
        self,
        hf_dataset: str,
        hf_subset: Optional[str] = None,
        hf_split: str = "train",
        text_column: Optional[str] = None,
        max_samples: Optional[int] = None,
        chunk_length: Optional[int] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        offline: bool = False,
    ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset
        self.hf_subset = hf_subset
        self.hf_split = hf_split
        self.text_column = text_column
        self.max_samples = max_samples
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer
        self.offline = offline
        self._token_buffer: list = []  # packing 模式下跨 epoch 保留的尾部 tokens

        # 只在主进程打印初始化信息
        if self._is_main_process():
            print(f"初始化 Streaming 数据集: {hf_dataset}")
            if hf_subset:
                print(f"  子集: {hf_subset}")
            if chunk_length:
                print(f"  Token packing: 每个 chunk 固定 {chunk_length} tokens (短文档拼接，长文档跨 chunk)")
            else:
                print(f"  警告：未设置 chunk_length，长文本不会被切分！")
            if not tokenizer:
                print(f"  警告：未传递 tokenizer，文本切分功能将被禁用！")
            if offline:
                print(f"  离线模式：使用已缓存的数据集")

    def _is_main_process(self) -> bool:
        """检查是否为主进程"""
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def _chunk_text(self, text: str) -> List[str]:
        """将长文本切分成多个固定长度的片段"""
        if not self.tokenizer or not self.chunk_length:
            return [text]

        # Tokenize 整个文本（临时抑制超长警告，因为我们会切分）
        import logging
        transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
        old_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        finally:
            transformers_logger.setLevel(old_level)

        # 如果文本短于 chunk_length，直接返回
        if len(tokens) <= self.chunk_length:
            return [text]

        # 切分成多个 chunks
        chunks = []
        for i in range(0, len(tokens), self.chunk_length):
            chunk_tokens = tokens[i:i + self.chunk_length]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """迭代器：按需加载和处理数据"""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("需要安装 datasets 库: pip install datasets")

        # 加载 streaming 数据集（离线模式在训练脚本启动时已设置环境变量）
        if self.hf_subset:
            dataset = load_dataset(
                self.hf_dataset,
                self.hf_subset,
                split=self.hf_split,
                streaming=True
            )
        else:
            dataset = load_dataset(
                self.hf_dataset,
                split=self.hf_split,
                streaming=True
            )

        # 自动检测文本列（只在第一个样本上检测）
        if self.text_column is None:
            first_item = next(iter(dataset))
            available_columns = list(first_item.keys())
            possible_columns = ['text', 'content', 'document', 'sentence', 'data']
            self.text_column = next(
                (col for col in possible_columns if col in available_columns),
                available_columns[0]
            )
            # 重新创建迭代器
            if self.hf_subset:
                dataset = load_dataset(
                    self.hf_dataset,
                    self.hf_subset,
                    split=self.hf_split,
                    streaming=True
                )
            else:
                dataset = load_dataset(
                    self.hf_dataset,
                    split=self.hf_split,
                    streaming=True
                )

        # 计算当前 worker 的分片参数
        worker_info = torch.utils.data.get_worker_info()
        num_shards = 1
        shard_id = 0

        if torch.distributed.is_initialized():
            # DDP: 每个 rank 处理不同的数据
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            if worker_info is not None:
                # DDP + DataLoader workers
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                # 总进程数 = world_size * num_workers
                num_shards = world_size * num_workers
                shard_id = rank * num_workers + worker_id
            else:
                # 只有 DDP，没有 DataLoader workers
                num_shards = world_size
                shard_id = rank
        elif worker_info is not None:
            # 只有 DataLoader workers，没有 DDP
            num_shards = worker_info.num_workers
            shard_id = worker_info.id

        # 计算当前 shard 的 max_samples（全局总数 / 分片数）
        if self.max_samples and num_shards > 1:
            per_shard_max_samples = self.max_samples // num_shards
            # 如果是最后一个 shard，处理余数
            if shard_id == num_shards - 1:
                per_shard_max_samples += self.max_samples % num_shards

            # 只在主进程打印分片信息
            if self._is_main_process():
                print(f"  数据分片: {num_shards} 个进程，每个进程最多处理 ~{self.max_samples // num_shards:,} 个样本")
        else:
            per_shard_max_samples = self.max_samples

        # 迭代数据并手动分片
        chunk_count = 0  # 当前 shard 已产出的 chunk 数量
        global_idx = 0   # 全局样本索引

        # Packing: 将多个文档拼接为固定长度的 token chunk
        use_packing = self.chunk_length is not None and self.tokenizer is not None
        token_buffer = self._token_buffer  # 复用上次剩余的 tokens

        if use_packing:
            # 一次性抑制 tokenizer 长文本警告
            import logging
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

        for item in dataset:
            # 手动分片：只处理属于当前 shard 的样本
            if global_idx % num_shards != shard_id:
                global_idx += 1
                continue

            text_content = item.get(self.text_column, "")
            if isinstance(text_content, str) and text_content.strip():
                text = text_content.strip()

                if use_packing:
                    # 文档间加 EOS 分隔
                    if token_buffer:
                        eos_id = self.tokenizer.eos_token_id
                        if eos_id is not None:
                            token_buffer.append(eos_id)
                    token_buffer.extend(
                        self.tokenizer.encode(text, add_special_tokens=False)
                    )
                    # 产出所有满长 chunk
                    while len(token_buffer) >= self.chunk_length:
                        if per_shard_max_samples and chunk_count >= per_shard_max_samples:
                            return
                        yield {"input_ids": token_buffer[:self.chunk_length]}
                        token_buffer = token_buffer[self.chunk_length:]
                        chunk_count += 1
                else:
                    if per_shard_max_samples and chunk_count >= per_shard_max_samples:
                        return
                    yield {"text": text}
                    chunk_count += 1

            global_idx += 1
        # 不足 chunk_length 的尾部 tokens 保留在 self._token_buffer，下次迭代继续使用
        self._token_buffer = token_buffer
