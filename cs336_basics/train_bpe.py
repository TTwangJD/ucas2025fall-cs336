import re
import heapq
from collections import Counter, defaultdict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None):
    """
    训练字节级 BPE 分词器。
    返回:
        vocab: dict[int, bytes] - Token ID 到字节序列的映射
        merges: list[tuple[bytes, bytes]] - 训练过程中产生的合并记录
    """
    # 1. 初始化基础词表 (Special Tokens + 256 Bytes)
    special_tokens = special_tokens or []
    vocab = {}
    stoi = {}
    
    # 首先分配特殊 Token
    for i, token in enumerate(special_tokens):
        t_bytes = token.encode("utf-8")
        vocab[i] = t_bytes
        stoi[t_bytes] = i
        
    # 接着分配 256 个基础字节
    offset = len(special_tokens)
    for b in range(256):
        t_bytes = bytes([b])
        vocab[offset + b] = t_bytes
        stoi[t_bytes] = offset + b

    # 2. 读取文件并进行词频压缩 (Pre-aggregation)
    # 使用 GPT-2 类似的正则规则防止非法合并
    GPT2_SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    
    word_freqs = Counter()
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 预分词
            chunks = re.findall(GPT2_SPLIT_PATTERN, line)
            for chunk in chunks:
                # 将每个单词转化为 ID 序列
                ids = tuple(stoi[bytes([b])] for b in chunk.encode("utf-8"))
                word_freqs[ids] += 1

    # 3. 统计初始 Pair 频率并构建大根堆
    pair_counts = defaultdict(int)
    for ids, freq in word_freqs.items():
        for pair in zip(ids[:-1], ids[1:]):
            pair_counts[pair] += freq

    # 堆中存放 (-频率, (id1, id2))
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    # 4. 迭代合并过程
    merges_record = [] # 存储 bytes 形式的合并记录
    num_merges = vocab_size - len(vocab)
    
    for _ in range(num_merges):
        # 寻找当前最高频且有效的 pair
        best_pair = None
        while heap:
            neg_freq, pair = heapq.heappop(heap)
            if -neg_freq == pair_counts.get(pair, 0):
                best_pair = pair
                break
        
        if not best_pair:
            break # 无可合并的 pair

        # 创建新 Token
        new_id = len(vocab)
        token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = token_bytes
        merges_record.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # 5. 更新语料库 (word_freqs) 并更新频率
        new_word_freqs = Counter()
        for ids, freq in word_freqs.items():
            if best_pair not in set(zip(ids[:-1], ids[1:])):
                new_word_freqs[ids] += freq
                continue
            
            # 执行合并逻辑
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            new_ids = tuple(new_ids)
            new_word_freqs[new_ids] += freq
            
            # 局部更新 pair_counts：减去旧 pair，增加新 pair
            # 更新旧的
            for p in zip(ids[:-1], ids[1:]):
                pair_counts[p] -= freq
            # 更新新的
            for p in zip(new_ids[:-1], new_ids[1:]):
                pair_counts[p] += freq
                if pair_counts[p] > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))

        word_freqs = new_word_freqs

    return vocab, merges_record