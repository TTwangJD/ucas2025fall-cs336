import os
import regex as re
import heapq
from collections import Counter
from collections import defaultdict

def merge_at_word_level(word_structure, p1, p2, new_id, freq):
    """
    在单个单词的双向链表结构中执行合并操作。
    
    参数:
        word_structure: [vals, prevs, nxts] 列表，存储单词的链表表示
        p1, p2: 待合并的两个 Token ID
        new_id: 合并后的新 Token ID
        freq: 该单词在语料中出现的频率
    
    返回:
        changes: dict, 记录受影响的 Pair 及其频率变化量 (delta)
    """
    vals, prevs, nxts = word_structure
    changes = defaultdict(int)
    
    i = 0
    while i != -1:
        # 检查当前位置 i 和下一个位置 nxts[i] 是否匹配 best_pair
        if nxts[i] != -1 and vals[i] == p1 and vals[nxts[i]] == p2:
            j = nxts[i]    # 第二个节点
            k = nxts[j]    # 第二个节点的后继 (可能为 -1)
            h = prevs[i]   # 第一个节点的前驱 (可能为 -1)

            # --- 1. 记录即将失效的 Pair 频率减量 ---
            if h != -1:
                # 前驱与 p1 组成的 pair 将消失
                changes[(vals[h], vals[i])] -= freq
            if k != -1:
                # p2 与后继组成的 pair 将消失
                changes[(vals[j], vals[k])] -= freq
            
            # --- 2. 执行双向链表原地合并 (In-place) ---
            vals[i] = new_id  # 第一个节点原地替换为新 ID
            nxts[i] = k       # 指向后继的后继，跳过节点 j
            if k != -1:
                prevs[k] = i  # 后继的前驱指向合并后的节点 i
            
            # --- 3. 记录新生成的 Pair 频率增量 ---
            if h != -1:
                # 前驱与新 ID 组成新 pair
                changes[(vals[h], vals[i])] += freq
            if k != -1:
                # 新 ID 与后继组成新 pair
                changes[(vals[i], vals[k])] += freq
            
            # 合并后不移动 i，继续检查新生成的节点 i 是否能与 k 再次合并
            # 处理类似 (a, a) 合并 aaa 的情况
            continue 
        
        i = nxts[i] # 移动到下一个节点
    return changes

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    高性能 BPE 训练流程。
    """
    # --- 阶段 1: 初始化基础词表 ---
    itos: dict[int, bytes] = {}
    stoi: dict[bytes, int] = {}
    sp_token_set = set()#用来检查special token是否在语料中出现
    
    # 首先分配特殊 Token (ID 从 0 开始)
    for i, token in enumerate(special_tokens):
        t_bytes = token.encode("utf-8")
        itos[i] = t_bytes
        stoi[t_bytes] = i
        sp_token_set.add(t_bytes)
        
    # 接着分配 256 个基础字节 ID
    offset = len(special_tokens)
    for b in range(256):
        t_bytes = bytes([b])
        itos[offset + b] = t_bytes
        stoi[t_bytes] = offset + b

    # --- 阶段 2: 语料库预处理与词频压缩 ---
    # GPT2 分词正则：确保不会跨单词边界合并
    GPT2_SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    
    word_freqs = Counter()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks = re.findall(GPT2_SPLIT_PATTERN, line)
            for chunk in chunks:
                if chunk.encode("utf-8") in sp_token_set:
                    continue
                ids = [stoi[bytes([b])] for b in chunk.encode("utf-8")]
                if ids:
                    word_freqs[tuple(ids)] += 1

    # --- 阶段 3: 构建链表结构与 Pair 索引 ---
    list_structures = {}     # 存储每个词的 [vals, prevs, nxts]
    pair_counts = defaultdict(int) # Pair 频率全局计数
    pair_to_words = defaultdict(set) # 索引：哪些单词包含该 Pair
    
    for word_tuple, freq in word_freqs.items():#word_tuple 是 tuple 类型的 ids
        n = len(word_tuple)
        vals = list(word_tuple)#从tuple还原为 list 即 ids
        prevs = list(range(-1, n - 1))
        nxts = list(range(1, n)) + [-1]
        list_structures[word_tuple] = [vals, prevs, nxts]
        
        for i in range(n - 1):
            p = (vals[i], vals[i+1])
            pair_counts[p] += freq
            pair_to_words[p].add(word_tuple)

    # 初始化大根堆
    heap = [(-count, p) for p, count in pair_counts.items() if count > 0]
    heapq.heapify(heap)

    # --- 阶段 4: 迭代合并 ---
    merges: list[tuple[bytes, bytes]] = []
    max_merges = vocab_size - len(itos)
    
    for _ in range(max_merges):
        # 获取当前最频繁且有效的 pair
        best_pair = None
        while heap:
            neg_f, p = heapq.heappop(heap)
            if -neg_f == pair_counts.get(p, 0) and -neg_f > 0:
                best_pair = p
                break
        if not best_pair: break

        # 记录合并规则
        merges.append((itos[best_pair[0]], itos[best_pair[1]]))
        new_id = len(itos)
        itos[new_id] = itos[best_pair[0]] + itos[best_pair[1]]

        # --- 核心优化：仅处理包含该 pair 的单词 ---
        affected_words = pair_to_words[best_pair]
        # 注意：best_pair 被合并后，在该集合对应的单词里频率将降为 0，需清空索引
        pair_to_words[best_pair] = set() 
        
        for word_tuple in list(affected_words):
            freq = word_freqs[word_tuple]
            # 执行局部合并并获取受影响的 pair 变化
            changes = merge_at_word_level(list_structures[word_tuple], 
                                         best_pair[0], best_pair[1], 
                                         new_id, freq)
            
            # 更新全局计数、堆以及索引
            for p, diff in changes.items():
                if diff == 0: continue
                pair_counts[p] += diff
                if diff > 0:
                    # 频率增加，推入堆并更新索引
                    heapq.heappush(heap, (-pair_counts[p], p))
                    pair_to_words[p].add(word_tuple)
        
        # 将 best_pair 本身标记为已处理
        pair_counts[best_pair] = 0 

    return itos, merges