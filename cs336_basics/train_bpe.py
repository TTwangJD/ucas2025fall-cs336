from typing import Iterable, Iterator, List, Dict, Tuple
import os
import regex as re
from array import array
import heapq
from collections import defaultdict, Counter
from functools import total_ordering
import json

GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def pretokenize(text: str) -> list[bytes]:
    """使用GPT-2的正则表达式将文本分割成“词块”，并编码为bytes。 This step is very important!!!! Otherwise the b'a\n\nb' will be transfer into 'a' '\n\n' 'b' instead of 'a' '\n' '\n' 'b'"""
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens


GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)


def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")


class BPETokenizer:
    """字节对编码（Byte Pair Encoding）分词器
    
    BPE是一种无损的数据压缩算法，可用于构建固定大小的词汇表：
    1. 从字节级别开始（256个初始token）
    2. 迭代地合并最频繁的相邻token对
    3. 直到达到目标词表大小
    
    这个实现使用链表和堆数据结构优化了合并过程的性能。
    """
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        """初始化BPE分词器
        
        Args:
            vocab_size: 目标词表大小
            special_tokens: 特殊token列表（如 <|endoftext|>），这些token在处理时保持完整
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]

        # 合并历史记录：存储所有执行的 (byte_p1, byte_p2) 合并
        self.merges: List[Tuple[bytes, bytes]] = []
        # string to id: token字节串 -> token ID的映射
        self.stoi: Dict[bytes, int] = {}
        # id to string: token ID -> token字节串的映射
        self.itos: Dict[int, bytes] = {}
        # 合并操作的排名：(p1, p2) -> 在合并历史中的位置
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

        # ===== 初始化词汇表 =====
        # 1. 添加特殊token到词表开头
        for i, token_bytes in enumerate(self.special_tokens_bytes):
            self.stoi[token_bytes] = i
            self.itos[i] = token_bytes

        # 2. 添加所有256个单字节token
        offset = len(self.special_tokens_bytes)
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])

        # 备份当前词汇表（用于序列化）
        self.vocab = self.itos.copy()
        # 初始化为空（训练后填充）
        self.merges_rank = {}
        # 词对快速查找表：(p1, p2) -> 合并后的token_id
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}

    # def _get_stats(self, token_groups: list[list[int]]):
    #     """Count the frequency of occurrence of all byte pairs."""
    #     pair_counts = {}
    #     for group in token_groups:
    #         for pair in zip(group, group[1:]):
    #             pair_counts[pair] = pair_counts.get(pair, 0) + 1
    #     return pair_counts

    def _merge_pair_in_groups(
        self, ids_group: list[list[int]], pair_to_merge: tuple[int, int], new_id: int
    ):
        """在所有词组中执行单次词对合并操作
        
        Args:
            ids_group: 词组列表，每个词组是token ID的列表
            pair_to_merge: 要合并的词对 (p1_id, p2_id)
            new_id: 合并后的新token ID
            
        Returns:
            合并后的词组列表
        """
        new_ids_group = []
        for group in ids_group:
            new_group = []
            i = 0
            # 逐个扫描当前词组中的token
            while i < len(group):
                # 检查是否找到要合并的词对
                if i < len(group) - 1 and (group[i], group[i + 1]) == pair_to_merge:
                    # 找到词对，用新token替换，跳过两个原token
                    new_group.append(new_id)
                    i += 2
                else:
                    # 不匹配，保留原token
                    new_group.append(group[i])
                    i += 1
            new_ids_group.append(new_group)
        return new_ids_group

    def fast_train(self, path: str | os.PathLike):
        """高效的BPE词表训练方法，使用链表和堆数据结构优化性能
        
        算法流程：
        1. 从文件读取文本内容
        2. 处理特殊token（保持其完整性）
        3. 预分词：使用GPT-2正则将文本分割为词块
        4. 初始化链表结构：用链表表示词序列，便于高效更新
        5. 构建优先队列：以词对频率排序，快速找到最常见的词对
        6. 迭代合并：
           - 从堆中弹出最高频词对
           - 验证其有效性（去重）
           - 合并词对，更新链表
           - 增量更新相邻词对的频率
           - 重复直到达到目标词表大小
        
        Args:
            path: 训练文本文件路径
        """
        def bytes_desc(b):
            """生成字节序列的逆序描述（用于排序）
            
            将每个字节反转（255-x），用于在堆中实现稳定的字节级排序
            """
            return bytes(255 - x for x in b)

        def pair_desc(pair):
            """为词对生成可排序的描述，用于打破堆中的频率平局
            
            Args:
                pair: (token_id1, token_id2) 的元组
                
            Returns:
                (bytes_desc_p1, bytes_desc_p2) 元组，用于字典序比较
            """
            a = self.itos[pair[0]]
            b = self.itos[pair[1]]
            max_len = 2
            # 填充到固定长度以保持字节级比较的一致性
            a_pad = a + bytes([0] * (max_len - len(a)))
            b_pad = b + bytes([0] * (max_len - len(b)))
            return (bytes_desc(a_pad), bytes_desc(b_pad))

        # 检查目标词表大小的有效性
        assert self.vocab_size >= len(self.stoi)

        # ========== 步骤1：读取文本 ==========
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # ========== 步骤2：处理特殊token ==========
        if self.special_tokens:  # 如果有特殊token定义
            # 构建正则表达式来识别特殊token
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            # 分割文本，保留特殊token作为单独部分
            text_parts = re.split(special_pattern, text)
        else:
            text_parts = [text]

        # ========== 步骤3：预分词 ==========
        # 构建逆映射：byte -> token_id（用于将字节转换为初始ID）
        initial_vocab_map = {v: k for k, v in self.itos.items()}
        token_groups = []
        for part in text_parts:
            # 跳过特殊token和空部分
            if part in self.special_tokens or not part:
                continue
            # 使用GPT-2正则将文本分割为词块（保持字节完整性）
            words_in_bytes = pretokenize(part)
            # 将每个词块转换为token ID序列
            for word in words_in_bytes:
                token_groups.append([initial_vocab_map[bytes([b])] for b in word])

        # ========== 步骤4：初始化链表结构 ==========
        idx = 0  # 全局token位置计数器
        pair_counts = {}  # 词对频率字典：(p1_id, p2_id) -> count
        token = {}  # token存储：idx -> token_id
        pre = {}  # 链表前驱指针：idx -> pre_idx 或 None
        nxt = {}  # 链表后继指针：idx -> nxt_idx 或 None
        pos = defaultdict(set)  # 词对位置集合：(p1_id, p2_id) -> {idx_set}

        # 初始化链表：为每个词组中的token建立双向链表
        for i, token_lst in enumerate(token_groups):
            if not token_lst or len(token_lst) <= 1:
                continue
            token_lst_len = len(token_lst)
            for j, token_id in enumerate(token_lst):
                idx += 1
                # 存储token
                token[idx] = token_id
                # 建立链表指针
                nxt[idx] = None if j == token_lst_len - 1 else idx + 1
                pre[idx] = None if j == 0 else idx - 1
                
                # 跳过最后一个token（没有后继词对）
                if j == token_lst_len - 1:
                    continue
                
                # 记录相邻token形成的词对及其位置
                token_pair = (token_id, token_lst[j + 1])
                pair_counts[token_pair] = pair_counts.get(token_pair, 0) + 1
                pos[token_pair].add(idx)

        # ========== 步骤5：构建优先队列（最大堆） ==========
        # 使用负频率实现最大堆（Python heapq 是最小堆）
        heap = [
            (
                -cnt,  # 频次取负，频率越高数值越小（优先出堆）
                pair_desc((a, b)),  # 字节级描述（用于打破频率平局）
                a, b,  # 词对的两个token ID
            )
            for (a, b), cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

        # ========== 嵌套函数：增量更新词对频率 ==========
        def update_pair(pair: tuple[int, int], delta: int, pos_idx: int | None = None):
            """更新词对计数，并可选地更新特定位置
            
            Args:
                pair: (p1_id, p2_id) 词对
                delta: 频率变化量（+1 或 -1）
                pos_idx: 可选，需要更新位置集合的特定索引
            """
            if pair is None or None in pair: 
                return
            
            # 更新频率
            pair_counts[pair] = pair_counts.get(pair, 0) + delta
            cnt = pair_counts[pair]
            
            # 如果频率降到0或以下，删除该词对
            if cnt <= 0:
                pair_counts.pop(pair, None)
                pos.pop(pair, None)
                return
            
            # 更新特定位置的记录
            if pos_idx is not None:
                ds = pos.setdefault(pair, set())
                if delta > 0:
                    ds.add(pos_idx)
                elif delta < 0:
                    ds.discard(pos_idx)
            
            # 将更新后的词对推入堆（允许重复，依靠验证去重）
            a, b = pair
            heapq.heappush(heap, (-cnt, pair_desc((a, b)), a, b))

        # ========== 步骤6：主合并循环 ==========
        num_merges_needed = self.vocab_size - len(self.stoi)  # 需要进行的合并次数
        
        while num_merges_needed > 0 and heap and len(heap) > 0:
            if not pair_counts: 
                break
            num_merges_needed -= 1
            
            # 内层循环：从堆中找到有效的词对
            while heap and len(heap) > 0:
                # 从堆中弹出最高频的词对
                neg_cnt, _, p1, p2 = heapq.heappop(heap)
                cnt = -neg_cnt
                
                # 验证该词对是否仍然有效（去除过期的堆元素）
                if (p1, p2) not in pair_counts or pair_counts[(p1, p2)] != cnt:
                    continue  # 已经被合并过了，跳过此元素

                # ===== 执行合并 =====
                # 记录合并操作（用于后续推理）
                self.merges.append((self.itos[p1], self.itos[p2]))

                # 创建新token
                p1_bytes, p2_bytes = self.itos[p1], self.itos[p2]
                new_token_bytes = p1_bytes + p2_bytes
                new_token_id = (
                    len(self.stoi)
                    if self.stoi.get(new_token_bytes) is None
                    else self.stoi[new_token_bytes]
                )
                self.stoi[new_token_bytes] = new_token_id
                self.itos[new_token_id] = new_token_bytes

                # ===== 更新链表结构 =====
                pos_lst = list(pos.get((p1, p2), set()))
                # 对该词对出现的每个位置进行链表更新
                for pos_idx in pos_lst:
                    pre_idx = pre[pos_idx]  # 前驱位置
                    nxt_idx = nxt[pos_idx]  # 后继位置（应该是p2）
                    nnxt_idx = nxt[nxt_idx] if nxt_idx is not None else None  # 后后继位置

                    # 验证链表中该位置仍然是要合并的词对
                    if nxt_idx is None or token[pos_idx] != p1 or token[nxt_idx] != p2: 
                        continue

                    # ===== 更新前驱关系 =====
                    if pre_idx is not None:
                        # 前驱指针保持不变（指向新token）
                        nxt[pre_idx] = pos_idx  # 前驱的后继变为pos_idx
                        # 更新受影响的词对频率
                        update_pair((token[pre_idx], token[pos_idx]), -1, pre_idx)
                        update_pair((token[pre_idx], new_token_id), 1, pre_idx)
                    
                    # ===== 更新后继关系 =====
                    if nnxt_idx is not None:
                        pre[nnxt_idx] = pos_idx  # 后后继的前驱变为pos_idx
                        # 更新受影响的词对频率
                        update_pair((token[nxt_idx], token[nnxt_idx]), -1, nxt_idx)
                        update_pair((new_token_id, token[nnxt_idx]), 1, pos_idx)
                    
                    # ===== 更新当前位置的state =====
                    pre[pos_idx] = pre_idx
                    nxt[pos_idx] = nnxt_idx
                    token[pos_idx] = new_token_id  # 替换为新token
                    
                    # ===== 删除已合并的原后继token =====
                    token[nxt_idx] = None  # 标记为已删除
                    pre[nxt_idx] = None
                    nxt[nxt_idx] = None
                    
                # 清空已合并词对的计数和位置记录
                pair_counts.pop((p1, p2), None)
                pos.pop((p1, p2), None)
                break  # 成功合并一个词对，回到外层循环

        # ========== 步骤7：后处理和序列化 ==========
        # 构建合并顺序的排名字典（用于推理时恢复合并顺序）
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}
        # 保存最终词汇表
        self.vocab = self.itos.copy()
        # 构建快速查找表：词对 -> 合并后的token ID
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}
