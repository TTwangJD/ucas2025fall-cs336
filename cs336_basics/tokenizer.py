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
    str_tokens = re.findall(GPT2_SPLIT_PATTERN, text)
    byte_tokens = [s.encode("utf-8") for s in str_tokens]
    return byte_tokens

class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [
            token.encode("utf-8") for token in self.special_tokens
        ]

        self.merges: List[Tuple[bytes, bytes]] = []
        self.stoi: Dict[bytes, int] = {}
        self.itos: Dict[int, bytes] = {}
        self.merges_rank: Dict[Tuple[bytes, bytes], int] = {}

        # init vocab
        for i, token_bytes in enumerate(self.special_tokens_bytes):  # special tokens
            self.stoi[token_bytes] = i
            self.itos[i] = token_bytes

        offset = len(self.special_tokens_bytes)  # 单字节 tokens
        for i in range(256):
            self.stoi[bytes([i])] = i + offset
            self.itos[i + offset] = bytes([i])

        self.vocab = self.itos.copy()  # for serialization
        self.merges_rank = {}  # for fast lookup
        # pair2new: (p1, p2) -> new_token_id
        self.pair2new = {(p1, p2): self.stoi[p1 + p2] for (p1, p2) in self.merges}


