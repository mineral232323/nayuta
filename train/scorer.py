"""文字レベル BERT による「読みスパン → 表記候補」の採点器。

設計の要点:
- 入力は [CLS] 文脈文字列 [SEP] 読み文字列 [SEP]。文字単位なので読みのスパン位置と
  モデルの位置が 1:1 で対応する。
- 各スパンの表現をプーリングし、候補表記の文字埋め込み平均との相互作用でスコアを出す。
- 候補表記の埋め込み表は持たない（文字埋め込みから構成）ので、辞書が大きくても軽い。
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharVocab:
    """BERT の vocab.txt から文字 -> id を作る。1 文字 = 1 id を保証する。"""

    def __init__(self, vocab_path: str):
        self.chars: Dict[str, int] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.chars[line.rstrip("\n")] = i
        self.unk = self.chars["[UNK]"]
        self.cls = self.chars["[CLS]"]
        self.sep = self.chars["[SEP]"]
        self.pad = self.chars["[PAD]"]
        self.size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        """NFKC 正規化しつつ、元の文字数と 1:1 の id 列を返す。"""
        out: List[int] = []
        for ch in text:
            n = unicodedata.normalize("NFKC", ch)
            key = n if len(n) == 1 else ch
            out.append(self.chars.get(key, self.unk))
        return out


class SpanScorer(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.emb = encoder.get_input_embeddings()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden, 1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    @staticmethod
    def pool_spans(
        hidden: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        """hidden (B,L,H), starts/ends (B,W) -> (B,W,H) の平均プーリング。"""
        B, L, H = hidden.shape
        W = starts.shape[1]
        pos = torch.arange(L, device=hidden.device).view(1, 1, L)
        m = (pos >= starts.unsqueeze(-1)) & (pos < ends.unsqueeze(-1))  # (B,W,L)
        m = m.to(hidden.dtype)
        denom = m.sum(-1).clamp(min=1.0)
        return torch.bmm(m, hidden) / denom.unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        cand_char_ids: torch.Tensor,  # (B,W,K,S)
        cand_char_mask: torch.Tensor,
        cand_index: torch.Tensor,  # (B,W,K) -> グローバル候補テーブルの index
        table_char_ids: torch.Tensor,  # (V,S)
        table_char_mask: torch.Tensor,  # (V,S)
        table_len: torch.Tensor,  # (V,)
    ) -> torch.Tensor:
        hidden = self.encode_text(input_ids, attention_mask)
        h_span = self.pool_spans(hidden, starts, ends)  # (B,W,H)
        B, W, H = h_span.shape
        K = cand_index.shape[-1]

        uid, inv = torch.unique(cand_index.reshape(-1), return_inverse=True)
        ce = self.emb(table_char_ids[uid])  # (U,S,H)
        cm = table_char_mask[uid].unsqueeze(-1).to(ce.dtype)
        ce = (ce * cm).sum(1) / table_len[uid].clamp(min=1.0).unsqueeze(-1)
        c = ce[inv].view(B, W, K, H)

        hs = h_span.unsqueeze(2).expand(-1, -1, K, -1)
        feats = torch.cat([hs, c, hs * c], dim=-1)
        return self.out(self.proj(feats)).squeeze(-1)  # (B,W,K)


def build_candidate_table(
    surfaces: Sequence[str], vocab: CharVocab, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """候補表記の文字 id 行列を作る。"""
    max_len = min(max((len(s) for s in surfaces), default=1), 12)
    ids = torch.full((len(surfaces), max_len), vocab.pad, dtype=torch.long)
    mask = torch.zeros((len(surfaces), max_len), dtype=torch.long)
    lens = torch.ones(len(surfaces), dtype=torch.long)
    for i, s in enumerate(surfaces):
        enc = vocab.encode(s)[:max_len]
        ids[i, : len(enc)] = torch.tensor(enc, dtype=torch.long)
        mask[i, : len(enc)] = 1
        lens[i] = max(len(enc), 1)
    return ids.to(device), mask.to(device), lens.to(device), max_len
