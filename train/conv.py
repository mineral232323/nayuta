"""かな漢字変換のコア実装（辞書・ラティス・Viterbi）。

モデル非依存の部分。BERT 採点は `extra_cost` コールバックで差し込む。
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_KATA_START, _KATA_END = 0x30A1, 0x30F6


def kata2hira(s: str) -> str:
    """カタカナ -> ひらがな。"""
    out = []
    for ch in s:
        o = ord(ch)
        out.append(chr(o - 0x60) if _KATA_START <= o <= _KATA_END else ch)
    return "".join(out)


def hira2kata(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        out.append(chr(o + 0x60) if 0x3041 <= o <= 0x3096 else ch)
    return "".join(out)


def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def is_japanese_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3041 <= o <= 0x309F  # ひらがな
        or 0x30A1 <= o <= 0x30FF  # カタカナ
        or 0x4E00 <= o <= 0x9FFF  # 漢字
        or o in (0x3005, 0x3006)  # 々 〆
    )


@dataclass(frozen=True)
class Candidate:
    surface: str
    count: int
    cost: float  # -log P（自然対数）


class Lexicon:
    """読み -> 表記候補の辞書。読みのトライで最長一致探索する。"""

    def __init__(
        self,
        entries: Dict[str, List[Tuple[str, int]]],
        topk: int = 32,
        unk_cost: float = 25.0,
    ):
        self.entries: Dict[str, List[Candidate]] = {}
        total = sum(it[1] for lst in entries.values() for it in lst) or 1
        for reading, lst in entries.items():
            # (表記, 頻度) または (表記, 頻度, コスト) の両方を受ける。
            # コストは小さいほど有利。無い場合は頻度から求める。
            is_explicit = len(next(iter(lst))) >= 3 if lst else False
            if is_explicit:
                lst2 = sorted(lst, key=lambda it: (it[2], -it[1], it[0]))[:topk]
                cands = [Candidate(it[0], it[1], float(it[2])) for it in lst2]
            else:
                lst2 = sorted(lst, key=lambda it: (-it[1], it[0]))[:topk]
                cands = [
                    Candidate(it[0], it[1], -math.log(max(it[1], 1) / total))
                    for it in lst2
                ]
            self.entries[reading] = cands
        self.unk_cost = unk_cost
        self.max_len = max((len(r) for r in self.entries), default=1)
        self.trie: dict = {}
        for reading in self.entries:
            node = self.trie
            for ch in reading:
                node = node.setdefault(ch, {})
            node[""] = reading

    def candidates(self, reading: str) -> List[Candidate]:
        return self.entries.get(reading, [])

    def spans(self, reading: str) -> List[Tuple[int, int, List[Candidate]]]:
        """読みの各位置から始まる辞書スパンを列挙する。"""
        n = len(reading)
        res: List[Tuple[int, int, List[Candidate]]] = []
        for i in range(n):
            node = self.trie
            j = i
            while j < n and reading[j] in node:
                node = node[reading[j]]
                j += 1
                r = node.get("")
                if r is not None:
                    res.append((i, j, self.entries[r]))
        return res


# ---------------------------------------------------------------- lattice


@dataclass
class Arc:
    start: int
    end: int
    cand: Candidate
    extra: float = 0.0

    @property
    def cost(self) -> float:
        return self.cand.cost + self.extra


ScoreFn = Callable[[int, int, Sequence[Candidate]], Sequence[float]]


def build_arcs(
    lex: Lexicon,
    reading: str,
    score_fn: Optional[ScoreFn] = None,
) -> Dict[int, List[Arc]]:
    """開始位置 -> 弧リスト。

    score_fn(start, end, candidates) は候補ごとの追加コスト（-β·logP など）を返す。
    未登録の1文字は通過ノードとして高コストで必ず用意する（到達不能を防ぐ）。
    """
    n = len(reading)
    by_start: Dict[int, List[Arc]] = defaultdict(list)
    seen_span = set()
    for i, j, cands in lex.spans(reading):
        extra = score_fn(i, j, cands) if score_fn else [0.0] * len(cands)
        for c, e in zip(cands, extra):
            by_start[i].append(Arc(i, j, c, e))
        seen_span.add((i, j))
    for i, ch in enumerate(reading):
        if (i, i + 1) not in seen_span:
            by_start[i].append(
                Arc(i, i + 1, Candidate(ch, 1, lex.unk_cost), 0.0)
            )
    return by_start


def viterbi(
    reading: str, by_start: Dict[int, List[Arc]]
) -> Tuple[float, List[Arc]]:
    """最小コストのパスを返す（単語 unigram なので厳密解）。"""
    n = len(reading)
    INF = float("inf")
    dp = [INF] * (n + 1)
    back: List[Optional[Arc]] = [None] * (n + 1)
    dp[0] = 0.0
    for i in range(n):
        if dp[i] == INF:
            continue
        for arc in by_start.get(i, ()):
            c = dp[i] + arc.cost
            if c < dp[arc.end]:
                dp[arc.end] = c
                back[arc.end] = arc
    if dp[n] == INF:
        return INF, []
    path: List[Arc] = []
    pos = n
    while pos > 0:
        arc = back[pos]
        if arc is None:
            return INF, []
        path.append(arc)
        pos = arc.start
    path.reverse()
    return dp[n], path


def to_surfaces(path: Sequence[Arc]) -> str:
    return "".join(a.cand.surface for a in path)


def nbest(
    reading: str,
    by_start: Dict[int, List[Arc]],
    n_best: int = 8,
) -> List[Tuple[float, List[Arc]]]:
    """簡易 n-best（forward DP + backward 列挙）。

    M0 では使わないが、再ランク方式に切り替える際に使う。
    """
    n = len(reading)
    INF = float("inf")
    fwd = [INF] * (n + 1)
    fwd[0] = 0.0
    for i in range(n):
        if fwd[i] == INF:
            continue
        for arc in by_start.get(i, ()):
            fwd[arc.end] = min(fwd[arc.end], fwd[i] + arc.cost)

    results: List[Tuple[float, List[Arc]]] = []

    def walk(pos: int, acc: List[Arc]) -> None:
        if len(results) >= n_best * 8:
            return
        if pos == 0:
            results.append((sum(a.cost for a in acc), list(reversed(acc))))
            return
        for i in range(pos):
            if fwd[i] == INF:
                continue
            for arc in by_start.get(i, ()):
                if arc.end == pos:
                    acc.append(arc)
                    walk(i, acc)
                    acc.pop()

    walk(n, [])
    results.sort(key=lambda x: x[0])
    return results[:n_best]
