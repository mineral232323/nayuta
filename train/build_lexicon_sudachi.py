"""SudachiDict 全エントリ + コーパス頻度から、読み→表記の辞書を作り直す。

コーパス由来だけだと「にほん → 日本」のように、コーパス中で別の読み
（ニッポン）でしか出現しなかった表記が抜ける。SudachiDict の全エントリから
(読み, 表記) の組を集め、頻度はコーパスで分かるものを使い、
分からないものは既定コストを与える。

出力は conv.Lexicon 互換（spans / unk_cost を持つ）。
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from train.conv import Candidate, Lexicon, is_japanese_char, kata2hira


def kana_ok(s: str) -> bool:
    return bool(s) and all(
        0x3041 <= ord(c) <= 0x3096
        or 0x30A1 <= ord(c) <= 0x30FA
        or ord(c) == 0x30FC
        for c in s
    )


def load_corpus_counts(path: str) -> Dict[str, Counter]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rd, surf, c = line.rstrip("\n").split("\t")
            counts[rd][surf] += int(c)
    return counts


def build(
    corpus_counts: Dict[str, Counter],
    topk: int = 16,
    default_cost: float = 14.0,
    unk_cost: float = 25.0,
    min_surface_len: int = 1,
) -> Lexicon:
    from sudachipy import dictionary

    d = dictionary.Dictionary()

    seen: set[Tuple[str, str]] = set()
    pairs: List[Tuple[str, str]] = []
    skip_pos = ("記号", "空白", "補助記号")
    for m in d.entries():
        pos = m.part_of_speech()
        if pos[0] in skip_pos:
            continue
        if len(pos) > 1 and pos[1] == "数詞":
            continue
        surf = m.surface()
        if not surf or len(surf) < min_surface_len:
            continue
        if not any(is_japanese_char(c) for c in surf):
            continue
        rd = kata2hira(m.reading_form() or surf)
        if not kana_ok(rd):
            continue
        key = (rd, surf)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    total = sum(c for cnt in corpus_counts.values() for c in cnt.values()) or 1
    lex = Lexicon({}, topk=topk, unk_cost=unk_cost)
    by_reading: Dict[str, List[Candidate]] = {}
    n_attested = 0
    for rd, surf in pairs:
        cnt = corpus_counts.get(rd, {}).get(surf, 0)
        if cnt > 0:
            cost = -math.log(cnt / total)
            n_attested += 1
        else:
            cost = default_cost
        by_reading.setdefault(rd, []).append(Candidate(surf, cnt, cost))

    for rd, cands in by_reading.items():
        cands.sort(key=lambda c: (c.cost, -c.count, c.surface))
        by_reading[rd] = cands[:topk]

    lex.entries = by_reading
    lex.max_len = max((len(r) for r in by_reading), default=1)
    trie: dict = {}
    for reading in by_reading:
        node = trie
        for ch in reading:
            node = node.setdefault(ch, {})
        node[""] = reading
    lex.trie = trie

    print(
        f"pairs={len(pairs)} readings={len(by_reading)} attested={n_attested} "
        f"total_tokens={total}",
        flush=True,
    )
    return lex
