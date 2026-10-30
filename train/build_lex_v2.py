"""読み→表記辞書を作る（M2 版）。

**方針**: コーパスで実際に観測できた (読み, 表記) を土台にし、
その表記が SudachiDict 上で持つ**別読み**を足す。

コーパスだけだと「にほん → 日本」が抜ける（Wikipedia では 日本 は ニッポン 読みで
しか出ない）。かといって SudachiDict の全 138 万組を入れると、未観測語の既定コストが
候補を濁して精度が落ち、端末のメモリも食う。観測済みの表記に対する読み替えだけを
足すのが、被覆と精度のバランスが最も良い。

別読みのコストは「その表記の総頻度」から求め、読み替え分のペナルティを足す:
    cost = -log(表記の総頻度 / 総トークン数) + VARIANT_PENALTY

出力:
  data/lexicon_v2.tsv   読み \t 表記 \t 頻度 \t コスト(*100)
  data/alt_readings.tsv 表記 \t 読み1,読み2,...   （学習時の読み替え拡張用）
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import is_japanese_char, kata2hira  # noqa: E402

SKIP_POS = ("記号", "空白", "補助記号")


def kana_ok(s: str) -> bool:
    return bool(s) and all(
        0x3041 <= ord(c) <= 0x3096 or 0x30A1 <= ord(c) <= 0x30FA or ord(c) == 0x30FC
        for c in s
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-lexicon", default="data/lexicon.tsv")
    ap.add_argument("--out", default="data/lexicon_v2.tsv")
    ap.add_argument("--alt-out", default="data/alt_readings.tsv")
    ap.add_argument("--variant-penalty", type=float, default=2.5)
    ap.add_argument("--max-reading-len", type=int, default=24)
    ap.add_argument("--min-freq", type=int, default=2)
    args = ap.parse_args()

    corpus: Dict[str, Counter] = defaultdict(Counter)
    with open(args.corpus_lexicon, encoding="utf-8") as f:
        for line in f:
            rd, surf, c = line.rstrip("\n").split("\t")
            corpus[rd][surf] += int(c)
    total = sum(c for cnt in corpus.values() for c in cnt.values()) or 1
    surf_total: Counter = Counter()
    for rd, cnt in corpus.items():
        for s, c in cnt.items():
            surf_total[s] += c
    attested = {(rd, s) for rd, cnt in corpus.items() for s in cnt}
    known_surfaces = set(surf_total)

    # SudachiDict から、コーパスに出現する表記の読み一覧を集める
    from sudachipy import dictionary

    d = dictionary.Dictionary()
    surface_readings: Dict[str, List[str]] = defaultdict(list)
    seen = set()
    for m in d.entries():
        pos = m.part_of_speech()
        if pos[0] in SKIP_POS:
            continue
        if len(pos) > 1 and pos[1] == "数詞":
            continue
        surf = m.surface()
        if surf not in known_surfaces:
            continue
        if not any(is_japanese_char(c) for c in surf):
            continue
        rd = kata2hira(m.reading_form() or surf)
        if not kana_ok(rd) or len(rd) > args.max_reading_len:
            continue
        if (surf, rd) in seen:
            continue
        seen.add((surf, rd))
        surface_readings[surf].append(rd)

    rows: List[Tuple[str, str, int, int]] = []
    n_variant = 0
    for s, rds in surface_readings.items():
        for rd in rds:
            if (rd, s) in attested:
                cnt = corpus[rd][s]
                if cnt < args.min_freq:
                    continue
                cost = -math.log(cnt / total)
            else:
                cnt = 0
                cost = -math.log(surf_total[s] / total) + args.variant_penalty
                n_variant += 1
            rows.append((rd, s, cnt, int(round(cost * 100))))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rd, surf, cnt, cost in rows:
            f.write(f"{rd}\t{surf}\t{cnt}\t{cost}\n")

    als = 0
    with open(args.alt_out, "w", encoding="utf-8") as f:
        for surf, rds in surface_readings.items():
            uniq = sorted(set(rds))
            if len(uniq) < 2:
                continue
            f.write(f"{surf}\t{','.join(uniq)}\n")
            als += 1

    print(f"rows={len(rows)} variants={n_variant} alt_surfaces={als}")
    print(f"-> {args.out} ({os.path.getsize(args.out)/1e6:.2f}MB)")


if __name__ == "__main__":
    main()
