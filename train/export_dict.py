"""端末同梱用の辞書アセットを書き出す。

入力: data/lexicon.tsv （読み \t 表記 \t 頻度）
出力: android/app/src/main/assets/dict/lexicon.tsv.gz

形式（TSV, gzip）:
  #NayutaDict  v1
  読み(ひらがな) \t 表記 \t コスト(int, = -log P * 100)
コストが小さいほど有利。端末側はここから Trie / Viterbi を組む。

読みは端末側で二分探索するため UTF-16 コード単位順に昇順ソートしておく。
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Lexicon  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon.tsv")
    ap.add_argument("--out", default="android/app/src/main/assets/dict/lexicon.tsv.gz")
    ap.add_argument("--topk", type=int, default=16, help="読みあたりの候補数上限")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-entries", type=int, default=400000)
    args = ap.parse_args()

    entries: dict[str, list[tuple[str, int]]] = {}
    with open(args.lexicon, encoding="utf-8") as f:
        for line in f:
            rd, surf, c = line.rstrip("\n").split("\t")
            c = int(c)
            if c < args.min_count:
                continue
            entries.setdefault(rd, []).append((surf, c))

    lex = Lexicon(entries, topk=args.topk)
    rows: list[tuple[str, str, int]] = []
    for rd, cands in lex.entries.items():
        for cand in cands:
            rows.append((rd, cand.surface, int(round(cand.cost * 100))))

    # 頻度の高い表記を優先して上限まで残す（読み→表記の最良候補が消えないように）
    if len(rows) > args.max_entries:
        keep: list[tuple[str, str, int]] = []
        per_reading = sorted({r[0] for r in rows})
        by_reading: dict[str, list[tuple[str, str, int]]] = {}
        for r in rows:
            by_reading.setdefault(r[0], []).append(r)
        rank = 0
        while len(keep) < args.max_entries and rank < args.topk:
            for rd in per_reading:
                lst = by_reading[rd]
                if rank < len(lst):
                    keep.append(lst[rank])
                    if len(keep) >= args.max_entries:
                        break
            rank += 1
        rows = keep

    # UTF-16 コード単位順（= Kotlin の String.compareTo と同じ順）でソート
    rows.sort(key=lambda r: (r[0].encode("utf-16-be"), r[1].encode("utf-16-be")))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = 0
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("#NayutaDict\tv1\t%d\n" % len(rows))
        for rd, surf, cost in rows:
            line = f"{rd}\t{surf}\t{cost}\n"
            raw += len(line.encode("utf-8"))
            f.write(line)

    size = os.path.getsize(args.out)
    readings = len({r[0] for r in rows})
    print(f"entries={len(rows)} readings={readings}")
    print(f"raw={raw/1e6:.1f}MB -> gz={size/1e6:.2f}MB  ({args.out})")


if __name__ == "__main__":
    main()
