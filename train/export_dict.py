"""端末同梱用の辞書アセットを書き出す。

入力: data/lexicon_v2.tsv （読み \t 表記 \t 頻度 \t コスト*100）
出力: android/app/src/main/assets/dict/lexicon.tsv.gz

形式（TSV, gzip）:
  #NayutaDict  v2
  読み(ひらがな) \t 表記 \t コスト(int, = -log P * 100)
読みは端末側で二分探索するため UTF-16 コード単位順に昇順ソートしておく。
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon_v2.tsv")
    ap.add_argument("--out", default="android/app/src/main/assets/dict/lexicon.tsv.gz")
    ap.add_argument("--topk", type=int, default=8, help="読みあたりの候補数上限")
    args = ap.parse_args()

    ents: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with open(args.lexicon, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rd, surf = parts[0], parts[1]
            if len(parts) >= 4:
                cost = int(parts[3])
            else:
                cost = 0
            ents[rd].append((cost, surf))

    rows: List[Tuple[str, str, int]] = []
    for rd, lst in ents.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        for cost, surf in lst[: args.topk]:
            rows.append((rd, surf, cost))

    rows.sort(key=lambda r: (r[0].encode("utf-16-be"), r[1].encode("utf-16-be")))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = 0
    with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("#NayutaDict\tv2\t%d\n" % len(rows))
        for rd, surf, cost in rows:
            line = f"{rd}\t{surf}\t{cost}\n"
            raw += len(line.encode("utf-8"))
            f.write(line)

    size = os.path.getsize(args.out)
    print(f"entries={len(rows)} readings={len(ents)}")
    print(f"raw={raw/1e6:.1f}MB -> gz={size/1e6:.2f}MB  ({args.out})")


if __name__ == "__main__":
    main()
