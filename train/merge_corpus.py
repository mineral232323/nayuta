"""複数の prep_data 出力を 1 つにまとめる。

`train/prep_data.py` を複数回・並列に走らせたとき（Wikipedia / web / SNS、
あるいはシャード分割）の train/valid/lexicon を結合する。

  python -m train.merge_corpus \
      --train data/train_v3_wiki.jsonl,data/train_v3_websns.jsonl \
      --valid data/valid_v3_wiki.jsonl,data/valid_v3_websns.jsonl \
      --lexicon data/lexicon_v3corpus_wiki.tsv,data/lexicon_v3corpus_websns.tsv \
      --out-train data/train_v3.jsonl \
      --out-valid data/valid_v3.jsonl \
      --out-lexicon data/lexicon_v3corpus.tsv

train/valid は単純結合（学習側は DataLoader がシャッフルするので順序は問わない）。
lexicon は (読み, 表記) ごとに頻度を合計する。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def split_paths(s: str) -> List[str]:
    return [p for p in s.split(",") if p]


def concat(paths: List[str], out: str, stats_key: str) -> Dict[str, int]:
    counts: Counter = Counter()
    n = 0
    with open(out, "w", encoding="utf-8") as fo:
        for path in paths:
            if not path or not os.path.exists(path):
                print(f"[warn] ありません: {path}", flush=True)
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    fo.write(line)
                    n += 1
                    if n % 500000 == 0:
                        print(f"  {stats_key} {n}", flush=True)
                    try:
                        src = line.rsplit('"src": "', 1)[1].split('"', 1)[0]
                    except IndexError:
                        src = "unknown"
                    counts[src] += 1
    print(f"{stats_key}: {n} -> {out}  by_domain={dict(counts)}", flush=True)
    return {stats_key: n, stats_key + "_by_domain": dict(counts)}


def merge_lexicon(paths: List[str], out: str) -> Dict[str, int]:
    merged: Dict[str, Counter] = defaultdict(Counter)
    for path in paths:
        if not path or not os.path.exists(path):
            print(f"[warn] ありません: {path}", flush=True)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                rd, surf, c = parts[0], parts[1], int(parts[2])
                merged[rd][surf] += c
                n += 1
        print(f"  lexicon {path}: {n} rows", flush=True)

    rows = 0
    with open(out, "w", encoding="utf-8") as fo:
        for rd, cnt in merged.items():
            for surf, c in cnt.items():
                fo.write(f"{rd}\t{surf}\t{c}\n")
                rows += 1
    print(
        f"lexicon: rows={rows} readings={len(merged)} "
        f"-> {out} ({os.path.getsize(out) / 1e6:.1f}MB)",
        flush=True,
    )
    return {"lexicon_rows": rows, "lexicon_readings": len(merged)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="カンマ区切りの jsonl")
    ap.add_argument("--valid", default="")
    ap.add_argument("--lexicon", required=True, help="カンマ区切りの tsv")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-valid", default="")
    ap.add_argument("--out-lexicon", required=True)
    ap.add_argument("--out-stats", default="")
    args = ap.parse_args()

    stats: Dict[str, object] = {}
    stats.update(concat(split_paths(args.train), args.out_train, "train"))
    if args.valid and args.out_valid:
        stats.update(concat(split_paths(args.valid), args.out_valid, "valid"))
    stats.update(merge_lexicon(split_paths(args.lexicon), args.out_lexicon))

    if args.out_stats:
        with open(args.out_stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
