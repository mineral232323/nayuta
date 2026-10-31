"""コーパスから単語 bigram を作る（連接コスト用）。

  python -m train.build_bigram --top-words 60000 --min-count 3

出力:
  data/bigram.tsv   prev_id \t cur_id \t コスト(*100)
  data/bigram_words.tsv  word_id \t 表記

コストは -log P(w2|w1)。未観測の連接は「unigram コスト + バックオフ罰則」で近似する。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="data/bigram.tsv")
    ap.add_argument("--words-out", default="data/bigram_words.tsv")
    ap.add_argument("--top-words", type=int, default=60000)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()

    # 1) 表記の頻度
    surf: Counter = Counter()
    n_lines = 0
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for _, s in rec["words"]:
                surf[s] += 1
            n_lines += 1
            if n_lines % 500000 == 0:
                print(f"  pass1 {n_lines}", flush=True)
    total = sum(surf.values())
    print(f"sentences={n_lines} tokens={total} distinct={len(surf)}", flush=True)

    words = [w for w, c in surf.most_common(args.top_words) if c >= 2]
    word_id: Dict[str, int] = {w: i for i, w in enumerate(words)}
    V = len(words)
    print(f"bigram vocabulary: {V}", flush=True)

    # 2) bigram の頻度
    bi: Counter = Counter()
    uni: Counter = Counter()
    n_lines = 0
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prev = -1
            for _, s in rec["words"]:
                cur = word_id.get(s, -1)
                if cur >= 0:
                    uni[cur] += 1
                    if prev >= 0:
                        bi[prev * V + cur] += 1
                prev = cur
            n_lines += 1
            if n_lines % 500000 == 0:
                print(f"  pass2 {n_lines} bigrams={len(bi)}", flush=True)

    print(f"bigram types: {len(bi)}", flush=True)
    kept = [(k, c) for k, c in bi.items() if c >= args.min_count]
    print(f"bigram types (>= {args.min_count}): {len(kept)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.words_out, "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            f.write(f"{i}\t{w}\t{surf[w]}\n")
    with open(args.out, "w", encoding="utf-8") as f:
        for key, c in kept:
            prev, cur = divmod(key, V)
            cost = -math.log(c / uni[prev])
            f.write(f"{prev}\t{cur}\t{int(round(cost * 100))}\t{c}\n")
    print(f"-> {args.out} ({os.path.getsize(args.out)/1e6:.2f}MB)")
    print(f"-> {args.words_out}")

    # 参考: バックオフ率
    print(f"bigram coverage: {sum(c for _, c in kept) / max(sum(uni.values()), 1):.3f}")


if __name__ == "__main__":
    main()
