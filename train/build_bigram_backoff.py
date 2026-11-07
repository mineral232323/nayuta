"""補間バックオフ付き bigram を作る（語彙を辞書側まで拡張する実験用）。

現行 `train/build_bigram.py` は上位 10 万語だけを語彙にし、未観測の連接は
「unigram コストへハードフォールバック」している。そのため `帰社` のような
辞書にはあるが頻度の低い語は、連接統計を持てない。

このスクリプトは
  * 語彙をコーパス頻度上位 N 語（N=20 万まで可）に広げる
  * 連接コストを Jelinek-Mercer 補間
        P(cur|prev) = (c(prev,cur) + α·P_uni(cur)) / (c(prev) + α)
    で計算し、**未観測の連接もバックオフで滑らかに**扱えるようにする
  * 出力は「生カウント」。α は読み込み側（BackoffBigramLM）で指定する

  python -m train.build_bigram_backoff --vocab-size 200000
  # => logs/bigram_bo200k/{words.tsv, pairs.tsv, meta.json}

  python -m train.build_bigram_backoff --vocab-size 100000 --max-sentences 200000  # 動作確認
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_vocab(
    lexicon_counts: str, vocab_size: int
) -> Tuple[List[str], Dict[str, int]]:
    """(読み, 表記, 頻度) から表記ごとの頻度を合計し、上位 vocab_size を返す。"""
    counts: Counter = Counter()
    with open(lexicon_counts, encoding="utf-8") as f:
        for line in f:
            _rd, surf, cnt = line.rstrip("\n").split("\t")
            counts[surf] += int(cnt)
    words = [w for w, _c in counts.most_common(vocab_size)]
    return words, {w: i for i, w in enumerate(words)}


def iter_sentences(path: str, max_sentences: int) -> Iterable[List[str]]:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_sentences and i >= max_sentences:
                return
            rec = json.loads(line)
            yield [surf for _rd, surf in rec["words"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="補間バックオフ付き bigram を作る")
    ap.add_argument("--train", default="data/train_v3.jsonl")
    ap.add_argument("--lexicon-counts", default="data/lexicon_v3corpus.tsv")
    ap.add_argument("--vocab-size", type=int, default=200_000)
    ap.add_argument("--min-pair-count", type=int, default=2,
                    help="この回数未満の連接は保存しない（バックオフで扱う）")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-sentences", type=int, default=0, help="0 で全件")
    args = ap.parse_args()
    out_dir = args.out_dir or f"logs/bigram_bo{args.vocab_size // 1000}k"
    os.makedirs(out_dir, exist_ok=True)

    words, word_id = load_vocab(args.lexicon_counts, args.vocab_size)
    print(f"vocab={len(words):,} (top {args.vocab_size:,})", flush=True)

    uni: Counter = Counter()
    pairs: Counter = Counter()
    total_tokens = 0
    n_sent = 0
    for tokens in iter_sentences(args.train, args.max_sentences):
        n_sent += 1
        total_tokens += len(tokens)
        prev = None
        for t in tokens:
            tid = word_id.get(t)
            if tid is not None:
                uni[tid] += 1
                if prev is not None:
                    pairs[(prev, tid)] += 1
            prev = tid
        if n_sent % 1_000_000 == 0:
            print(f"  {n_sent:,} sentences  pairs={len(pairs):,}", flush=True)

    print(f"sentences={n_sent:,} tokens={total_tokens:,} "
          f"pairs(all)={len(pairs):,}", flush=True)

    kept = {k: v for k, v in pairs.items() if v >= args.min_pair_count}
    print(f"pairs(kept>={args.min_pair_count})={len(kept):,}", flush=True)

    with open(os.path.join(out_dir, "words.tsv"), "w", encoding="utf-8") as f:
        for i, w in enumerate(words):
            f.write(f"{i}\t{w}\t{uni.get(i, 0)}\n")
    with open(os.path.join(out_dir, "pairs.tsv"), "w", encoding="utf-8") as f:
        for (p, c), n in kept.items():
            f.write(f"{p}\t{c}\t{n}\n")
    meta = {
        "total_tokens": total_tokens,
        "sentences": n_sent,
        "vocab_size": len(words),
        "min_pair_count": args.min_pair_count,
        "pairs_all": len(pairs),
        "pairs_kept": len(kept),
        "train": args.train,
        "lexicon_counts": args.lexicon_counts,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
