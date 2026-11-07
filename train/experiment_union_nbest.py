"""unigram / bigram / 両方（union）の N-best と再ランク精度を比べる実験。

    python -m train.experiment_union_nbest --data data/eval/wiki2000.jsonl --limit 2000
    python -m train.experiment_union_nbest --data data/eval/suru_amb.jsonl --limit 300

各文で
  * unigram ビーム（端末の現行生成）         … oracle
  * bigram ビーム（評価ハーネスの生成）      … oracle
  * union（両方を表記で重複除去してマージ）  … oracle
を作り、それぞれを現行 BERT（スパン採点 + β=0.5）で再ランクした文正解率を出す。
union は「同じ探索コストを 2 回払う」だけで到達率が上がるかの確認。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.evaluate_bigram import BigramLM, beam_search_bigram  # noqa: E402
from train.evaluate_rerank import beam_search as beam_search_unigram  # noqa: E402
from train.evaluate_v2 import (  # noqa: E402
    OnnxScorer,
    build_lattice,
    gold_info,
    path_surface,
    rank_paths,
    score_spans,
)
from train.train import load_lexicon  # noqa: E402


def dedupe(paths: Sequence[Sequence]) -> List[List]:
    seen = set()
    out = []
    for p in paths:
        s = path_surface(p)
        if s in seen:
            continue
        seen.add(s)
        out.append(list(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="union N-best の到達率と再ランク精度")
    ap.add_argument("--data", default="data/eval/wiki2000.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--bigram", default="data/bigram_v3.tsv")
    ap.add_argument("--bigram-words", default="data/bigram_words_v3.tsv")
    ap.add_argument("--onnx", default="android/app/src/main/assets/model/scorer.onnx")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--want", type=int, default=48)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon, 8)
    lm = BigramLM(args.bigram, args.bigram_words)
    sc = OnnxScorer(args.onnx, args.vocab, threads=args.threads)
    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    n = 0
    oracle: Dict[str, int] = {"unigram": 0, "bigram": 0, "union": 0}
    acc: Dict[str, int] = {"bigram": 0, "union": 0}
    t0 = time.time()
    for rec in rows:
        read, gold, _b = gold_info(rec)
        if not read:
            continue
        ctx = rec.get("ctx", "")
        nn, arcs, _spans = build_lattice(lex, read)
        pu = dedupe(beam_search_unigram(nn, arcs, args.beam, args.want))
        pb = dedupe(beam_search_bigram(nn, arcs, lm, args.beam, args.want, 1.0))
        union = dedupe(list(pu) + list(pb))
        n += 1
        for name, paths in (("unigram", pu), ("bigram", pb), ("union", union)):
            if any(path_surface(p) == gold for p in paths):
                oracle[name] += 1
        for name, paths in (("bigram", pb), ("union", union)):
            logp = score_spans(sc, ctx, read, paths)
            ranked = rank_paths(paths, lm, logp, args.beta)
            if ranked and path_surface(ranked[0]) == gold:
                acc[name] += 1
        if n % 200 == 0:
            print(f"  {n}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    n = max(n, 1)
    print()
    print(f"data: {args.data}  n={n}  beam={args.beam} want={args.want} beta={args.beta}")
    print("oracle (N-best に正解が入る率)")
    for k, v in oracle.items():
        print(f"  {k:<8} {v / n:.4f}")
    print("sent_acc (現行 BERT 再ランク)")
    for k, v in acc.items():
        print(f"  {k:<8} {v / n:.4f}")


if __name__ == "__main__":
    main()
