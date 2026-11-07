"""文頭語正解率を checkpoint（torch）で測る小さな評価ツール。

  python -m train.eval_head_acc --checkpoint models/scorer_rerank_head20k.pt \
      --data data/eval/head_prior_wrong.jsonl --beta 0.5 --delta 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.evaluate_bigram import BigramLM, beam_search_bigram  # noqa: E402
from train.evaluate_v2 import (  # noqa: E402
    TorchScorer,
    build_lattice,
    path_surface,
    rank_paths,
    score_spans,
)
from train.train import load_lexicon  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="checkpoint で文頭語正解率を測る")
    ap.add_argument("--data", default="data/eval/head_prior_wrong.jsonl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--bigram", default="data/bigram_v3.tsv")
    ap.add_argument("--bigram-words", default="data/bigram_words_v3.tsv")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--want", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--errors", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    lex = load_lexicon(args.lexicon, 8)
    lm = BigramLM(args.bigram, args.bigram_words)
    scorer = TorchScorer(args.checkpoint, device)
    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break

    n = head_ok = sent_ok = 0
    shown = 0
    for rec in rows:
        words = rec.get("words") or []
        if not words:
            continue
        read = rec.get("read") or "".join(w[0] for w in words)
        gold = "".join(w[1] for w in words)
        head_gold = words[0][1]
        n_, arcs, _ = build_lattice(lex, read)
        paths = beam_search_bigram(n_, arcs, lm, args.beam, args.want, args.delta)
        if not paths:
            continue
        logp = score_spans(scorer, rec.get("ctx", ""), read, paths)
        pred = path_surface(rank_paths(paths, lm, logp, args.beta)[0])
        n += 1
        ok = pred.startswith(head_gold)
        head_ok += ok
        sent_ok += pred == gold
        if not ok and shown < args.errors:
            shown += 1
            print(f"NG head: gold={head_gold} pred={pred[:40]}")
    print(
        f"{os.path.basename(args.checkpoint)}: n={n} "
        f"head_acc={head_ok / max(n, 1):.4f} ({head_ok}/{n}) "
        f"sent_acc={sent_ok / max(n, 1):.4f}"
    )


if __name__ == "__main__":
    main()
