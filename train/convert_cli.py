"""端末と同じアルゴリズムで任意の読みを変換して中身を見る CLI。

  python -m train.convert_cli --text きょうは
  python -m train.convert_cli --text きょうは --context "おはよう、"
  python -m train.convert_cli --text にほん --top 5 --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Arc, Candidate, Lexicon  # noqa: E402
from train.evaluate_bigram import BigramLM  # noqa: E402
from train.evaluate_rerank import OnnxModelScorer, beam_search  # noqa: E402
from train.train import load_lexicon  # noqa: E402

MAX_SPAN_LEN = 16
MAX_CANDIDATES = 8


def build_arcs(lex: Lexicon, read: str) -> Dict[int, List[Arc]]:
    arcs: Dict[int, List[Arc]] = {}
    for i, j, cands in lex.spans(read):
        if j - i > MAX_SPAN_LEN:
            continue
        for c in cands[:MAX_CANDIDATES]:
            arcs.setdefault(i, []).append(Arc(i, j, c))
    for i in range(len(read)):
        ch = read[i]
        arcs.setdefault(i, []).append(Arc(i, i + 1, Candidate(ch, 1, lex.unk_cost), lex.unk_cost))
    return arcs


def convert(
    lex: Lexicon,
    scorer,
    read: str,
    ctx: str,
    beta: float,
    beam: int,
    want: int,
    top: int = 5,
    verbose: bool = False,
    lm: BigramLM | None = None,
):
    arcs = build_arcs(lex, read)
    paths = beam_search(len(read), arcs, beam, want)
    if not paths:
        return []
    span_surfaces: Dict[tuple, List[str]] = {}
    for p in paths:
        for a in p:
            key = (a.start, a.end)
            lst = span_surfaces.setdefault(key, [])
            if len(lst) < MAX_CANDIDATES and a.cand.surface not in lst:
                lst.append(a.cand.surface)
    keys = list(span_surfaces.keys())
    req = [(s, e, [Candidate(x, 1, 0.0) for x in span_surfaces[(s, e)]]) for s, e in keys]
    lps = scorer.span_logprobs(ctx, read, req)
    logp = {
        (s, e): {x: float(lp[k]) for k, x in enumerate(span_surfaces[(s, e)])}
        for (s, e), lp in zip(keys, lps)
    }
    scored = []
    for p in paths:
        c = 0.0
        detail = []
        prev = None
        for a in p:
            conn = lm.conn(prev, a.cand.surface, 0.0) if lm is not None else None
            base = a.cost if conn is None else conn
            c += base
            lp = logp[(a.start, a.end)].get(a.cand.surface, -6.0)
            c += -beta * lp
            detail.append((a.cand.surface, round(base, 1), round(lp, 3)))
            prev = a.cand.surface
        scored.append((c, "".join(a.cand.surface for a in p), detail))
    scored.sort(key=lambda x: x[0])
    seen = set()
    out = []
    for c, s, d in scored:
        if s in seen:
            continue
        seen.add(s)
        out.append((c, s, d))
        if len(out) >= top:
            break
    if verbose:
        print(f"  spans={len(keys)} paths={len(paths)}")
        for c, s, d in out:
            print(f"  {c:8.2f} {s}")
            for surf, uc, lp in d:
                print(f"           {surf:10s} unigram={uc:7.1f} logP={lp:6.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--context", default="")
    ap.add_argument("--lexicon", default="data/lexicon_v2.tsv")
    ap.add_argument("--onnx", default="android/app/src/main/assets/model/scorer.onnx")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--want", type=int, default=48)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--bigram", default="")
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--score-mode", default="logprob", choices=["logprob", "logit"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon, MAX_CANDIDATES)
    scorer = None if args.no_bert else OnnxModelScorer(args.onnx, args.vocab, mode=args.score_mode)
    print(f"input : {args.text}")
    print(f"ctx   : {args.context!r}")
    lm = None
    if args.bigram:
        lm = BigramLM(args.bigram, 'data/bigram_words.tsv')
    out = convert(
        lex, scorer, args.text, args.context, args.beta,
        args.beam, args.want, args.top, args.verbose, lm,
    )
    for c, s, _ in out:
        print(f"{c:9.2f}  {s}")


if __name__ == "__main__":
    main()
