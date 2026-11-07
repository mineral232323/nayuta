"""bigram の語彙拡張＋補間バックオフの効果を測る実験。

    python -m train.experiment_bigram_backoff \
        --dirs logs/bigram_bo100k logs/bigram_bo200k --alpha 100

比較:
  * baseline: 既存 `data/bigram_v3.tsv`（10万語・未観測は unigram へハードフォールバック）
  * backoff : `train/build_bigram_backoff.py` の出力（語彙拡張＋Jelinek-Mercer 補間）

データ: wiki2000（β=0.5）/ sns2000（β=2.0）/ suru_amb 300（β=0.5）。
最後に `きしゃする` の候補と順位も出す。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.evaluate_bigram import (  # noqa: E402
    BackoffBigramLM,
    BigramLM,
    HybridBigramLM,
    beam_search_bigram,
)
from train.evaluate_v2 import (  # noqa: E402
    OnnxScorer,
    build_lattice,
    gold_info,
    path_surface,
    rank_paths,
    run_eval,
    score_spans,
)
from train.train import load_lexicon  # noqa: E402

DATASETS = [
    ("wiki", "data/eval/wiki2000.jsonl", 2000, 0.5),
    ("sns", "data/eval/sns2000.jsonl", 2000, 2.0),
    ("suru", "data/eval/suru_amb.jsonl", 300, 0.5),
]


def load_rows(path: str, limit: int) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def top_paths(
    lex, lm, scorer, read: str, ctx: str, top: int = 5
) -> List[Tuple[float, str]]:
    n, arcs, _spans = build_lattice(lex, read)
    paths = beam_search_bigram(n, arcs, lm, 32, 48, 1.0)
    if not paths:
        return []
    logp = score_spans(scorer, ctx, read, paths)
    ranked = rank_paths(paths, lm, logp, 0.5)
    out: List[Tuple[float, str]] = []
    seen = set()
    for p in ranked:
        s = path_surface(p)
        if s in seen:
            continue
        seen.add(s)
        cost = 0.0
        prev: Optional[str] = None
        for a in p:
            conn = lm.conn(prev, a.cand.surface, 0.0)
            cost += a.cost if conn is None else conn
            lp = logp.get((a.start, a.end), {}).get(a.cand.surface, -6.0)
            cost += -0.5 * lp
            prev = a.cand.surface
        out.append((cost, s))
        if len(out) >= top:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="bigram 語彙拡張＋バックオフの評価")
    ap.add_argument("--dirs", nargs="*", default=["logs/bigram_bo100k"],
                    help="build_bigram_backoff.py の出力ディレクトリ")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--hybrid", action="store_true",
                    help="生成は現行（観測済みのみ）、再ランクだけバックオフ")
    ap.add_argument("--datasets", default="wiki,sns,suru")
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--onnx", default="android/app/src/main/assets/model/scorer.onnx")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    sets = [(n, p, lim, beta) for (n, p, lim, beta) in DATASETS if n in names]

    lex = load_lexicon(args.lexicon, 8)
    sc = OnnxScorer(args.onnx, args.vocab, threads=args.threads)

    models: List[Tuple[str, object]] = [
        ("baseline", BigramLM("data/bigram_v3.tsv", "data/bigram_words_v3.tsv")),
    ]
    for d in args.dirs:
        pairs = os.path.join(d, "pairs.tsv")
        words = os.path.join(d, "words.tsv")
        meta = os.path.join(d, "meta.json")
        if not all(os.path.exists(p) for p in (pairs, words, meta)):
            raise SystemExit(f"ビルド結果がありません: {d}")
        cls = HybridBigramLM if args.hybrid else BackoffBigramLM
        models.append((os.path.basename(d), cls(pairs, words, meta, args.alpha)))

    results: List[dict] = []
    print(f"\n{'model':<20} {'data':<6} {'acc':>7} {'oracle':>7} {'ceiling':>7} "
          f"{'p50ms':>7}")
    for mname, lm in models:
        for dname, path, limit, beta in sets:
            rows = load_rows(path, limit)
            t0 = time.perf_counter()
            res = run_eval(
                rows, lex, lm, sc, gen="bigram", deltas=[1.0], betas=[beta],
                beam=32, want=48,
            )
            key = f"bigram_bert(d=1.0,b={beta})"
            m = res["metrics"][key]
            o = res["oracle"]["bigram(d=1.0)"]["rate"]
            c = res["counts"]["lattice_ceiling_rate"]
            p50 = res["latency"]["total_ms"]["p50"]
            print(f"{mname:<20} {dname:<6} {m['sent_acc']:>7.4f} {o:>7.4f} "
                  f"{c:>7.4f} {p50:>7.1f}", flush=True)
            results.append({
                "model": mname, "data": dname, "limit": limit,
                "acc": m["sent_acc"], "cer": m["cer_micro"],
                "oracle": o, "ceiling": c, "p50_ms": p50,
                "seconds": time.perf_counter() - t0,
            })

    print("\n[case] きしゃする")
    for mname, lm in models:
        tp = top_paths(lex, lm, sc, "きしゃする", "")
        print(f"  {mname}: " + " / ".join(f"{s}({c:.1f})" for c, s in tp))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
