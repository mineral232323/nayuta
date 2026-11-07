"""文頭 hard negative 入りの N-best を作る（listwise/pairwise 再ランク学習用）。

学習用コーパスから、次の条件を満たす文の N-best を書き出す:

  * 正解の経路が N-best に入っている
  * `--head-only` のとき、文頭の語が「頻度 prior と異なる正解」である
  * さらに、文頭に prior 側の表記を持つ経路も N-best に含まれている（hard negative）

  python -m train.build_rerank_nbest --limit 100000 --head-only \
      --out logs/rerank_nbest_head.jsonl
  # 学習用に全体から混ぜる場合
  python -m train.build_rerank_nbest --limit 300000 --general-prob 0.3 \
      --out logs/rerank_nbest_mix.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.evaluate_bigram import BigramLM, beam_search_bigram  # noqa: E402
from train.evaluate_v2 import build_lattice, path_surface  # noqa: E402
from train.train import load_lexicon  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="再ランク学習用 N-best の生成")
    ap.add_argument("--train", default="data/train_v3.jsonl")
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--bigram", default="data/bigram_v3.tsv")
    ap.add_argument("--bigram-words", default="data/bigram_words_v3.tsv")
    ap.add_argument("--out", default="logs/rerank_nbest.jsonl")
    ap.add_argument("--limit", type=int, default=20000, help="書き出す文数")
    ap.add_argument("--max-scan", type=int, default=0, help="走査する文数の上限（0=無制限）")
    ap.add_argument("--skip-lines", type=int, default=0, help="先頭から読み飛ばす行数（並列生成用）")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--want", type=int, default=16)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--head-only", action="store_true",
                    help="文頭 prior-wrong の文だけを対象にする")
    ap.add_argument("--general-prob", type=float, default=0.0,
                    help="prior-wrong でない文もこの確率で混ぜる")
    ap.add_argument("--require-hard-head", action="store_true", default=True,
                    help="文頭 prior 側の経路が N-best に無い文は捨てる")
    ap.add_argument("--no-require-hard-head", dest="require_hard_head", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    lex = load_lexicon(args.lexicon, 8)
    lm = BigramLM(args.bigram, args.bigram_words)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    written = scanned = skipped_gold = skipped_hard = 0
    head_hard = 0
    with open(args.train, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for line_no, line in enumerate(f):
            if line_no < args.skip_lines:
                continue
            if written >= args.limit:
                break
            scanned += 1
            if args.max_scan and scanned > args.max_scan:
                break
            rec = json.loads(line)
            words = rec.get("words") or []
            if len(words) < 2:
                continue
            read = rec.get("read") or "".join(w[0] for w in words)
            gold = "".join(w[1] for w in words)
            ctx = rec.get("ctx", "")

            rd0, gold0 = words[0]
            cands = lex.entries.get(rd0, [])
            surfaces = [c.surface for c in cands]
            prior = surfaces[0] if len(surfaces) >= 2 else None
            prior_wrong = prior is not None and prior != gold0 and gold0 in surfaces
            if args.head_only and not prior_wrong:
                continue
            if not prior_wrong and random.random() >= args.general_prob:
                continue

            n, arcs, _spans = build_lattice(lex, read)
            paths = beam_search_bigram(n, arcs, lm, args.beam, args.want, args.delta)
            if not paths:
                skipped_gold += 1
                continue
            gold_idx = None
            for i, p in enumerate(paths):
                if path_surface(p) == gold:
                    gold_idx = i
                    break
            if gold_idx is None:
                skipped_gold += 1
                continue

            # hard negative: 文頭が prior 側の表記になっている経路
            hard_ok = True
            if args.require_hard_head and prior_wrong:
                hard_ok = any(p[0].cand.surface == prior for p in paths)
            if not hard_ok:
                skipped_hard += 1
                continue

            arcs_out = []
            for p in paths[: args.want]:
                arcs_out.append([[a.start, a.end, a.cand.surface] for a in p])
            out.write(
                json.dumps(
                    {
                        "ctx": ctx,
                        "read": read,
                        "paths": arcs_out,
                        "gold": gold_idx,
                        "head": {
                            "reading": rd0,
                            "gold": gold0,
                            "prior": prior,
                            "prior_wrong": prior_wrong,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            head_hard += int(prior_wrong)
            if written % 2000 == 0:
                print(f"  {written} written / {scanned} scanned", flush=True)

    print(
        f"wrote {args.out}: {written} 文（scanned={scanned}, head_prior_wrong={head_hard}, "
        f"skip gold={skipped_gold}, skip hard={skipped_hard}）"
    )


if __name__ == "__main__":
    main()
