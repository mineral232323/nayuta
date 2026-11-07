"""bigram 連接コストを入れた変換の精度を測る（BERT 再ランク込み）。

  python -m train.evaluate_bigram --limit 1500 --betas 2,6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Arc, Candidate, Lexicon  # noqa: E402
from train.evaluate import ModelScorer  # noqa: E402
from train.evaluate_rerank import OnnxModelScorer, edit_distance  # noqa: E402
from train.train import load_lexicon  # noqa: E402

MAX_SPAN_LEN = 16
MAX_CANDIDATES = 8


class BigramLM:
    def __init__(self, bi_path: str, words_path: str):
        self.word_id: Dict[str, int] = {}
        self.words: List[str] = []
        with open(words_path, encoding="utf-8") as f:
            for line in f:
                i, w, _c = line.rstrip("\n").split("\t")
                self.word_id[w] = int(i)
                self.words.append(w)
        self.table: Dict[Tuple[int, int], float] = {}
        with open(bi_path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                self.table[(int(p[0]), int(p[1]))] = int(p[2]) / 100.0
        print(f"bigram: words={len(self.words)} pairs={len(self.table)}", flush=True)

    def conn(self, prev: Optional[str], cur: str, backoff: float) -> Optional[float]:
        """観測済みの連接なら -log P(cur|prev)、未観測なら None。"""
        if prev is None:
            return None
        pi = self.word_id.get(prev)
        ci = self.word_id.get(cur)
        if pi is None or ci is None:
            return None
        return self.table.get((pi, ci))

class BackoffBigramLM:
    """補間バックオフ付き bigram（`train/build_bigram_backoff.py` の出力用）。

    P(cur|prev) = (c(prev,cur) + alpha * P_uni(cur)) / (c(prev) + alpha)

    * 観測済みの連接は保存済みのカウントから補間コストを作る
    * 未観測の連接もバックオフ式で返す（None にしない）
    * 語彙外の語は None（呼び出し側が辞書コストへフォールバック）
    """

    def __init__(
        self,
        pairs_path: str,
        words_path: str,
        meta_path: str,
        alpha: float = 100.0,
    ):
        self.word_id: Dict[str, int] = {}
        self.words: List[str] = []
        self.counts: List[int] = []
        with open(words_path, encoding="utf-8") as f:
            for line in f:
                i, w, c = line.rstrip("\n").split("\t")
                self.word_id[w] = int(i)
                self.words.append(w)
                self.counts.append(int(c))
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.total_tokens = max(int(meta["total_tokens"]), 1)
        self.alpha = float(alpha)
        self.log_alpha = math.log(self.alpha)
        self.log_denom = [math.log(c + self.alpha) for c in self.counts]
        self.log_uni = [
            math.log(max(c, 1) / self.total_tokens) for c in self.counts
        ]
        self.table: Dict[Tuple[int, int], float] = {}
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                p, c, n = line.rstrip("\n").split("\t")
                pi, ci, cnt = int(p), int(c), int(n)
                prob = (cnt + self.alpha * math.exp(self.log_uni[ci])) / (
                    self.counts[pi] + self.alpha
                )
                self.table[(pi, ci)] = -math.log(prob)
        print(
            f"backoff-bigram: words={len(self.words)} pairs={len(self.table)} "
            f"alpha={self.alpha}",
            flush=True,
        )

    def conn(
        self, prev: Optional[str], cur: str, backoff: float
    ) -> Optional[float]:
        if prev is None:
            return None
        pi = self.word_id.get(prev)
        ci = self.word_id.get(cur)
        if pi is None or ci is None:
            return None
        cost = self.table.get((pi, ci))
        if cost is not None:
            return cost
        return self.log_denom[pi] - self.log_alpha - self.log_uni[ci]


class HybridBigramLM(BackoffBigramLM):
    """生成は現行と同じ（観測済み連接のみ、未観測は呼び出し側が unigram+δ）、
    再ランクだけ補間バックオフを使うハイブリッド。

    `beam_search_bigram` は第3引数に delta(>0)、`rank_paths` は 0.0 を渡すので、
    それをモード判定に使う（実験用）。
    """

    def __init__(
        self,
        pairs_path: str,
        words_path: str,
        meta_path: str,
        alpha: float = 100.0,
    ):
        super().__init__(pairs_path, words_path, meta_path, alpha)
        self.ml_table: Dict[Tuple[int, int], float] = {}
        with open(pairs_path, encoding="utf-8") as f:
            for line in f:
                p, c, n = line.rstrip("\n").split("\t")
                pi, ci, cnt = int(p), int(c), int(n)
                self.ml_table[(pi, ci)] = -math.log(cnt / max(self.counts[pi], 1))
        print(f"hybrid-bigram: ml_pairs={len(self.ml_table)}", flush=True)

    def conn(
        self, prev: Optional[str], cur: str, backoff: float
    ) -> Optional[float]:
        if prev is None:
            return None
        pi = self.word_id.get(prev)
        ci = self.word_id.get(cur)
        if pi is None or ci is None:
            return None
        if backoff > 0.5:  # 生成: 観測済みのみ（現行と同じ）
            return self.ml_table.get((pi, ci))
        return super().conn(prev, cur, backoff)


def beam_search_bigram(
    n: int,
    by_start: Dict[int, List[Arc]],
    lm: BigramLM,
    beam: int,
    want: int,
    delta: float,
) -> List[List[Arc]]:
    # 仮説は (コスト, 直前の表記, 経路)
    beams: List[List[Tuple[float, Optional[str], List[Arc]]]] = [[] for _ in range(n + 1)]
    beams[0].append((0.0, None, []))
    for i in range(n):
        cur = beams[i]
        if not cur:
            continue
        cur.sort(key=lambda x: x[0])
        for cost, prev, path in cur[:beam]:
            for arc in by_start.get(i, ()):
                c = arc.cost
                if delta < 1e8:
                    conn = lm.conn(prev, arc.cand.surface, delta)
                    if conn is not None:
                        c = conn
                    elif prev is not None:
                        c = arc.cost + delta
                beams[arc.end].append((cost + c, arc.cand.surface, path + [arc]))
    last = beams[n]
    if not last:
        return []
    last.sort(key=lambda x: x[0])
    return [p for _, _, p in last[:want]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/small_v3.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lexicon", default="lexicon_v2.tsv")
    ap.add_argument("--bigram", default="data/bigram.tsv")
    ap.add_argument("--bigram-words", default="data/bigram_words.tsv")
    ap.add_argument("--onnx", default="android/app/src/main/assets/model/scorer.onnx")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--betas", default="2.0,6.0")
    ap.add_argument("--deltas", default="1.0,2.0,3.0")
    ap.add_argument("--gen", default="bigram", choices=["unigram", "bigram"],
                    help="仮説生成に使うコスト。採点は別途")
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--want", type=int, default=48)
    args = ap.parse_args()

    lex: Lexicon = load_lexicon(os.path.join(args.data_dir, args.lexicon), MAX_CANDIDATES)
    lm = BigramLM(args.bigram, args.bigram_words)
    if args.onnx:
        scorer = OnnxModelScorer(args.onnx, args.vocab)
    else:
        scorer = ModelScorer(args.checkpoint, torch.device("cuda"))

    rows = []
    with open(os.path.join(args.data_dir, "valid.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rows.append(json.loads(line))

    deltas = [float(x) for x in args.deltas.split(",")]
    betas = [float(x) for x in args.betas.split(",")]

    def arcs_of(read: str) -> Tuple[int, Dict[int, List[Arc]]]:
        arcs: Dict[int, List[Arc]] = {}
        for i, j, cands in lex.spans(read):
            if j - i > MAX_SPAN_LEN:
                continue
            for c in cands[:MAX_CANDIDATES]:
                arcs.setdefault(i, []).append(Arc(i, j, c))
        for i in range(len(read)):
            ch = read[i]
            arcs.setdefault(i, []).append(
                Arc(i, i + 1, Candidate(ch, 1, lex.unk_cost), lex.unk_cost)
            )
        return len(read), arcs

    res: Dict[str, List[bool]] = {}
    per_len: Dict[int, List[Tuple[bool, bool]]] = {}

    for rec in rows:
        read, gold = rec["read"], "".join(w[1] for w in rec["words"])
        ctx = rec["ctx"]
        n, arcs = arcs_of(read)

        paths_u = beam_search_bigram(n, arcs, lm, args.beam, args.want, 1e9)
        uni_best = "".join(a.cand.surface for a in paths_u[0]) if paths_u else ""
        res.setdefault("unigram", []).append(uni_best == gold)
        res.setdefault("unigram_oracle", []).append(
            any("".join(a.cand.surface for a in p) == gold for p in paths_u)
        )

        best_for_len: Dict[str, bool] = {"unigram": uni_best == gold}
        for d in deltas:
            paths = (
                paths_u
                if args.gen == "unigram"
                else beam_search_bigram(n, arcs, lm, args.beam, args.want, d)
            )
            res.setdefault(f"{args.gen}_oracle(d={d})", []).append(
                any("".join(a.cand.surface for a in p) == gold for p in paths)
            )

            def surf(p):
                return "".join(a.cand.surface for a in p)

            bg_best = surf(paths[0])
            key = f"bigram(d={d})"
            res.setdefault(key, []).append(bg_best == gold)
            best_for_len[key] = bg_best == gold

            span_surfaces: Dict[Tuple[int, int], List[str]] = {}
            for p in paths:
                for a in p:
                    lst = span_surfaces.setdefault((a.start, a.end), [])
                    if len(lst) < MAX_CANDIDATES and a.cand.surface not in lst:
                        lst.append(a.cand.surface)
            keys = list(span_surfaces.keys())
            req = [(s, e, [Candidate(x, 1, 0.0) for x in span_surfaces[(s, e)]]) for s, e in keys]
            lps = scorer.span_logprobs(ctx, read, req)
            logp = {
                (s, e): {x: float(lp[k]) for k, x in enumerate(span_surfaces[(s, e)])}
                for (s, e), lp in zip(keys, lps)
            }
            for b in betas:
                scored = []
                for p in paths:
                    c, prev = 0.0, None
                    for a in p:
                        conn = lm.conn(prev, a.cand.surface, 0)
                        c += a.cost if conn is None else conn
                        c += -b * logp[(a.start, a.end)].get(a.cand.surface, -6.0)
                        prev = a.cand.surface
                    scored.append((c, p))
                scored.sort(key=lambda x: x[0])
                kk = f"{args.gen}+bigram+BERT(d={d},b={b})"
                ok = surf(scored[0][1]) == gold
                res.setdefault(kk, []).append(ok)
                best_for_len[kk] = ok

        per_len.setdefault(n, []).append((best_for_len["unigram"], best_for_len[f"bigram(d={deltas[0]})"]))

    print(f"\nsentences={len(rows)}")
    for k, v in res.items():
        print(f"  {k:<26} {sum(v)/len(v):.4f}")


if __name__ == "__main__":
    main()
