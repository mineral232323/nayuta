"""端末と同じ 2 段階（unigram ビーム探索 → BERT 再ランク）の精度を測る。

android の LatticeConverter と同じ手順を Python で再現するので、
ここで出た数字がそのまま実機の見込みになる。

  python -m train.evaluate_rerank --checkpoint models/small_v3.pt --betas 0.5,1.0,2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Arc, Candidate, Lexicon  # noqa: E402
from train.evaluate import ModelScorer  # noqa: E402
from train.train import load_lexicon  # noqa: E402


class OnnxModelScorer:
    """配布物（int8 ONNX）をそのまま評価するためのラッパ。"""

    def __init__(self, onnx_path: str, vocab_path: str, k: int = 8, mode: str = "logprob"):
        import onnxruntime as ort

        from train.scorer import CharVocab

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.vocab = CharVocab(vocab_path)
        self.k = k
        self.mode = mode
        # オンライン学習用の w 入力の次元（無いモデルなら 384）
        wshape = [i.shape for i in self.sess.get_inputs() if i.name == "w"]
        self.hidden = int(wshape[0][1]) if wshape else 384

    def span_logprobs(self, ctx, read, spans, chunk=256):
        import numpy as np

        S = 12
        cids = self.vocab.encode(ctx[-64:])
        rids = self.vocab.encode(read)
        seq = [self.vocab.cls] + cids + [self.vocab.sep] + rids + [self.vocab.sep]
        off = 1 + len(cids) + 1
        input_ids = np.array([seq], dtype=np.int64)
        am = np.ones_like(input_ids)
        out = []
        K = self.k  # ONNX の cand 軸は固定長なので必ず K で詰める
        pad = self.vocab.pad
        for s0 in range(0, len(spans), chunk):
            part = spans[s0 : s0 + chunk]
            W = len(part)
            starts = np.zeros((1, W), dtype=np.int64)
            ends = np.zeros((1, W), dtype=np.int64)
            cidx = np.full((1, W, K, S), pad, dtype=np.int64)
            cmask = np.zeros((1, W, K, S), dtype=np.int64)
            for w, (i, j, cands) in enumerate(part):
                starts[0, w] = off + i
                ends[0, w] = off + j
                for c_i, c in enumerate(cands):
                    ids = self.vocab.encode(c.surface)[:S]
                    cidx[0, w, c_i, : len(ids)] = ids
                    cmask[0, w, c_i, : len(ids)] = 1
            logits = self.sess.run(
                None,
                {
                    "input_ids": input_ids,
                    "attention_mask": am,
                    "starts": starts,
                    "ends": ends,
                    "cand_ids": cidx,
                    "cand_mask": cmask,
                    # オンライン学習用の入力。評価では未学習（0）で固定する。
                    "w": np.zeros((1, self.hidden), dtype=np.float32),
                },
            )[0][0]  # (W,K)
            if self.mode == "logit":
                # 生の logit。候補が 1 件のスパンでも 0 にならないので、
                # 分割の違いを跨いで比べられる可能性がある。
                lp = logits
            else:
                logits = np.where(cmask[0, :, :, 0] == 0, -1e4, logits)
                m = logits.max(-1, keepdims=True)
                lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
                lp = logits - lse
            for w, (_, _, cands) in enumerate(part):
                out.append(torch.tensor(lp[w, : len(cands)]))
        return out

MAX_SPAN_LEN = 16
MAX_CANDIDATES = 8


def beam_search(
    n: int,
    by_start: Dict[int, List[Arc]],
    beam: int,
    want: int,
) -> List[List[Arc]]:
    beams: List[List[Tuple[float, List[Arc]]]] = [[] for _ in range(n + 1)]
    beams[0].append((0.0, []))
    for i in range(n):
        cur = beams[i]
        if not cur:
            continue
        cur.sort(key=lambda x: x[0])
        for cost, path in cur[:beam]:
            for arc in by_start.get(i, ()):
                beams[arc.end].append((cost + arc.cost, path + [arc]))
    last = beams[n]
    if not last:
        return []
    last.sort(key=lambda x: x[0])
    return [p for _, p in last[:want]]


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/small_v3.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--lexicon", default="lexicon_v2.tsv")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--betas", default="0.5,1.0,2.0")
    ap.add_argument("--onnx", default="")
    ap.add_argument("--score-mode", default="logprob", choices=["logprob", "logit"])
    ap.add_argument("--mode", default="free", choices=["free", "fixed"],
                    help="free: 分割ごと再ランク / fixed: 分割は unigram で固定し表記だけ BERT")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--want", type=int, default=24)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.onnx:
        scorer = OnnxModelScorer(args.onnx, args.vocab, mode=args.score_mode)
        print(f"scorer: onnx {args.onnx}")
    else:
        scorer = ModelScorer(args.checkpoint, device)
    lex: Lexicon = load_lexicon(os.path.join(args.data_dir, args.lexicon), MAX_CANDIDATES)
    betas = [float(b) for b in args.betas.split(",")]

    rows = []
    with open(os.path.join(args.data_dir, "valid.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rows.append(json.loads(line))

    base_hit = 0
    hits = {b: 0 for b in betas}
    fixed_hits = {b: 0 for b in betas}
    per_sent = []  # (入力長, unigram正解, {beta: 正解}, unigram CER, {beta: CER})
    # BERT 無しでどこまで候補が入っているか（オラクル）
    oracle = 0
    n_sent = 0

    for rec in rows:
        read = rec["read"]
        ctx = rec["ctx"]
        n = len(read)
        spans = lex.spans(read)
        arcs_by_start: Dict[int, List[Arc]] = {}
        for i, j, cands in spans:
            if j - i > MAX_SPAN_LEN:
                continue
            for c in cands[:MAX_CANDIDATES]:
                arcs_by_start.setdefault(i, []).append(Arc(i, j, c))
        for i in range(n):
            ch = read[i]
            arcs_by_start.setdefault(i, []).append(
                Arc(i, i + 1, Candidate(ch, 1, lex.unk_cost), lex.unk_cost)
            )

        paths = beam_search(n, arcs_by_start, args.beam, args.want)
        if not paths:
            continue
        n_sent += 1
        gold = "".join(w[1] for w in rec["words"])

        def surf(p: Sequence[Arc]) -> str:
            return "".join(a.cand.surface for a in p)

        base_ok = surf(paths[0]) == gold
        if base_ok:
            base_hit += 1
        cb = edit_distance(surf(paths[0]), gold) / max(len(gold), 1)
        if any(surf(p) == gold for p in paths):
            oracle += 1

        # 仮説に出てくるスパンだけ採点
        span_surfaces: Dict[Tuple[int, int], List[str]] = {}
        for p in paths:
            for a in p:
                key = (a.start, a.end)
                lst = span_surfaces.setdefault(key, [])
                if len(lst) < MAX_CANDIDATES and a.cand.surface not in lst:
                    lst.append(a.cand.surface)
        keys = list(span_surfaces.keys())
        req = [
            (s, e, [Candidate(x, 1, 0.0) for x in span_surfaces[(s, e)]])
            for s, e in keys
        ]
        lps = scorer.span_logprobs(ctx, read, req)
        logp: Dict[Tuple[int, int], Dict[str, float]] = {}
        for (s, e), lp in zip(keys, lps):
            logp[(s, e)] = {
                x: float(lp[k]) for k, x in enumerate(span_surfaces[(s, e)])
            }

        row = {}
        cerr = {}
        for b in betas:
            scored = []
            for p in paths:
                c = 0.0
                for a in p:
                    c += a.cost
                    c += -b * logp[(a.start, a.end)].get(a.cand.surface, -6.0)
                scored.append((c, p))
            scored.sort(key=lambda x: x[0])
            row[b] = surf(scored[0][1]) == gold
            if row[b]:
                hits[b] += 1
            cerr[b] = edit_distance(surf(scored[0][1]), gold) / max(len(gold), 1)
        # 分割を unigram 最尤で固定し、表記だけ BERT に選ばせる
        seg = [(a.start, a.end) for a in paths[0]]
        span_map = {(i, j): c for i, j, c in spans}
        if all(k in span_map for k in seg):
            req2 = [(st, en, span_map[(st, en)][:MAX_CANDIDATES]) for st, en in seg]
            lps2 = scorer.span_logprobs(ctx, read, req2)
            for b in betas:
                parts = []
                for (st, en, cands), lp in zip(req2, lps2):
                    best_i, best_c = 0, None
                    for ci, c in enumerate(cands):
                        cc = c.cost - b * float(lp[ci])
                        if best_c is None or cc < best_c:
                            best_c, best_i = cc, ci
                    parts.append(cands[best_i].surface)
                if "".join(parts) == gold:
                    fixed_hits[b] += 1
        per_sent.append((n, base_ok, row, cb, cerr))

    print(f"sentences={n_sent}")
    print(f"unigram beam best        : {base_hit / n_sent:.4f}")
    print(f"oracle (正解が仮説内)     : {oracle / n_sent:.4f}")
    for b in betas:
        print(f"bert rerank beta={b:<4}  : {hits[b] / n_sent:.4f}   (分割固定なら {fixed_hits[b] / n_sent:.4f})")

    print("\n入力長ごとの文正解率")
    print(f"{'文字数':>10} {'件数':>6} " + " ".join(f"{'uni':>7}" for _ in range(1)) + " " + " ".join(f"b={b:<5}" for b in betas) + "  改善")
    edges = [(0, 5), (6, 9), (10, 13), (14, 18), (19, 24), (25, 99)]
    for lo, hi in edges:
        sel = [r for r in per_sent if lo <= r[0] <= hi]
        if not sel:
            continue
        bu = sum(1 for r in sel if r[1]) / len(sel)
        cells = []
        for b in betas:
            cells.append(sum(1 for r in sel if r[2][b]) / len(sel))
        cerr_u = sum(r[3] for r in sel) / len(sel)
        cerr_b = sum(r[4][betas[0]] for r in sel) / len(sel)
        print(f"{lo:>4}-{hi:<4} {len(sel):>6} {bu:>7.3f} " + " ".join(f"{c:>7.3f}" for c in cells) + f"  {cells[0]-bu:+.3f}   CER {cerr_u:.3f} -> {cerr_b:.3f}")


if __name__ == "__main__":
    main()
