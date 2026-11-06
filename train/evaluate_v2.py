"""評価ハーネス v2: 指標の拡張と天井の分解測定（Phase 0 / Phase 6 の基盤）。

`train/evaluate_bigram.py` と同一の生成（辞書ラティス + bigram ビーム探索）と
再ランク（char-level BERT, int8 ONNX）の手順を保ったまま、以下を 1 回の実行で
機械可読に出力する。

* 文正解率（完全一致）
* CER（micro = 総編集距離/総正解長、文単位平均）
* 天井率（N-best oracle。unigram 生成と bigram 生成の両方）
* ラティス天井（正解の全スパンが辞書に揃う率 = 正解経路がラティスに存在する率）
* gold span coverage（正解スパンの読み/表記が辞書にある率。Phase 1 の効果確認用）
* 境界 F 値（予測した単語境界位置 vs 正解境界位置。読み座標で比較）
* 正解率@k（再ランク後の上位 k 仮説、表記文字列の重複除去）
* レイテンシ p50/p95/p99（CPU int8 ONNX を端末 proxy とする）
* beam/want スイープ（`--sweep-only` で天井だけを高速測定）

例::

    # 現行アセットのベースライン（端末相当の int8 ONNX / CPU）
    PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
        --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
        --deltas 2.0 --betas 1.0,2.0,4.0,6.0 \
        --json-out logs/eval_v2_wiki.json

    # Phase 0: 天井だけを beam/want スイープで測る
    PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
        --data data/eval/wiki2000.jsonl --domain wiki --sweep-only \
        --beam-sweep "32,48;64,128;128,256" \
        --json-out logs/phase0_ceiling.json

注意:
    * 生成・再ランクの数式は `evaluate_bigram.py` と同一。`--gen bigram` のときの
      「unigram」系の指標は bigram を切ったビーム探索の結果。
    * レイテンシの total は
      「ラティス構築 + 主 delta のビーム探索 + その BERT 推論 + 主 beta の再ランク」。
      (Phase 0 用に unigram ビームを余分に回す時間は含めない)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Arc, Candidate, Lexicon  # noqa: E402
from train.evaluate_bigram import BigramLM, beam_search_bigram  # noqa: E402
from train.evaluate_rerank import beam_search as beam_search_unigram  # noqa: E402
from train.evaluate_rerank import edit_distance  # noqa: E402
from train.train import load_lexicon  # noqa: E402

MAX_SPAN_LEN = 16
MAX_CANDIDATES = 8
TOOL_VERSION = 1

# 入力長ごとの内訳（evaluate_rerank.py と同じ区切り）
LENGTH_EDGES: List[Tuple[int, int]] = [
    (0, 5),
    (6, 9),
    (10, 13),
    (14, 18),
    (19, 24),
    (25, 999),
]


# ---------------------------------------------------------------------------
# 採点器ラッパ（int8 ONNX / checkpoint 両対応）
# ---------------------------------------------------------------------------


class OnnxScorer:
    """配布物の int8 ONNX を CPU で回す（端末 proxy）。

    `train.evaluate_rerank.OnnxModelScorer` と同じ推論グラフだが、
    * `w` 入力（オンライン学習用）が無いモデルでも動く
    * CPU スレッド数を固定できる（端末に近いレイテンシを測るため）
    という 2 点だけ拡張している。
    """

    def __init__(
        self,
        onnx_path: str,
        vocab_path: str,
        k: int = MAX_CANDIDATES,
        threads: int = 4,
        mode: str = "logprob",
    ):
        import onnxruntime as ort

        from train.scorer import CharVocab

        so = ort.SessionOptions()
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.vocab = CharVocab(vocab_path)
        self.k = k
        self.mode = mode
        names = [i.name for i in self.sess.get_inputs()]
        self.has_w = "w" in names
        wshape = [i.shape for i in self.sess.get_inputs() if i.name == "w"]
        self.hidden = int(wshape[0][1]) if wshape else 384
        self.meta: Dict[str, Any] = {
            "kind": "onnx",
            "path": onnx_path,
            "vocab": vocab_path,
            "inputs": names,
            "k": k,
            "threads": int(threads),
            "has_w_input": self.has_w,
            "providers": self.sess.get_providers(),
            "onnxruntime": ort.__version__,
        }

    def span_logprobs(
        self,
        ctx: str,
        read: str,
        spans: Sequence[Tuple[int, int, Sequence[Candidate]]],
        chunk: int = 256,
    ) -> List[torch.Tensor]:
        import numpy as np

        S = 12
        cids = self.vocab.encode(ctx[-64:])
        rids = self.vocab.encode(read)
        seq = [self.vocab.cls] + cids + [self.vocab.sep] + rids + [self.vocab.sep]
        off = 1 + len(cids) + 1
        input_ids = np.array([seq], dtype=np.int64)
        am = np.ones_like(input_ids)
        out: List[torch.Tensor] = []
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
            feed: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": am,
                "starts": starts,
                "ends": ends,
                "cand_ids": cidx,
                "cand_mask": cmask,
            }
            if self.has_w:
                # オンライン学習用の入力。評価では未学習（0）で固定する。
                feed["w"] = np.zeros((1, self.hidden), dtype=np.float32)
            logits = self.sess.run(None, feed)[0][0]  # (W,K)
            if self.mode == "logit":
                lp = logits
            else:
                logits = np.where(cmask[0, :, :, 0] == 0, -1e4, logits)
                m = logits.max(-1, keepdims=True)
                lse = m + np.log(np.exp(logits - m).sum(-1, keepdims=True))
                lp = logits - lse
            for w, (_, _, cands) in enumerate(part):
                out.append(torch.tensor(lp[w, : len(cands)]))
        return out


class TorchScorer:
    """checkpoint を GPU/CPU で回すラッパ（端末 proxy ではない）。"""

    def __init__(self, checkpoint: str, device: torch.device):
        from train.evaluate import ModelScorer

        self.inner = ModelScorer(checkpoint, device)
        self.meta: Dict[str, Any] = {
            "kind": "checkpoint",
            "path": checkpoint,
            "device": str(device),
            "k": self.inner.k,
        }

    def span_logprobs(
        self,
        ctx: str,
        read: str,
        spans: Sequence[Tuple[int, int, Sequence[Candidate]]],
    ) -> List[torch.Tensor]:
        return self.inner.span_logprobs(ctx, read, spans)


# ---------------------------------------------------------------------------
# データ / 辞書 / ラティス
# ---------------------------------------------------------------------------


def load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def gold_info(rec: Dict[str, Any]) -> Tuple[str, str, set]:
    """(読み, 正解文, 正解境界位置の集合) を返す。

    境界は読み座標（0 < pos < len(read)）で表す。表記の文字数と読みの文字数が
    違う語があるため、予測（弧の境界）と同じ座標系にするには読み座標が必要。
    """
    read = rec["read"]
    gold = "".join(w[1] for w in rec["words"])
    bounds: set = set()
    pos = 0
    for rd, _surf in rec["words"]:
        pos += len(rd)
        if 0 < pos < len(read):
            bounds.add(pos)
    return read, gold, bounds


def build_lattice(
    lex: Lexicon, read: str
) -> Tuple[int, Dict[int, List[Arc]], Dict[Tuple[int, int], List[Candidate]]]:
    """開始位置 -> 弧リストと、辞書スパンの写像を作る。

    `evaluate_bigram.py` の `arcs_of` と同じ構成（未登録の 1 文字も高コストで必ず入れる）。
    """
    n = len(read)
    arcs: Dict[int, List[Arc]] = {}
    spans: Dict[Tuple[int, int], List[Candidate]] = {}
    for i, j, cands in lex.spans(read):
        if j - i > MAX_SPAN_LEN:
            continue
        spans[(i, j)] = list(cands)
        lst = arcs.setdefault(i, [])
        for c in cands[:MAX_CANDIDATES]:
            lst.append(Arc(i, j, c))
    for i in range(n):
        ch = read[i]
        arcs.setdefault(i, []).append(
            Arc(i, i + 1, Candidate(ch, 1, lex.unk_cost), lex.unk_cost)
        )
    return n, arcs, spans


def path_surface(path: Sequence[Arc]) -> str:
    return "".join(a.cand.surface for a in path)


def path_boundaries(path: Sequence[Arc], n: int) -> set:
    return {a.end for a in path if a.end < n}


def gold_path_in_lattice(
    spans: Dict[Tuple[int, int], List[Candidate]], words: Sequence[Sequence[str]]
) -> bool:
    """正解の全スパン（読み・表記）がラティスに存在するか = ラティス天井。"""
    pos = 0
    for rd, surf in words:
        j = pos + len(rd)
        cands = spans.get((pos, j))
        if not rd or cands is None:
            return False
        if not any(c.surface == surf for c in cands[:MAX_CANDIDATES]):
            return False
        pos = j
    return True


def analyze_gold_coverage(
    lex: Lexicon, rows: Sequence[Dict[str, Any]], max_examples: int = 40
) -> Dict[str, Any]:
    """gold span coverage（辞書カバレッジ律速かどうかの分解）。

    * `span_in_dict`: 正解スパンの読みが辞書スパンとして存在する率
    * `surface_in_candidates`: 上記に加えて正解表記が候補（上位 K）に入る率
    * `sentence_path_rate`: 全スパンが揃う文の率（= ラティス天井）
    """
    n_spans = n_span_ok = n_surface_ok = 0
    n_too_long = 0
    n_sent = n_sent_full = 0
    missing_by_reason: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[Tuple[int, int], List[Candidate]]] = {}
    for rec in rows:
        read = rec["read"]
        if read not in cache:
            sm: Dict[Tuple[int, int], List[Candidate]] = {}
            for i, j, cands in lex.spans(read):
                sm[(i, j)] = list(cands)
            cache[read] = sm
        sm = cache[read]
        n_sent += 1
        full = True
        pos = 0
        for rd, surf in rec["words"]:
            j = pos + len(rd)
            n_spans += 1
            if j - pos > MAX_SPAN_LEN:
                n_too_long += 1
                missing_by_reason["too_long"] = missing_by_reason.get("too_long", 0) + 1
                full = False
                pos = j
                continue
            cands = sm.get((pos, j))
            if cands is None:
                full = False
                missing_by_reason["no_span_in_dict"] = (
                    missing_by_reason.get("no_span_in_dict", 0) + 1
                )
                if len(examples) < max_examples:
                    examples.append(
                        {"read": rd, "surface": surf, "reason": "no_span_in_dict"}
                    )
                pos = j
                continue
            n_span_ok += 1
            if any(c.surface == surf for c in cands[:MAX_CANDIDATES]):
                n_surface_ok += 1
            else:
                full = False
                missing_by_reason["surface_not_in_candidates"] = (
                    missing_by_reason.get("surface_not_in_candidates", 0) + 1
                )
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "read": rd,
                            "surface": surf,
                            "reason": "surface_not_in_candidates",
                            "candidates": [c.surface for c in cands[:MAX_CANDIDATES]],
                        }
                    )
            pos = j
        if full:
            n_sent_full += 1
    return {
        "num_sentences": n_sent,
        "num_gold_spans": n_spans,
        "span_in_dict": n_span_ok,
        "span_in_dict_rate": _r(n_span_ok / max(n_spans, 1)),
        "surface_in_candidates": n_surface_ok,
        "surface_in_candidates_rate": _r(n_surface_ok / max(n_spans, 1)),
        "too_long_spans": n_too_long,
        "missing_spans": n_spans - n_surface_ok,
        "missing_by_reason": missing_by_reason,
        "sentences_with_full_path": n_sent_full,
        "sentence_path_rate": _r(n_sent_full / max(n_sent, 1)),
        "missing_examples": examples,
    }


# ---------------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------------


def _r(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


class Acc:
    """1 設定ぶんの指標を貯める。"""

    def __init__(self) -> None:
        self.n = 0
        self.correct = 0
        self.edit_sum = 0
        self.gold_len = 0
        self.cer_sum = 0.0
        self.b_tp = 0
        self.b_fp = 0
        self.b_fn = 0
        self.n_gold_b = 0
        self.n_pred_b = 0

    def add(self, hyp: str, gold: str, gold_b: set, pred_b: set) -> None:
        self.n += 1
        if hyp == gold:
            self.correct += 1
        ed = edit_distance(hyp, gold)
        self.edit_sum += ed
        self.gold_len += len(gold)
        self.cer_sum += ed / max(len(gold), 1)
        self.b_tp += len(pred_b & gold_b)
        self.b_fp += len(pred_b - gold_b)
        self.b_fn += len(gold_b - pred_b)
        self.n_gold_b += len(gold_b)
        self.n_pred_b += len(pred_b)

    def metrics(self) -> Dict[str, Any]:
        n = max(self.n, 1)
        p = self.b_tp / max(self.b_tp + self.b_fp, 1)
        rc = self.b_tp / max(self.b_tp + self.b_fn, 1)
        f1 = 2 * p * rc / max(p + rc, 1e-12)
        return {
            "num_sentences": self.n,
            "sent_acc": _r(self.correct / n),
            "cer_micro": _r(self.edit_sum / max(self.gold_len, 1)),
            "cer_sent_mean": _r(self.cer_sum / n),
            "edit_distance_total": self.edit_sum,
            "gold_chars_total": self.gold_len,
            "boundary": {
                "precision": _r(p),
                "recall": _r(rc),
                "f1": _r(f1),
                "tp": self.b_tp,
                "fp": self.b_fp,
                "fn": self.b_fn,
                "gold_boundaries_total": self.n_gold_b,
                "pred_boundaries_total": self.n_pred_b,
                "gold_per_sentence": _r(self.n_gold_b / n),
                "pred_per_sentence": _r(self.n_pred_b / n),
            },
        }


def score_spans(
    scorer: Any, ctx: str, read: str, paths: Sequence[Sequence[Arc]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """仮説に出てくるスパンだけ BERT にかけ、表記ごとの log P を返す。"""
    span_surfaces: Dict[Tuple[int, int], List[str]] = {}
    for p in paths:
        for a in p:
            lst = span_surfaces.setdefault((a.start, a.end), [])
            if len(lst) < MAX_CANDIDATES and a.cand.surface not in lst:
                lst.append(a.cand.surface)
    keys = list(span_surfaces.keys())
    if not keys:
        return {}
    req = [
        (s, e, [Candidate(x, 1, 0.0) for x in span_surfaces[(s, e)]])
        for s, e in keys
    ]
    lps = scorer.span_logprobs(ctx, read, req)
    logp: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (s, e), lp in zip(keys, lps):
        logp[(s, e)] = {x: float(lp[k]) for k, x in enumerate(span_surfaces[(s, e)])}
    return logp


def rank_paths(
    paths: Sequence[Sequence[Arc]],
    lm: Optional[BigramLM],
    logp: Dict[Tuple[int, int], Dict[str, float]],
    beta: float,
) -> List[List[Arc]]:
    """bigram 連接コスト + BERT スパン logP で再ランクする（evaluate_bigram.py と同一式）。"""
    scored: List[Tuple[float, List[Arc]]] = []
    for p in paths:
        c = 0.0
        prev: Optional[str] = None
        for a in p:
            conn = lm.conn(prev, a.cand.surface, 0.0) if lm is not None else None
            c += a.cost if conn is None else conn
            c += -beta * logp[(a.start, a.end)].get(a.cand.surface, -6.0)
            prev = a.cand.surface
        scored.append((c, list(p)))
    scored.sort(key=lambda x: x[0])
    return [p for _, p in scored]


def unique_surfaces(paths: Sequence[Sequence[Arc]], limit: int) -> List[str]:
    seen: List[str] = []
    for p in paths:
        s = path_surface(p)
        if s not in seen:
            seen.append(s)
            if len(seen) >= limit:
                break
    return seen


def by_length(
    records: Sequence[Tuple[int, bool, int, int]], edges: Sequence[Tuple[int, int]]
) -> List[Dict[str, Any]]:
    """records = (読み長, 正解か, 編集距離, 正解長)。"""
    out: List[Dict[str, Any]] = []
    for lo, hi in edges:
        sel = [r for r in records if lo <= r[0] <= hi]
        if not sel:
            continue
        edit = sum(r[2] for r in sel)
        glen = sum(r[3] for r in sel)
        out.append(
            {
                "read_len_min": lo,
                "read_len_max": hi,
                "num_sentences": len(sel),
                "sent_acc": _r(sum(1 for r in sel if r[1]) / len(sel)),
                "cer_micro": _r(edit / max(glen, 1)),
            }
        )
    return out


def latency_summary(samples: Dict[str, List[float]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, vals in samples.items():
        if not vals:
            continue
        out[name] = {
            "n": len(vals),
            "mean": _r(sum(vals) / len(vals), 3),
            "p50": _r(_pct(vals, 0.50) or 0.0, 3),
            "p95": _r(_pct(vals, 0.95) or 0.0, 3),
            "p99": _r(_pct(vals, 0.99) or 0.0, 3),
            "max": _r(max(vals), 3),
        }
    return out


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------


def run_eval(
    rows: Sequence[Dict[str, Any]],
    lex: Lexicon,
    lm: Optional[BigramLM],
    scorer: Any,
    *,
    gen: str,
    deltas: Sequence[float],
    betas: Sequence[float],
    beam: int,
    want: int,
    acc_k: int = 3,
) -> Dict[str, Any]:
    d0 = deltas[0]
    b0 = betas[0]
    if gen == "unigram":
        primary_key = f"unigram_bert(b={b0})"
    else:
        primary_key = f"bigram_bert(d={d0},b={b0})"

    acc: Dict[str, Acc] = {}
    oracle_hits: Dict[str, int] = {}
    acc_at_k_hits: Dict[str, int] = {}
    lat: Dict[str, List[float]] = {
        "lattice_ms": [],
        "beam_ms": [],
        "bert_ms": [],
        "rerank_ms": [],
    }
    primary_records: List[Tuple[int, bool, int, int]] = []
    n_eval = n_skipped = 0
    n_lattice_ok = 0

    def bump(d: Dict[str, int], k: str) -> None:
        d[k] = d.get(k, 0) + 1

    for rec in rows:
        read, gold, gold_b = gold_info(rec)
        if not read:
            n_skipped += 1
            continue
        ctx = rec.get("ctx", "")
        words = rec["words"]

        t0 = time.perf_counter()
        n, arcs, spans = build_lattice(lex, read)
        t1 = time.perf_counter()

        if lm is not None:
            tu0 = time.perf_counter()
            paths_u = beam_search_bigram(n, arcs, lm, beam, want, 1e9)
            tu1 = time.perf_counter()
            unigram_beam_ms = (tu1 - tu0) * 1000.0
        else:
            tu0 = time.perf_counter()
            paths_u = beam_search_unigram(n, arcs, beam, want)
            tu1 = time.perf_counter()
            unigram_beam_ms = (tu1 - tu0) * 1000.0

        if not paths_u:
            n_skipped += 1
            continue
        n_eval += 1
        if gold_path_in_lattice(spans, words):
            n_lattice_ok += 1

        # --- unigram 生成の指標（常に出す） ---
        hyp_u = path_surface(paths_u[0])
        acc.setdefault("unigram_beam", Acc()).add(
            hyp_u, gold, gold_b, path_boundaries(paths_u[0], n)
        )
        if any(path_surface(p) == gold for p in paths_u):
            bump(oracle_hits, "unigram")

        # --- 生成（bigram なら主 delta を先に） ---
        if gen == "bigram":
            tb0 = time.perf_counter()
            paths_gen = beam_search_bigram(n, arcs, lm, beam, want, d0)  # type: ignore[arg-type]
            tb1 = time.perf_counter()
            beam_primary_ms = (tb1 - tb0) * 1000.0
        else:
            paths_gen = paths_u
            beam_primary_ms = unigram_beam_ms

        if not paths_gen:
            n_skipped += 1
            n_eval -= 1
            continue

        if gen == "bigram":
            acc.setdefault(f"bigram_beam(d={d0})", Acc()).add(
                path_surface(paths_gen[0]),
                gold,
                gold_b,
                path_boundaries(paths_gen[0], n),
            )
            if any(path_surface(p) == gold for p in paths_gen):
                bump(oracle_hits, f"bigram(d={d0})")

        # --- BERT 再ランク（主 delta。タイミングもここで取る） ---
        tt0 = time.perf_counter()
        logp = score_spans(scorer, ctx, read, paths_gen)
        tt1 = time.perf_counter()
        bert_primary_ms = (tt1 - tt0) * 1000.0

        tr0 = time.perf_counter()
        for bi, b in enumerate(betas):
            ranked = rank_paths(paths_gen, lm, logp, b)
            key = (
                f"unigram_bert(b={b})"
                if gen == "unigram"
                else f"bigram_bert(d={d0},b={b})"
            )
            acc.setdefault(key, Acc()).add(
                path_surface(ranked[0]),
                gold,
                gold_b,
                path_boundaries(ranked[0], n),
            )
            if gold in unique_surfaces(ranked, acc_k):
                bump(acc_at_k_hits, key)
            if bi == 0:
                tr1 = time.perf_counter()
                if key == primary_key:
                    hyp = path_surface(ranked[0])
                    primary_records.append(
                        (n, hyp == gold, edit_distance(hyp, gold), len(gold))
                    )

        # --- 残りの delta（タイミング対象外） ---
        if gen == "bigram":
            for d in deltas[1:]:
                paths_d = beam_search_bigram(n, arcs, lm, beam, want, d)  # type: ignore[arg-type]
                if not paths_d:
                    continue
                acc.setdefault(f"bigram_beam(d={d})", Acc()).add(
                    path_surface(paths_d[0]),
                    gold,
                    gold_b,
                    path_boundaries(paths_d[0], n),
                )
                if any(path_surface(p) == gold for p in paths_d):
                    bump(oracle_hits, f"bigram(d={d})")
                logp_d = score_spans(scorer, ctx, read, paths_d)
                for b in betas:
                    ranked = rank_paths(paths_d, lm, logp_d, b)
                    key = f"bigram_bert(d={d},b={b})"
                    acc.setdefault(key, Acc()).add(
                        path_surface(ranked[0]),
                        gold,
                        gold_b,
                        path_boundaries(ranked[0], n),
                    )
                    if gold in unique_surfaces(ranked, acc_k):
                        bump(acc_at_k_hits, key)

        lat["lattice_ms"].append((t1 - t0) * 1000.0)
        lat["beam_ms"].append(beam_primary_ms)
        lat["bert_ms"].append(bert_primary_ms)
        lat["rerank_ms"].append((tr1 - tr0) * 1000.0)

    total_ms = [
        a + b + c + d
        for a, b, c, d in zip(
            lat["lattice_ms"], lat["beam_ms"], lat["bert_ms"], lat["rerank_ms"]
        )
    ]

    metrics: Dict[str, Any] = {
        name: a.metrics() for name, a in acc.items()
    }
    for name, m in metrics.items():
        if name in acc_at_k_hits:
            m[f"acc_at_{acc_k}"] = _r(acc_at_k_hits[name] / max(m["num_sentences"], 1))

    oracles = {
        name: {
            "hits": hits,
            "rate": _r(hits / max(n_eval, 1)),
            "num_sentences": n_eval,
        }
        for name, hits in oracle_hits.items()
    }

    lat_out = latency_summary(lat)
    lat_out["total_ms"] = latency_summary({"x": total_ms})["x"]

    return {
        "config": {
            "gen": gen,
            "deltas": list(deltas),
            "betas": list(betas),
            "beam": beam,
            "want": want,
            "acc_k": acc_k,
            "max_span_len": MAX_SPAN_LEN,
            "max_candidates": MAX_CANDIDATES,
        },
        "counts": {
            "num_input_sentences": len(rows),
            "num_evaluated": n_eval,
            "num_skipped": n_skipped,
            "num_with_gold_path_in_lattice": n_lattice_ok,
            "lattice_ceiling_rate": _r(n_lattice_ok / max(n_eval, 1)),
        },
        "metrics": metrics,
        "oracle": oracles,
        "latency": lat_out,
        "primary_config": primary_key,
        "primary_by_read_len": by_length(primary_records, LENGTH_EDGES),
    }


def run_sweep(
    rows: Sequence[Dict[str, Any]],
    lex: Lexicon,
    lm: Optional[BigramLM],
    configs: Sequence[Tuple[int, int]],
    deltas: Sequence[float],
    gen: str,
) -> List[Dict[str, Any]]:
    """beam/want ごとに天井（N-best oracle）と 1 位精度だけを測る（BERT 無し）。"""
    results: List[Dict[str, Any]] = []
    for beam, want in configs:
        n_eval = 0
        uni_hits = 0
        uni_best = 0
        bg_hits: Dict[str, int] = {}
        bg_best: Dict[str, int] = {}
        for rec in rows:
            read, gold, _gb = gold_info(rec)
            if not read:
                continue
            n, arcs, _spans = build_lattice(lex, read)
            if lm is not None:
                paths_u = beam_search_bigram(n, arcs, lm, beam, want, 1e9)
            else:
                paths_u = beam_search_unigram(n, arcs, beam, want)
            if not paths_u:
                continue
            n_eval += 1
            if path_surface(paths_u[0]) == gold:
                uni_best += 1
            if any(path_surface(p) == gold for p in paths_u):
                uni_hits += 1
            if gen == "bigram" and lm is not None:
                for d in deltas:
                    ds = str(d)
                    paths_d = beam_search_bigram(n, arcs, lm, beam, want, d)
                    if not paths_d:
                        continue
                    bg_hits[ds] = bg_hits.get(ds, 0) + int(
                        any(path_surface(p) == gold for p in paths_d)
                    )
                    bg_best[ds] = bg_best.get(ds, 0) + int(
                        path_surface(paths_d[0]) == gold
                    )
        entry: Dict[str, Any] = {
            "beam": beam,
            "want": want,
            "num_sentences": n_eval,
            "unigram_best_acc": _r(uni_best / max(n_eval, 1)),
            "unigram_oracle": _r(uni_hits / max(n_eval, 1)),
        }
        if gen == "bigram" and lm is not None:
            entry["bigram_best_acc"] = {
                k: _r(v / max(n_eval, 1)) for k, v in bg_best.items()
            }
            entry["bigram_oracle"] = {
                k: _r(v / max(n_eval, 1)) for k, v in bg_hits.items()
            }
        results.append(entry)
        print(
            f"  beam={beam:<4} want={want:<4} "
            f"uni_best={entry['unigram_best_acc']:.4f} "
            f"uni_oracle={entry['unigram_oracle']:.4f}"
            + (
                " "
                + " ".join(
                    f"bg_oracle(d={k})={v:.4f}"
                    for k, v in entry.get("bigram_oracle", {}).items()
                )
                if entry.get("bigram_oracle")
                else ""
            ),
            flush=True,
        )
    return results


# ---------------------------------------------------------------------------
# メタ情報
# ---------------------------------------------------------------------------


def resolve_asset(path: str, data_dir: str) -> str:
    """アセットのパス解決。

    `--lexicon lexicon_v2.tsv`（data-dir 相対）と
    `--lexicon data/lexicon_v3_frozen.tsv`（cwd 相対）の両方を受ける。
    """
    if not path:
        return ""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(data_dir, path)


def file_meta(path: str, do_hash: bool = True) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    meta: Dict[str, Any] = {
        "path": path,
        "exists": True,
        "size": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(
            timespec="seconds"
        ),
    }
    if do_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        meta["sha256"] = h.hexdigest()
    return meta


def parse_sweep(text: str) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        b, w = part.split(",")
        configs.append((int(b), int(w)))
    return configs


def write_json(path: str, obj: Dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def print_metrics(res: Dict[str, Any], domain: str, data_path: str) -> None:
    cfg = res["config"]
    print(f"\ndomain={domain or '-'}  data={data_path}")
    print(
        f"gen={cfg['gen']} beam={cfg['beam']} want={cfg['want']} "
        f"deltas={cfg['deltas']} betas={cfg['betas']}"
    )
    print(f"sentences: evaluated={res['counts']['num_evaluated']} "
          f"skipped={res['counts']['num_skipped']} "
          f"lattice_ceiling={res['counts']['lattice_ceiling_rate']:.4f}")
    print()
    acc_key = f"acc_at_{cfg['acc_k']}"
    header = (
        f"{'config':<28} {'sent_acc':>8} {'CER_mic':>8} {'CER_sent':>8} "
        f"{'bF1':>6} {'bP':>6} {'bR':>6} {acc_key:>7}"
    )
    print(header)
    print("-" * len(header))
    for name, m in res["metrics"].items():
        b = m["boundary"]
        extra = f" {m[acc_key]:>7.4f}" if acc_key in m else f" {'':>7}"
        print(
            f"{name:<28} {m['sent_acc']:>8.4f} {m['cer_micro']:>8.4f} "
            f"{m['cer_sent_mean']:>8.4f} {b['f1']:>6.4f} {b['precision']:>6.4f} "
            f"{b['recall']:>6.4f}{extra}"
        )
    print("\noracle (N-best に正解文が入る率 / 天井)")
    for name, o in res["oracle"].items():
        print(f"  {name:<24} {o['rate']:.4f}  ({o['hits']}/{o['num_sentences']})")
    print("\nlatency [ms/sentence, CPU int8]")
    for name in ("lattice_ms", "beam_ms", "bert_ms", "rerank_ms", "total_ms"):
        if name in res["latency"]:
            x = res["latency"][name]
            print(
                f"  {name:<11} p50={x['p50']:>8.2f} p95={x['p95']:>8.2f} "
                f"p99={x['p99']:>8.2f} mean={x['mean']:>8.2f}"
            )
    print("\nby read length (primary config = " + res["primary_config"] + ")")
    for row in res["primary_by_read_len"]:
        print(
            f"  {row['read_len_min']:>3}-{row['read_len_max']:<3} "
            f"n={row['num_sentences']:>4} sent_acc={row['sent_acc']:.4f} "
            f"CER={row['cer_micro']:.4f}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="変換精度の評価 v2（CER / 天井 / 境界F / acc@k / レイテンシ / スイープ）"
    )
    ap.add_argument("--data", default="data/valid.jsonl", help="評価用 jsonl")
    ap.add_argument("--data-dir", default="data", help="辞書/データのルート")
    ap.add_argument("--lexicon", default="lexicon_v2.tsv")
    ap.add_argument("--bigram", default="data/bigram.tsv",
                    help="空文字で bigram 無し（unigram+BERT）")
    ap.add_argument("--bigram-words", default="data/bigram_words.tsv")
    ap.add_argument("--checkpoint", default="models/small_v3.pt")
    ap.add_argument("--onnx", default="android/app/src/main/assets/model/scorer.onnx",
                    help="空文字で checkpoint を使う")
    ap.add_argument("--vocab", default="android/app/src/main/assets/model/vocab.txt")
    ap.add_argument("--onnx-threads", type=int, default=4,
                    help="ONNX Runtime の intra_op スレッド数（端末 proxy として 4）")
    ap.add_argument("--score-mode", default="logprob", choices=["logprob", "logit"])
    ap.add_argument("--domain", default="", help="評価セットのドメイン名（JSON に記録）")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--betas", default="1.0,2.0,4.0,6.0")
    ap.add_argument("--deltas", default="1.0,2.0,3.0")
    ap.add_argument("--gen", default="bigram", choices=["unigram", "bigram"])
    ap.add_argument("--beam", type=int, default=32)
    ap.add_argument("--want", type=int, default=48)
    ap.add_argument("--acc-k", type=int, default=3)
    ap.add_argument("--json-out", default="", help="結果 JSON の出力先")
    ap.add_argument("--beam-sweep", default="",
                    help='"32,48;64,128;128,256" 形式。beam;want の組を複数')
    ap.add_argument("--sweep-only", action="store_true",
                    help="天井スイープだけ実行（BERT を読み込まない）")
    ap.add_argument("--coverage-only", action="store_true",
                    help="gold span coverage の分析だけ実行（ビームも BERT も回さない）")
    ap.add_argument("--no-hash", action="store_true", help="アセットの sha256 を省く")
    args = ap.parse_args()

    if args.gen == "bigram" and not args.bigram:
        ap.error("--gen bigram には --bigram が必要です")

    data_path = args.data
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        data_path = os.path.join(args.data_dir, data_path)
    if not os.path.exists(data_path):
        ap.error(f"評価データが見つかりません: {args.data}")

    rows = load_rows(data_path, args.limit)
    print(f"data: {data_path}  rows={len(rows)}  domain={args.domain or '-'}", flush=True)

    lex_path = resolve_asset(args.lexicon, args.data_dir)
    bigram_path = resolve_asset(args.bigram, args.data_dir)
    bigram_words_path = resolve_asset(args.bigram_words, args.data_dir)
    onnx_path = resolve_asset(args.onnx, args.data_dir)
    vocab_path = resolve_asset(args.vocab, args.data_dir)
    checkpoint_path = resolve_asset(args.checkpoint, args.data_dir)

    lex: Lexicon = load_lexicon(lex_path, MAX_CANDIDATES)
    lm: Optional[BigramLM] = None
    if bigram_path and not args.coverage_only:
        lm = BigramLM(bigram_path, bigram_words_path)

    scorer: Optional[Any] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.sweep_only and not args.coverage_only:
        if onnx_path:
            scorer = OnnxScorer(
                onnx_path, vocab_path, threads=args.onnx_threads, mode=args.score_mode
            )
            print(f"scorer: onnx(int8 cpu, threads={args.onnx_threads}) {onnx_path}")
        else:
            scorer = TorchScorer(checkpoint_path, device)
            print(f"scorer: checkpoint {checkpoint_path} ({device})")

    payload: Dict[str, Any] = {
        "meta": {
            "tool": "train.evaluate_v2",
            "version": TOOL_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "domain": args.domain,
            "argv": sys.argv,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "note": (
                "生成・再ランクの数式は evaluate_bigram.py と同一。"
                "レイテンシは CPU int8 ONNX の 1 文あたり"
                "(ラティス+主 delta ビーム+BERT+主 beta 再ランク)。"
            ),
        },
        "data": {
            "path": data_path,
            "domain": args.domain,
            "limit": args.limit,
            "num_sentences": len(rows),
            "sha256": file_meta(data_path, not args.no_hash).get("sha256"),
        },
        "assets": {
            "lexicon": file_meta(lex_path, not args.no_hash),
            "bigram": file_meta(bigram_path, not args.no_hash) if bigram_path else {},
            "bigram_words": (
                file_meta(bigram_words_path, not args.no_hash) if bigram_path else {}
            ),
            "onnx": file_meta(onnx_path, not args.no_hash) if onnx_path else {},
            "vocab": file_meta(vocab_path, not args.no_hash) if onnx_path else {},
            "checkpoint": (
                file_meta(checkpoint_path, not args.no_hash)
                if (not onnx_path)
                else {}
            ),
        },
        "scorer": scorer.meta if scorer is not None else {},
        "lexicon_stats": {
            "entries": len(lex.entries),
            "max_len": lex.max_len,
            "unk_cost": lex.unk_cost,
            "topk": MAX_CANDIDATES,
        },
    }
    if lm is not None:
        payload["bigram_stats"] = {
            "words": len(lm.words),
            "pairs": len(lm.table),
        }

    coverage = analyze_gold_coverage(lex, rows)
    payload["gold_span_coverage"] = coverage
    print(
        "gold span coverage: "
        f"span_in_dict={coverage['span_in_dict_rate']:.4f} "
        f"surface_in_candidates={coverage['surface_in_candidates_rate']:.4f} "
        f"full_path_sentence={coverage['sentence_path_rate']:.4f} "
        f"({coverage['sentences_with_full_path']}/{coverage['num_sentences']})",
        flush=True,
    )
    print(
        "  missing spans by reason: "
        + ", ".join(
            f"{k}={v}" for k, v in sorted(coverage["missing_by_reason"].items())
        ),
        flush=True,
    )

    if args.coverage_only:
        if args.json_out:
            write_json(args.json_out, payload)
        return

    if not args.sweep_only:
        res = run_eval(
            rows,
            lex,
            lm,
            scorer,
            gen=args.gen,
            deltas=[float(x) for x in args.deltas.split(",")],
            betas=[float(x) for x in args.betas.split(",")],
            beam=args.beam,
            want=args.want,
            acc_k=args.acc_k,
        )
        payload.update(res)
        print_metrics(res, args.domain, data_path)

    if args.beam_sweep:
        sweep_configs = parse_sweep(args.beam_sweep)
        print(
            f"\nbeam/want sweep: {sweep_configs}  "
            f"(lattice_ceiling={coverage['sentence_path_rate']:.4f})",
            flush=True,
        )
        payload["beam_sweep"] = run_sweep(
            rows,
            lex,
            lm,
            sweep_configs,
            [float(x) for x in args.deltas.split(",")],
            args.gen,
        )
        payload["beam_sweep_note"] = (
            "天井は N-best に正解文が入る率。unigram 系は bigram 無しのビーム、"
            "bigram 系は各 delta のビーム。"
        )

    if args.json_out:
        write_json(args.json_out, payload)


if __name__ == "__main__":
    main()
