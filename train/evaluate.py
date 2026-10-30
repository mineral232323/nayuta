"""変換精度の評価: 辞書 unigram ベースライン vs BERT 採点。

  python -m train.evaluate --checkpoint models/scorer.pt --data-dir data --limit 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Arc, Candidate, Lexicon, build_arcs, to_surfaces, viterbi  # noqa: E402
from train.scorer import CharVocab, SpanScorer, build_candidate_table  # noqa: E402
from train.train import load_lexicon  # noqa: E402

MAX_CTX = 64


class ModelScorer:
    def __init__(self, checkpoint: str, device: torch.device):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(ck["model_name"])
        self.model = SpanScorer(encoder, ck["hidden"]).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        from huggingface_hub import hf_hub_download

        self.vocab = CharVocab(hf_hub_download(ck["model_name"], "vocab.txt"))
        self.surfaces: List[str] = ck["surfaces"]
        self.surface_id = {s: i for i, s in enumerate(self.surfaces)}
        self.k = ck["k"]
        self.device = device
        ids, mask, lens, _ = build_candidate_table(self.surfaces, self.vocab, device)
        self.table = (ids, mask, lens)

    @torch.no_grad()
    def span_logprobs(
        self,
        ctx: str,
        read: str,
        spans: Sequence[Tuple[int, int, Sequence[Candidate]]],
        chunk: int = 256,
    ) -> List[torch.Tensor]:
        """各スパンの候補に対する log P を返す。"""
        cids = self.vocab.encode(ctx[-MAX_CTX:])
        rids = self.vocab.encode(read)
        seq = [self.vocab.cls] + cids + [self.vocab.sep] + rids + [self.vocab.sep]
        off = 1 + len(cids) + 1
        input_ids = torch.tensor([seq], device=self.device)
        am = torch.ones_like(input_ids)
        hidden = self.model.encode_text(input_ids, am)

        out: List[torch.Tensor] = []
        K = max(len(c) for _, _, c in spans)
        for s in range(0, len(spans), chunk):
            part = spans[s : s + chunk]
            W = len(part)
            starts = torch.zeros((1, W), dtype=torch.long, device=self.device)
            ends = torch.zeros((1, W), dtype=torch.long, device=self.device)
            cidx = torch.zeros((1, W, K), dtype=torch.long, device=self.device)
            cmask = torch.zeros((1, W, K), dtype=torch.float, device=self.device)
            for w, (i, j, cands) in enumerate(part):
                starts[0, w] = off + i
                ends[0, w] = off + j
                for c_i, c in enumerate(cands):
                    cidx[0, w, c_i] = self.surface_id[c.surface]
                    cmask[0, w, c_i] = 1.0
            h_span = self.model.pool_spans(hidden, starts, ends)
            _, _, H = h_span.shape
            tci, tcm, tlen = self.table
            uid, inv = torch.unique(cidx.reshape(-1), return_inverse=True)
            ce = self.model.emb(tci[uid])
            m = tcm[uid].unsqueeze(-1).to(ce.dtype)
            ce = (ce * m).sum(1) / tlen[uid].clamp(min=1.0).unsqueeze(-1)
            c = ce[inv].view(1, W, K, H)
            hs = h_span.unsqueeze(2).expand(-1, -1, K, -1)
            feats = torch.cat([hs, c, hs * c], dim=-1)
            logits = self.model.out(self.model.proj(feats)).squeeze(-1)[0]  # (W,K)
            logits = logits.masked_fill(cmask[0] == 0, -1e4)
            lp = torch.log_softmax(logits.float(), dim=-1)
            for w, (_, _, cands) in enumerate(part):
                out.append(lp[w, : len(cands)].cpu())
        return out


def gold_sentence(words: Sequence[Sequence[str]]) -> str:
    return "".join(w[1] for w in words)


def gold_spans(words: Sequence[Sequence[str]]):
    pos = 0
    res = []
    for rd, surf in words:
        res.append((pos, pos + len(rd), surf))
        pos += len(rd)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/scorer.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--betas", default="0.2,0.5,1.0,2.0")
    ap.add_argument("--no-context", action="store_true", help="文脈を落として寄与を測る")
    ap.add_argument("--dump", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ModelScorer(args.checkpoint, device)
    lex: Lexicon = load_lexicon(os.path.join(args.data_dir, "lexicon.tsv"), scorer.k)
    betas = [float(b) for b in args.betas.split(",")]

    rows = []
    with open(os.path.join(args.data_dir, "valid.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.limit:
                break

    # --- 正解分割が与えられたときの候補選択精度 ---
    base_hit = model_hit = total = cover = 0
    base_sent = 0
    model_sent = {b: 0 for b in betas}
    win = {b: 0 for b in betas}   # BERT だけ正解
    lose = {b: 0 for b in betas}  # ベースラインだけ正解
    n_sent = 0
    dumps: List[str] = []

    for rec in rows:
        read, words = rec["read"], rec["words"]
        ctx = "" if args.no_context else rec["ctx"]
        spans = lex.spans(read)
        if not spans:
            continue
        span_map: Dict[Tuple[int, int], List[Candidate]] = {
            (i, j): list(c) for i, j, c in spans
        }
        lps = scorer.span_logprobs(ctx, read, [(i, j, c) for (i, j), c in span_map.items()])
        lp_map = {k: lp for k, lp in zip(span_map.keys(), lps)}

        gold = gold_spans(words)
        gold_str = gold_sentence(words)
        n_sent += 1
        total += len(gold)

        for s, e, surf in gold:
            cands = span_map.get((s, e))
            if cands is None:
                continue
            cover += 1
            lp = lp_map[(s, e)]
            costs = [c.cost for c in cands]
            if cands[costs.index(min(costs))].surface == surf:
                base_hit += 1
            if cands[int(lp.argmax())].surface == surf:
                model_hit += 1

        arcs = build_arcs(lex, read)
        _, path = viterbi(read, arcs)
        base_hyp = to_surfaces(path)
        base_ok = base_hyp == gold_str
        if base_ok:
            base_sent += 1

        for b in betas:
            extra = {
                key: [float(-b * lp[k]) for k in range(len(cands))]
                for key, cands in span_map.items()
                for lp in [lp_map[key]]
            }

            def score_fn(i, j, cands, _e=extra):
                return _e.get((i, j), [0.0] * len(cands))

            arcs2 = build_arcs(lex, read, score_fn)
            _, path2 = viterbi(read, arcs2)
            hyp = to_surfaces(path2)
            ok = hyp == gold_str
            if ok:
                model_sent[b] += 1
                if not base_ok:
                    win[b] += 1
            elif base_ok:
                lose[b] += 1
            if not ok and not base_ok and len(dumps) < args.dump:
                dumps.append(
                    f"read : {read}\n  gold: {gold_str}\n  base: {base_hyp}\n  bert: {hyp}\n"
                )

    print(f"sentences={n_sent}")
    print(f"gold span coverage={cover}/{total} ({cover / max(total, 1):.3f})")
    print("candidate acc (gold segmentation given):")
    print(f"  unigram argmax : {base_hit / max(cover, 1):.4f}")
    print(f"  BERT argmax    : {model_hit / max(cover, 1):.4f}")
    print("sentence accuracy (with segmentation search):")
    for b in betas:
        print(
            f"  beta={b:<4} unigram={base_sent / max(n_sent, 1):.4f} "
            f"bert={model_sent[b] / max(n_sent, 1):.4f} "
            f"(bert_win={win[b]} bert_lose={lose[b]})"
        )
    if dumps:
        print("\n--- 両方外した例 ---")
        for d in dumps:
            print(d)


if __name__ == "__main__":
    main()
