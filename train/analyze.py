"""文脈の寄与を切り分ける分析。

「BERT が賢い」のか「BERT が前方文脈を使っている」のかを分離する。
正解分割を与えた上での候補選択精度を、入力条件を変えて比較する。

  A) unigram            : 辞書頻度のみ
  B) BERT(単語のみ)      : その単語の読みだけを入力（文脈ゼロ）
  C) BERT(文のみ)        : 文全体の読み（前方文脈なし）
  D) BERT(文+前方文脈)   : 文全体の読み + 直前の確定済みテキスト

C -> D の差が「前方文脈」の寄与、B -> C の差が「文内の読み文脈」の寄与。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Candidate, Lexicon  # noqa: E402
from train.evaluate import ModelScorer, gold_spans  # noqa: E402
from train.train import load_lexicon  # noqa: E402

MAX_CTX = 64


@torch.no_grad()
def score_rows(
    scorer: ModelScorer,
    rows: Sequence[Tuple[str, str, int, int, Sequence[Candidate]]],
    batch_size: int = 128,
) -> List[torch.Tensor]:
    """rows = (ctx, read, start, end, cands) をまとめて採点する。"""
    out: List[torch.Tensor] = []
    for b0 in range(0, len(rows), batch_size):
        part = rows[b0 : b0 + batch_size]
        seqs, spans, K = [], [], max(len(r[4]) for r in part)
        for ctx, read, s, e, cands in part:
            cids = scorer.vocab.encode(ctx[-MAX_CTX:])
            rids = scorer.vocab.encode(read)
            seq = [scorer.vocab.cls] + cids + [scorer.vocab.sep] + rids + [
                scorer.vocab.sep
            ]
            off = 1 + len(cids) + 1
            seqs.append(seq)
            spans.append((off + s, off + e))
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(part), L), dtype=torch.long, device=scorer.device)
        am = torch.zeros((len(part), L), dtype=torch.long, device=scorer.device)
        starts = torch.zeros((len(part), 1), dtype=torch.long, device=scorer.device)
        ends = torch.zeros((len(part), 1), dtype=torch.long, device=scorer.device)
        cidx = torch.zeros(
            (len(part), 1, K), dtype=torch.long, device=scorer.device
        )
        cmask = torch.zeros(
            (len(part), 1, K), dtype=torch.float, device=scorer.device
        )
        for i, (seq, (s, e), (_, _, _, _, cands)) in enumerate(
            zip(seqs, spans, part)
        ):
            ids[i, : len(seq)] = torch.tensor(seq, device=scorer.device)
            am[i, : len(seq)] = 1
            starts[i, 0] = s
            ends[i, 0] = e
            for j, c in enumerate(cands):
                cidx[i, 0, j] = scorer.surface_id[c.surface]
                cmask[i, 0, j] = 1.0

        hidden = scorer.model.encode_text(ids, am)
        h_span = scorer.model.pool_spans(hidden, starts, ends)  # (B,1,H)
        _, _, H = h_span.shape
        tci, tcm, tlen = scorer.table
        uid, inv = torch.unique(cidx.reshape(-1), return_inverse=True)
        ce = scorer.model.emb(tci[uid])
        m = tcm[uid].unsqueeze(-1).to(ce.dtype)
        ce = (ce * m).sum(1) / tlen[uid].clamp(min=1.0).unsqueeze(-1)
        c = ce[inv].view(len(part), 1, K, H)
        hs = h_span.unsqueeze(2).expand(-1, -1, K, -1)
        feats = torch.cat([hs, c, hs * c], dim=-1)
        logits = scorer.model.out(scorer.model.proj(feats)).squeeze(-1)[:, 0]
        logits = logits.masked_fill(cmask[:, 0] == 0, -1e4)
        lp = torch.log_softmax(logits.float(), dim=-1)
        for i, (_, _, _, _, cands) in enumerate(part):
            out.append(lp[i, : len(cands)].cpu())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/scorer_v2.pt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = ModelScorer(args.checkpoint, device)
    lex: Lexicon = load_lexicon(os.path.join(args.data_dir, "lexicon.tsv"), scorer.k)

    items = []  # (ctx, read, s, e, cands, gold_surface, has_ctx)
    with open(os.path.join(args.data_dir, "valid.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            rec = json.loads(line)
            read = rec["read"]
            ctx = rec["ctx"]
            has_ctx = bool(ctx.strip())
            span_map = {(a, b): list(c) for a, b, c in lex.spans(read)}
            for s, e, surf in gold_spans(rec["words"]):
                cands = span_map.get((s, e))
                if cands is None:
                    continue
                items.append((ctx, read, s, e, cands, surf, has_ctx))
    print(f"spans={len(items)} (with ctx={sum(1 for x in items if x[6])})")

    conds = {
        "A unigram": None,
        "B word-only": [(("", x[1][x[2] : x[3]], 0, x[3] - x[2], x[4])) for x in items],
        "C sentence": [(("", x[1], x[2], x[3], x[4])) for x in items],
        "D +preceding": [((x[0], x[1], x[2], x[3], x[4])) for x in items],
    }

    results: Dict[str, List[bool]] = {}
    for name, rows in conds.items():
        if rows is None:
            hits = []
            for ctx, read, s, e, cands, surf, has in items:
                best = min(range(len(cands)), key=lambda k: cands[k].cost)
                hits.append(cands[best].surface == surf)
            results[name] = hits
        else:
            lps = score_rows(scorer, rows)
            hits = []
            for (ctx, read, s, e, cands, surf, has), lp in zip(items, lps):
                hits.append(cands[int(lp.argmax())].surface == surf)
            results[name] = hits

    groups = {
        "ALL": list(range(len(items))),
        "with preceding ctx": [i for i, x in enumerate(items) if x[6]],
        "no preceding ctx": [i for i, x in enumerate(items) if not x[6]],
    }
    print(f"\n{'condition':<16}" + "".join(f"{g:>22}" for g in groups))
    for name, hits in results.items():
        row = f"{name:<16}"
        for g, idx in groups.items():
            acc = sum(hits[i] for i in idx) / max(len(idx), 1)
            row += f"{acc:>22.4f}"
        print(row)
    print()
    for g, idx in groups.items():
        print(f"{g}: n={len(idx)}")


if __name__ == "__main__":
    main()
