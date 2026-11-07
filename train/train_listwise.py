"""経路（path）単位の listwise / pairwise 再ランク学習。

文頭 prior-wrong の N-best（`train.build_rerank_nbest` の出力）を使い、
「正解経路 > hard negative 経路」になるようにスパン採点器をファインチューニングする。

  # 1) N-best を作る（CPU・数分〜）
  python -m train.build_rerank_nbest --limit 50000 --head-only \
      --out logs/rerank_nbest_head.jsonl

  # 2) 学習（GPU 推奨）
  python -m train.train_listwise --data logs/rerank_nbest_head.jsonl \
      --init models/scorer_v3.pt --out models/scorer_rerank.pt \
      --epochs 1 --batch-sentences 8 --loss listwise

  # 3) 評価（文頭の prior-wrong セット）
  python -m train.evaluate_v2 --data data/eval/head_prior_wrong.jsonl \
      --checkpoint models/scorer_rerank.pt --domain head --limit 0 \
      --deltas 1.0 --betas 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.scorer import (  # noqa: E402
    CharVocab,
    SpanScorer,
    build_candidate_table,
    build_encoder,
)

MAX_K = 8
MAX_CTX = 64


def load_records(path: str, limit: int) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def build_sentence(rec: dict, vocab: CharVocab, device: torch.device, max_paths: int):
    paths = rec["paths"][:max_paths]
    gold = rec["gold"]
    if gold >= len(paths):
        return None
    span_id: Dict[Tuple[int, int], int] = {}
    cands: List[List[str]] = []
    for p in paths:
        for s, e, surf in p:
            key = (s, e)
            if key not in span_id:
                span_id[key] = len(cands)
                cands.append([])
            lst = cands[span_id[key]]
            if surf not in lst and len(lst) < MAX_K:
                lst.append(surf)
    W = len(cands)
    if W == 0:
        return None
    spans = [None] * W
    for key, idx in span_id.items():
        spans[idx] = key

    ctx_ids = vocab.encode(rec.get("ctx", "")[-MAX_CTX:])
    read_ids = vocab.encode(rec["read"])
    seq = [vocab.cls] + ctx_ids + [vocab.sep] + read_ids + [vocab.sep]
    off = 1 + len(ctx_ids) + 1
    surfaces = [s for lst in cands for s in lst]
    sid = {s: i for i, s in enumerate(surfaces)}
    cidx = torch.zeros((1, W, MAX_K), dtype=torch.long, device=device)
    cmask = torch.zeros((1, W, MAX_K), dtype=torch.float, device=device)
    starts = torch.zeros((1, W), dtype=torch.long, device=device)
    ends = torch.zeros((1, W), dtype=torch.long, device=device)
    for w, (s, e) in enumerate(spans):
        starts[0, w] = off + s
        ends[0, w] = off + e
        for j, surf in enumerate(cands[w]):
            cidx[0, w, j] = sid[surf]
            cmask[0, w, j] = 1.0
    table = build_candidate_table(surfaces, vocab, device)[:3]
    path_idx: List[List[Tuple[int, int]]] = []
    for p in paths:
        one = []
        for s, e, surf in p:
            w = span_id[(s, e)]
            j = cands[w].index(surf)
            one.append((w, j))
        path_idx.append(one)
    return {
        "input_ids": torch.tensor([seq], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(seq)), dtype=torch.long, device=device),
        "starts": starts,
        "ends": ends,
        "cand_index": cidx,
        "cand_mask": cmask,
        "table": table,
        "paths": path_idx,
        "gold": gold,
    }


def path_scores(model, batch) -> torch.Tensor:
    logits = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["starts"],
        batch["ends"],
        batch["cand_index"],
        *batch["table"],
    )
    logits = logits.masked_fill(batch["cand_mask"] == 0, -1e4)
    logp = F.log_softmax(logits, dim=-1)  # (1,W,K)
    scores = []
    for seq in batch["paths"]:
        sc = torch.zeros((), device=logp.device)
        for w, j in seq:
            sc = sc + logp[0, w, j]
        scores.append(sc)
    return torch.stack(scores)


def main() -> None:
    ap = argparse.ArgumentParser(description="経路単位の listwise/pairwise 再ランク学習")
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default="models/scorer_v3.pt")
    ap.add_argument("--out", default="models/scorer_rerank.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-sentences", type=int, default=8)
    ap.add_argument("--loss", choices=["listwise", "pairwise"], default="listwise")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--max-paths", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "scratch-small")
    vocab_size = ck.get("vocab_size", 7027)
    model_name = ck.get("model_name", "")
    vocab_path = ck.get("vocab_path") or "android/app/src/main/assets/model/vocab.txt"
    print(f"init: arch={arch} hidden={ck['hidden']} vocab={vocab_size}", flush=True)

    encoder = build_encoder(arch, vocab_size, model_name)
    model = SpanScorer(encoder, ck["hidden"]).to(device)
    model.load_state_dict(ck["model"])
    model.train()
    vocab = CharVocab(vocab_path)

    records = load_records(args.data, args.limit)
    print(f"records: {len(records)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(records)
        opt.zero_grad()
        loss_sum = 0.0
        n_batch = 0
        for i, rec in enumerate(records):
            batch = build_sentence(rec, vocab, device, args.max_paths)
            if batch is None or len(batch["paths"]) < 2:
                continue
            scores = path_scores(model, batch)
            gold = batch["gold"]
            if args.loss == "listwise":
                loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold], device=scores.device))
            else:
                others = torch.cat([scores[:gold], scores[gold + 1 :]])
                loss = F.softplus(-(scores[gold] - others - args.margin)).mean()
            (loss / args.batch_sentences).backward()
            loss_sum += loss.item()
            n_batch += 1
            if n_batch % args.batch_sentences == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                if step % args.log_every == 0:
                    print(
                        f"  step={step} loss={loss_sum / n_batch:.4f} "
                        f"({time.time() - t0:.0f}s)",
                        flush=True,
                    )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        print(
            f"[epoch {epoch}] sentences={n_batch} loss={loss_sum / max(n_batch, 1):.4f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": ck.get("model_name", ""),
            "arch": arch,
            "vocab_path": vocab_path,
            "hidden": ck["hidden"],
            "surfaces": ck.get("surfaces", []),
            "k": ck.get("k", MAX_K),
            "vocab_size": vocab_size,
        },
        args.out,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
