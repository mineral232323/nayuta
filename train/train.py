"""BERT 文節スパン採点器の学習。

  python -m train.train --data-dir data --epochs 2 --batch-size 8

入力: data/lexicon.tsv, data/train.jsonl, data/valid.jsonl
出力: models/scorer.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import Lexicon  # noqa: E402
from train.scorer import CharVocab, SpanScorer, build_candidate_table  # noqa: E402

MAX_CTX = 64
MAX_READ = 48
MAX_WORDS = 24


def load_lexicon(path: str, topk: int) -> Lexicon:
    entries: Dict[str, List[Tuple[str, int]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rd, surf, c = line.rstrip("\n").split("\t")
            entries.setdefault(rd, []).append((surf, int(c)))
    return Lexicon(entries, topk=topk)


class ConvDataset(Dataset):
    def __init__(self, path: str, lex: Lexicon, vocab: CharVocab, k: int):
        self.path = path
        self.lex = lex
        self.vocab = vocab
        self.k = k
        self.offsets: List[int] = []
        self._fh = None
        with open(path, "rb") as f:
            pos = 0
            for line in f:
                self.offsets.append(pos)
                pos += len(line)

    def __len__(self) -> int:
        return len(self.offsets)

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.path, "rb")
        return self._fh

    def __getitem__(self, idx: int) -> Optional[dict]:
        fh = self._handle()
        fh.seek(self.offsets[idx])
        rec = json.loads(fh.readline())
        read: str = rec["read"]
        ctx: str = rec["ctx"][-MAX_CTX:]
        pos = 0
        starts, ends, cand_lists, golds = [], [], [], []
        for rd, surf in rec["words"]:
            s, e = pos, pos + len(rd)
            pos = e
            cands = self.lex.candidates(rd)
            if len(cands) < 2:
                continue
            surfaces = [c.surface for c in cands[: self.k]]
            if surf not in surfaces:
                continue
            starts.append(s)
            ends.append(e)
            cand_lists.append(surfaces)
            golds.append(surfaces.index(surf))
        if not starts:
            return None
        if len(starts) > MAX_WORDS:
            sel = random.sample(range(len(starts)), MAX_WORDS)
            starts = [starts[i] for i in sorted(sel)]
            ends = [ends[i] for i in sorted(sel)]
            cand_lists = [cand_lists[i] for i in sorted(sel)]
            golds = [golds[i] for i in sorted(sel)]
        return {
            "ctx": self.vocab.encode(ctx),
            "read": self.vocab.encode(read),
            "starts": starts,
            "ends": ends,
            "cands": cand_lists,
            "gold": golds,
            "raw": (ctx, read, rec["words"]),
        }


class Collator:
    def __init__(self, vocab: CharVocab, surface_id: Dict[str, int], k: int):
        self.vocab = vocab
        self.surface_id = surface_id
        self.k = k

    def __call__(self, batch: List[Optional[dict]]) -> Optional[dict]:
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        B = len(batch)
        W = max(len(b["starts"]) for b in batch)
        L = max(3 + len(b["ctx"]) + len(b["read"]) for b in batch)
        K, S = self.k, 12

        ids = torch.zeros((B, L), dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        starts = torch.zeros((B, W), dtype=torch.long)
        ends = torch.zeros((B, W), dtype=torch.long)
        wmask = torch.zeros((B, W), dtype=torch.float)
        gold = torch.zeros((B, W), dtype=torch.long)
        cidx = torch.zeros((B, W, K), dtype=torch.long)
        cmask = torch.zeros((B, W, K), dtype=torch.float)

        for i, b in enumerate(batch):
            seq = [self.vocab.cls] + b["ctx"] + [self.vocab.sep] + b["read"] + [
                self.vocab.sep
            ]
            ids[i, : len(seq)] = torch.tensor(seq)
            am[i, : len(seq)] = 1
            off = 1 + len(b["ctx"]) + 1
            for w in range(len(b["starts"])):
                starts[i, w] = off + b["starts"][w]
                ends[i, w] = off + b["ends"][w]
                wmask[i, w] = 1.0
                gold[i, w] = b["gold"][w]
                for j, surf in enumerate(b["cands"][w][:K]):
                    cidx[i, w, j] = self.surface_id[surf]
                    cmask[i, w, j] = 1.0

        return {
            "input_ids": ids,
            "attention_mask": am,
            "starts": starts,
            "ends": ends,
            "wmask": wmask,
            "gold": gold,
            "cand_index": cidx,
            "cand_mask": cmask,
        }


def forward_batch(model, batch, table, device):
    tci, tcm, tlen = table[0], table[1], table[2]
    logits = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["starts"].to(device),
        batch["ends"].to(device),
        None,
        None,
        batch["cand_index"].to(device),
        tci,
        tcm,
        tlen,
    )
    cmask = batch["cand_mask"].to(device)
    logits = logits.masked_fill(cmask == 0, -1e4)
    gold = batch["gold"].to(device)
    wmask = batch["wmask"].to(device)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        gold.reshape(-1),
        reduction="none",
    ).view_as(wmask)
    loss = (loss * wmask).sum() / wmask.sum().clamp(min=1.0)
    with torch.no_grad():
        pred = logits.argmax(-1)
        correct = ((pred == gold).float() * wmask).sum()
    return loss, correct.item(), wmask.sum().item()


@torch.no_grad()
def evaluate(model, loader, table, device):
    model.eval()
    tot_c = tot_n = 0
    tot_loss = 0.0
    nb = 0
    for batch in loader:
        if batch is None:
            continue
        loss, c, n = forward_batch(model, batch, table, device)
        tot_c += c
        tot_n += n
        tot_loss += loss.item()
        nb += 1
    model.train()
    return tot_loss / max(nb, 1), tot_c / max(tot_n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="tohoku-nlp/bert-base-japanese-char-v3")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="models/scorer.pt")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lex = load_lexicon(os.path.join(args.data_dir, "lexicon.tsv"), args.k)
    print(f"lexicon: {len(lex.entries)} readings", flush=True)

    # 候補表記テーブル
    surfaces: List[str] = []
    surface_id: Dict[str, int] = {}
    for cands in lex.entries.values():
        for c in cands:
            if c.surface not in surface_id:
                surface_id[c.surface] = len(surfaces)
                surfaces.append(c.surface)
    print(f"candidate surfaces: {len(surfaces)}", flush=True)

    from transformers import AutoModel
    from huggingface_hub import hf_hub_download

    vocab_path = hf_hub_download(args.model, "vocab.txt")
    vocab = CharVocab(vocab_path)
    print(f"char vocab: {vocab.size}", flush=True)

    table = build_candidate_table(surfaces, vocab, device)

    encoder = AutoModel.from_pretrained(args.model)
    hidden = encoder.config.hidden_size
    model = SpanScorer(encoder, hidden).to(device)
    print(f"encoder: {args.model} hidden={hidden}", flush=True)

    train_ds = ConvDataset(
        os.path.join(args.data_dir, "train.jsonl"), lex, vocab, args.k
    )
    valid_ds = ConvDataset(
        os.path.join(args.data_dir, "valid.jsonl"), lex, vocab, args.k
    )
    if args.limit:
        train_ds.offsets = train_ds.offsets[: args.limit]
    print(f"train: {len(train_ds)} valid: {len(valid_ds)}", flush=True)

    coll = Collator(vocab, surface_id, args.k)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=coll,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=2,
        collate_fn=coll,
    )

    head_params = (
        list(model.proj.parameters()) + list(model.out.parameters())
    )
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": args.lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    total_steps = int(len(train_loader) * args.epochs) // args.grad_accum
    warmup = max(1, int(total_steps * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / (total_steps - warmup))),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    step = 0
    t0 = time.time()
    for epoch in range(math.ceil(args.epochs)):
        model.train()
        running = 0.0
        nb = 0
        for batch in train_loader:
            if batch is None:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = forward_batch(model, batch, table, device)
            (loss / args.grad_accum).backward()
            running += loss.item()
            nb += 1
            if nb % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if nb % 200 == 0:
                el = time.time() - t0
                print(
                    f"ep{epoch} {nb}/{len(train_loader)} step={step} "
                    f"loss={running / 200:.4f} lr={sched.get_last_lr()[0]:.2e} "
                    f"{el:.0f}s",
                    flush=True,
                )
                running = 0.0
        vl, va = evaluate(model, valid_loader, table, device)
        print(f"[epoch {epoch}] valid_loss={vl:.4f} cand_acc={va:.4f}", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": args.model,
                "hidden": hidden,
                "surfaces": surfaces,
                "k": args.k,
                "vocab_size": vocab.size,
            },
            args.out,
        )
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
