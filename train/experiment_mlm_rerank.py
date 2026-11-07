"""事前学習 char BERT の MLM 疑似尤度で N-best を再ランクする実験（P2）。

現行パイプライン（辞書 prior + bigram + 自作スパン BERT）で上位 K 仮説を作り、
その**表記列**の MLM 疑似尤度を事前学習 char BERT（tohoku-nlp/bert-base-japanese-char-v3）
で計算して、γ の重みで再ランクする。

  # する接続の評価セット（train.make_suru_eval で生成）で試す
  python -m train.experiment_mlm_rerank --data data/eval/suru_amb.jsonl --limit 300

  # Wikipedia でも同じ条件で
  python -m train.experiment_mlm_rerank --data data/eval/wiki2000.jsonl --limit 200 \
      --json-out logs/experiment_mlm_wiki.json

モデルは HF キャッシュから読む（`--model` でパス指定も可）。
文脈を入れると遅くなるので既定は文脈なし。`--ctx` で有効化する。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.cli import Engine  # noqa: E402
from train.evaluate_v2 import gold_info  # noqa: E402
from train.scorer import CharVocab  # noqa: E402

MODEL_ID = "tohoku-nlp/bert-base-japanese-char-v3"


def find_model_dir(model_id: str, override: str) -> str:
    if override:
        return override
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    base = os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots")
    found = []
    for snap in glob.glob(os.path.join(base, "*")):
        if not os.path.exists(os.path.join(snap, "config.json")):
            continue
        if any(
            os.path.exists(os.path.join(snap, f))
            for f in ("pytorch_model.bin", "model.safetensors")
        ):
            found.append(snap)
    if not found:
        raise SystemExit(
            f"事前学習モデルが見つかりません: {model_id}\n"
            "  huggingface-cli download で取得するか --model でパスを指定してください"
        )
    return found[0]


class MLMScorer:
    """char-level BERT の MLM 疑似尤度（1 文字ずつマスクして logP を足す）。"""

    def __init__(
        self,
        model_dir: str,
        vocab_path: str,
        threads: int = 8,
        max_ctx: int = 32,
        chunk: int = 64,
    ):
        from transformers import AutoConfig, AutoModelForMaskedLM

        torch.set_num_threads(threads)
        self.vocab = CharVocab(vocab_path)
        self.mask_id = self.vocab.chars["[MASK]"]
        cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_dir, config=cfg, local_files_only=True
        ).eval()
        self.max_ctx = max_ctx
        self.chunk = chunk
        self.n_tokens = 0

    @torch.no_grad()
    def pll_many(
        self, texts: Sequence[str], ctx: str = ""
    ) -> List[Tuple[float, int]]:
        """各 text の合計 logP と文字数を返す（全 text をまとめてバッチ化）。"""
        cids = self.vocab.encode(ctx[-self.max_ctx:]) if ctx else []
        base = [self.vocab.cls] + cids + [self.vocab.sep] if cids else [self.vocab.cls]
        off = len(base)
        rows: List[List[int]] = []
        meta: List[Tuple[int, int, int]] = []  # (text index, char index, char id)
        for ti, text in enumerate(texts):
            ids = self.vocab.encode(text)
            if not ids:
                continue
            seq = base + ids + [self.vocab.sep]
            for i, cid in enumerate(ids):
                row = list(seq)
                row[off + i] = self.mask_id
                rows.append(row)
                meta.append((ti, i, cid))

        total = [0.0] * len(texts)
        count = [0] * len(texts)
        if not rows:
            return [(0.0, 0)] * len(texts)
        L = off + max(len(self.vocab.encode(t)) for t in texts) + 1
        for s0 in range(0, len(rows), self.chunk):
            part = rows[s0 : s0 + self.chunk]
            X = torch.full((len(part), L), self.vocab.pad, dtype=torch.long)
            for j, row in enumerate(part):
                X[j, : len(row)] = torch.tensor(row, dtype=torch.long)
            att = (X != self.vocab.pad).long()
            logits = self.model(input_ids=X, attention_mask=att).logits
            self.n_tokens += X.numel()
            for j, (ti, i, cid) in enumerate(meta[s0 : s0 + self.chunk]):
                lp = torch.log_softmax(logits[j, off + i], dim=-1)
                total[ti] += float(lp[cid])
                count[ti] += 1
        return list(zip(total, count))


def load_rows(path: str, limit: int, skip: int = 0) -> List[dict]:
    rows = []
    seen = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen += 1
            if seen <= skip:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def build_engine(threads: int) -> Engine:
    """train.cli と同じ v3 アセット・wiki プリセットで Engine を作る。"""
    return Engine(
        lexicon="data/lexicon_v3_frozen.tsv",
        bigram="data/bigram_v3.tsv",
        bigram_words="data/bigram_words_v3.tsv",
        onnx="android/app/src/main/assets/model/scorer.onnx",
        vocab="android/app/src/main/assets/model/vocab.txt",
        gen="bigram",
        delta=1.0,
        beta=0.5,
        beam=32,
        want=48,
        threads=4,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MLM 疑似尤度による N-best 再ランク実験")
    ap.add_argument("--data", default="data/eval/suru_amb.jsonl")
    ap.add_argument("--limit", type=int, default=300, help="0 で全件")
    ap.add_argument("--skip", type=int, default=0, help="先頭 N 文を読み飛ばす（held-out 用）")
    ap.add_argument("--top-k", type=int, default=6, help="再ランクに使う候補数")
    ap.add_argument("--gammas", default="0.02,0.05,0.1,0.2,0.4")
    ap.add_argument("--ctx", action="store_true", help="文脈も条件に含める（遅い）")
    ap.add_argument("--model", default="", help="事前学習モデルのディレクトリ")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    model_dir = find_model_dir(MODEL_ID, args.model)
    vocab_path = os.path.join(model_dir, "vocab.txt")
    if not os.path.exists(vocab_path):
        vocab_path = "android/app/src/main/assets/model/vocab.txt"

    print(f"model: {model_dir}", flush=True)
    engine = build_engine(args.threads)
    mlm = MLMScorer(model_dir, vocab_path, threads=args.threads)
    print(f"engine/MLM ready ({engine.load_ms:.0f}ms)", flush=True)

    rows = load_rows(args.data, args.limit, args.skip)
    n = base_ok = cover_ok = 0
    ranks: List[int] = []
    acc: Dict[float, int] = {g: 0 for g in gammas}
    fixed: Dict[float, int] = {g: 0 for g in gammas}
    broken: Dict[float, int] = {g: 0 for g in gammas}
    t0 = time.time()

    for rec in rows:
        read, gold, _b = gold_info(rec)
        if not read:
            continue
        ctx = rec.get("ctx", "") if args.ctx else ""
        results = engine.convert(read, ctx, top=args.top_k)
        if not results:
            continue
        n += 1
        surfaces = [r[1] for r in results]
        costs = [r[0] for r in results]
        plls = mlm.pll_many(surfaces, ctx)
        if gold in surfaces:
            cover_ok += 1
            ranks.append(surfaces.index(gold))
        base_pred = surfaces[0]
        if base_pred == gold:
            base_ok += 1
        for g in gammas:
            best = min(
                range(len(surfaces)),
                key=lambda i: costs[i] - g * plls[i][0],
            )
            pred = surfaces[best]
            if pred == gold:
                acc[g] += 1
                if base_pred != gold:
                    fixed[g] += 1
            elif base_pred == gold:
                broken[g] += 1
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    n = max(n, 1)
    print()
    print(f"data    : {args.data}  n={n}  top_k={args.top_k}  ctx={args.ctx}")
    print(f"baseline: {base_ok / n:.4f}  ({base_ok}/{n})")
    print(f"gold in top-{args.top_k}: {cover_ok / n:.4f}")
    if ranks:
        buckets = {
            "rank 0": sum(1 for r in ranks if r == 0),
            "rank 1-3": sum(1 for r in ranks if 1 <= r <= 3),
            "rank 4-7": sum(1 for r in ranks if 4 <= r <= 7),
            "rank 8+": sum(1 for r in ranks if r >= 8),
        }
        print("gold rank in candidates: " + "  ".join(f"{k}={v}" for k, v in buckets.items()))
    print(f"{'gamma':>8} {'acc':>8} {'fixed':>6} {'broken':>7}")
    for g in gammas:
        print(f"{g:>8.3f} {acc[g] / n:>8.4f} {fixed[g]:>6} {broken[g]:>7}")

    print("\n[case] きしゃする")
    case = engine.convert("きしゃする", "", top=max(args.top_k, 5))
    plls = mlm.pll_many([c[1] for c in case], "")
    for (cost, surface, _rows), (pll, nc) in zip(case, plls):
        print(
            f"  cost={cost:7.2f} pll={pll:8.2f} ({pll / max(nc, 1):6.2f}/char)  {surface}"
        )

    if args.json_out:
        payload = {
            "data": args.data,
            "n": n,
            "top_k": args.top_k,
            "ctx": args.ctx,
            "model": model_dir,
            "baseline": base_ok / n,
            "coverage": cover_ok / n,
            "gammas": {
                str(g): {"acc": acc[g] / n, "fixed": fixed[g], "broken": broken[g]}
                for g in gammas
            },
            "tokens": mlm.n_tokens,
            "seconds": time.time() - t0,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
