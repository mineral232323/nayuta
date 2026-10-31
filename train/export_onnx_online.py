"""オンライン学習用に、BERT 採点器を再エクスポートする。

通常のスコアに加えて、端末で学習する「薄い線形層」のための出力を持つ:

  inputs : input_ids / attention_mask / starts / ends / cand_ids / cand_mask / w
  outputs: logits  (B,W,K)       … 元のスコア
           extra   (B,W,K)       … Σ_d w_d · h_d · c_d （学習した重みによる補正）
           phi     (B,W,K,H)     … h ⊙ c （w を更新するための勾配特徴）

端末は
  - 通常時: logits と extra だけを要求（phi は ONNX ランタイムが刈り取る）
  - 確定時: phi だけを要求（W=1 の小さい forward）
という使い分けをする。

  python -m train.export_onnx_online --checkpoint models/small_v3.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.scorer import SpanScorer, build_encoder  # noqa: E402


class OnlineScorer(nn.Module):
    def __init__(self, scorer: SpanScorer):
        super().__init__()
        self.encoder = scorer.encoder
        self.emb = scorer.emb
        self.proj = scorer.proj
        self.out = scorer.out

    def forward(self, input_ids, attention_mask, starts, ends, cand_ids, cand_mask, w):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        h_span = SpanScorer.pool_spans(hidden, starts, ends)  # (B,W,H)
        B, W, H = h_span.shape
        K = cand_ids.shape[2]
        ce = self.emb(cand_ids)  # (B,W,K,S,H)
        m = cand_mask.unsqueeze(-1).to(ce.dtype)
        c = (ce * m).sum(3) / m.sum(3).clamp(min=1.0)  # (B,W,K,H)

        hs = h_span.unsqueeze(2)  # (B,W,1,H)
        hs_k = hs.expand(-1, -1, K, -1)
        feats = torch.cat([hs_k, c, hs_k * c], dim=-1)
        logits = self.out(self.proj(feats)).squeeze(-1)  # (B,W,K)

        # オンライン学習用の特徴。スケールを揃えるため単位ベクトルにしてから積を取る。
        h_n = h_span / h_span.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        c_n = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # √H を掛けて、φ のノルムがおよそ 1 になるようにする
        # （学習率とスコアのスケールを揃えるため）
        phi = (h_n.unsqueeze(2) * c_n) * (H ** 0.5)      # (B,W,K,H)
        extra = (phi * w.view(1, 1, 1, H)).sum(-1)       # (B,W,K)
        return logits, extra, phi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/small_v3.pt")
    ap.add_argument("--fp32-out", default="models/scorer_online_fp32.onnx")
    ap.add_argument("--out", default="android/app/src/main/assets/model/scorer.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "pretrained")
    vocab_size = ck.get("vocab_size", 7027)
    hidden = ck["hidden"]
    k = ck.get("k", 8)

    encoder = build_encoder(arch, vocab_size, ck.get("model_name", ""))
    scorer = SpanScorer(encoder, hidden)
    scorer.load_state_dict(ck["model"], strict=False)
    scorer.eval()
    net = OnlineScorer(scorer).eval()

    S = 12
    dummy = (
        torch.randint(0, vocab_size, (1, 64), dtype=torch.long),
        torch.ones((1, 64), dtype=torch.long),
        torch.tensor([[10, 20]], dtype=torch.long),
        torch.tensor([[12, 23]], dtype=torch.long),
        torch.randint(0, vocab_size, (1, 2, k, S), dtype=torch.long),
        torch.ones((1, 2, k, S), dtype=torch.long),
        torch.zeros((1, hidden), dtype=torch.float),
    )

    with torch.no_grad():
        ref = net(*dummy)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "starts": {0: "batch", 1: "spans"},
        "ends": {0: "batch", 1: "spans"},
        "cand_ids": {0: "batch", 1: "spans"},
        "cand_mask": {0: "batch", 1: "spans"},
        "logits": {0: "batch", 1: "spans"},
        "extra": {0: "batch", 1: "spans"},
        "phi": {0: "batch", 1: "spans"},
    }
    os.makedirs(os.path.dirname(args.fp32_out) or ".", exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        args.fp32_out,
        input_names=[
            "input_ids", "attention_mask", "starts", "ends", "cand_ids", "cand_mask", "w",
        ],
        output_names=["logits", "extra", "phi"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"fp32: {args.fp32_out} ({os.path.getsize(args.fp32_out)/1e6:.1f}MB)")

    import numpy as np
    import onnxruntime as ort

    feeds = {
        "input_ids": dummy[0].numpy(),
        "attention_mask": dummy[1].numpy(),
        "starts": dummy[2].numpy(),
        "ends": dummy[3].numpy(),
        "cand_ids": dummy[4].numpy(),
        "cand_mask": dummy[5].numpy(),
        "w": dummy[6].numpy(),
    }
    sess = ort.InferenceSession(args.fp32_out, providers=["CPUExecutionProvider"])
    got = sess.run(["logits", "extra", "phi"], feeds)
    print(f"max|logits-torch|={np.abs(got[0]-ref[0].numpy()).max():.2e}")
    print(f"max|extra-torch|={np.abs(got[1]-ref[1].numpy()).max():.2e}")
    print(f"max|phi-torch|={np.abs(got[2]-ref[2].numpy()).max():.2e}")

    # 特徴が実際に効くか（w を変えると extra が変わる）
    import copy

    feeds2 = copy.deepcopy(feeds)
    rng = np.random.default_rng(0)
    feeds2["w"] = rng.standard_normal((1, hidden)).astype(np.float32)
    got2 = sess.run(["extra"], feeds2)[0]
    print(f"w を乱数にしたときの extra の変化: max={np.abs(got2-got[1]).max():.3f}")

    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quantize_dynamic(args.fp32_out, args.out, weight_type=QuantType.QInt8)
    print(f"int8: {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")

    sess8 = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got8 = sess8.run(["logits", "extra"], feeds)
    print(f"quantized max|logits-fp32|={np.abs(got8[0]-got[0]).max():.2e}")


if __name__ == "__main__":
    main()
