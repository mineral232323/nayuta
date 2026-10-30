"""学習済み採点器を ONNX に書き出し、動的量子化（int8）する。

  python -m train.export_onnx --checkpoint models/small_v3.pt

出力:
  android/app/src/main/assets/model/scorer.onnx       （int8, 端末用）
  models/scorer_fp32.onnx                             （検証用）
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.scorer import InferenceScorer, SpanScorer, build_encoder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/small_v3.pt")
    ap.add_argument("--fp32-out", default="models/scorer_fp32.onnx")
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
    net = InferenceScorer(scorer).eval()

    S = 12
    dummy = (
        torch.randint(0, vocab_size, (1, 64), dtype=torch.long),
        torch.ones((1, 64), dtype=torch.long),
        torch.tensor([[10, 20]], dtype=torch.long),
        torch.tensor([[12, 23]], dtype=torch.long),
        torch.randint(0, vocab_size, (1, 2, k, S), dtype=torch.long),
        torch.ones((1, 2, k, S), dtype=torch.long),
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
    }
    os.makedirs(os.path.dirname(args.fp32_out) or ".", exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        args.fp32_out,
        input_names=[
            "input_ids",
            "attention_mask",
            "starts",
            "ends",
            "cand_ids",
            "cand_mask",
        ],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"fp32 onnx: {args.fp32_out} ({os.path.getsize(args.fp32_out)/1e6:.1f}MB)")

    # 数値一致の確認
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(args.fp32_out, providers=["CPUExecutionProvider"])
    got = sess.run(
        None,
        {
            "input_ids": dummy[0].numpy(),
            "attention_mask": dummy[1].numpy(),
            "starts": dummy[2].numpy(),
            "ends": dummy[3].numpy(),
            "cand_ids": dummy[4].numpy(),
            "cand_mask": dummy[5].numpy(),
        },
    )[0]
    diff = np.abs(got - ref.numpy()).max()
    print(f"max |onnx - torch| = {diff:.2e}")

    # 動的量子化（重み int8 / 活性は実行時量子化）
    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quantize_dynamic(
        args.fp32_out,
        args.out,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    print(f"int8 onnx: {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got8 = sess.run(
        None,
        {
            "input_ids": dummy[0].numpy(),
            "attention_mask": dummy[1].numpy(),
            "starts": dummy[2].numpy(),
            "ends": dummy[3].numpy(),
            "cand_ids": dummy[4].numpy(),
            "cand_mask": dummy[5].numpy(),
        },
    )[0]
    print(f"max |int8 - fp32| = {np.abs(got8 - got).max():.2e}")

    # 実データでのレイテンシ（PC CPU）
    import time

    seq = np.random.randint(0, vocab_size, (1, 110)).astype(np.int64)
    am = np.ones((1, 110), dtype=np.int64)
    st = np.tile(np.arange(5, 105, 10)[:10], (1, 1)).astype(np.int64)
    en = (st + 8).astype(np.int64)
    ci = np.random.randint(0, vocab_size, (1, 10, k, S)).astype(np.int64)
    cm = np.ones((1, 10, k, S), dtype=np.int64)
    for name, s in (("fp32", sess_fp32 := ort.InferenceSession(args.fp32_out, providers=["CPUExecutionProvider"])), ("int8", sess)):
        for _ in range(3):
            s.run(None, {"input_ids": seq, "attention_mask": am, "starts": st, "ends": en, "cand_ids": ci, "cand_mask": cm})
        t0 = time.time()
        for _ in range(20):
            s.run(None, {"input_ids": seq, "attention_mask": am, "starts": st, "ends": en, "cand_ids": ci, "cand_mask": cm})
        print(f"{name}: {(time.time()-t0)/20*1000:.1f} ms / 110 chars, 10 spans")


if __name__ == "__main__":
    main()
