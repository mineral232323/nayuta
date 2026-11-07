"""fp32 ONNX を端末出荷用の動的 int8 (QInt8) に量子化する。

train.export_onnx_online の量子化ステップと同一処理。checkpoint が手元に無くて
fp32 ONNX だけある場合（HF で公開済みの重みなど）でも torch 無しで int8 を作れる。

  python -m train.quantize_onnx --fp32 models/scorer_online_fp32.onnx \
      --out android/app/src/main/assets/model/scorer.onnx
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    quantize_dynamic(args.fp32, args.out, weight_type=QuantType.QInt8)

    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    outputs = [o.name for o in sess.get_outputs()]
    fp_mb = os.path.getsize(args.fp32) / 1e6
    int8_mb = os.path.getsize(args.out) / 1e6
    print(f"int8: {args.out} ({int8_mb:.1f}MB, fp32 {fp_mb:.1f}MB) outputs={outputs}")


if __name__ == "__main__":
    main()
