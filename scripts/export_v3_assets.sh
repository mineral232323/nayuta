#!/usr/bin/env bash
# v3 の学習済みアセットを端末用に書き出す。
#
#   辞書   data/lexicon_v3_frozen.tsv          -> assets/dict/lexicon.ndict（mmap 用パック形式）
#   bigram data/bigram_v3.tsv (+ words)        -> assets/dict/bigram.bin
#   モデル models/scorer_v3.pt                 -> assets/model/scorer.onnx (int8)
#
# 使い方:
#   bash scripts/export_v3_assets.sh
#   LEX=... BIG=... WORDS=... CKPT=... bash scripts/export_v3_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

LEX=${LEX:-data/lexicon_v3_frozen.tsv}
BIG=${BIG:-data/bigram_v3.tsv}
WORDS=${WORDS:-data/bigram_words_v3.tsv}
CKPT=${CKPT:-models/scorer_v3.pt}

echo "== 辞書 =="
.venv/bin/python -m train.export_dict --lexicon "$LEX" --topk 8

echo "== bigram =="
.venv/bin/python -m train.export_bigram --bigram "$BIG" --words "$WORDS"

echo "== モデル (int8 ONNX) =="
.venv/bin/python -m train.export_onnx_online --checkpoint "$CKPT"

echo "== 完了 =="
ls -la android/app/src/main/assets/dict/ android/app/src/main/assets/model/
