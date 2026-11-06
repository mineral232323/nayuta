#!/usr/bin/env bash
# 新アセット（v3 辞書 + v3 bigram + 新 ONNX）の評価を 4 通り回す。
#
#   bash scripts/eval_v3_assets.sh
#   bash scripts/eval_v3_assets.sh data/lexicon_v3_frozen.tsv data/bigram_v3.tsv \
#        data/bigram_words_v3.tsv android/app/src/main/assets/model/scorer.onnx
#
# 出力: logs/eval_v3_<name>.json / .log
#   各 JSON の assets.*.sha256 が「どのアセットで測ったか」の記録になる。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

PY=.venv/bin/python
LEXICON=${1:-data/lexicon_v3_frozen.tsv}
BIGRAM=${2:-data/bigram_v3.tsv}
BIGRAM_WORDS=${3:-data/bigram_words_v3.tsv}
ONNX=${4:-android/app/src/main/assets/model/scorer.onnx}

for f in "$LEXICON" "$BIGRAM" "$BIGRAM_WORDS" "$ONNX" \
         data/eval/wiki2000.jsonl data/eval/sns2000.jsonl; do
  if [ ! -f "$f" ]; then
    echo "missing: $f" >&2
    exit 1
  fi
done

echo "lexicon     : $LEXICON"
echo "bigram      : $BIGRAM ($BIGRAM_WORDS)"
echo "onnx        : $ONNX"
echo "git         : $(git rev-parse --short HEAD 2>/dev/null || echo '-' )"
echo

run() {  # name data domain beam want gen deltas betas
  local name=$1 data=$2 domain=$3 beam=$4 want=$5 gen=$6 deltas=$7 betas=$8
  echo "=== $name : data=$data beam=$beam want=$want gen=$gen deltas=$deltas betas=$betas"
  $PY -m train.evaluate_v2 \
    --data "$data" --domain "$domain" --limit 2000 \
    --lexicon "$LEXICON" --bigram "$BIGRAM" --bigram-words "$BIGRAM_WORDS" \
    --onnx "$ONNX" \
    --gen "$gen" --beam "$beam" --want "$want" \
    --deltas "$deltas" --betas "$betas" \
    --json-out "logs/eval_v3_${name}.json" \
    2>&1 | tee "logs/eval_v3_${name}.log" | tail -45
  echo
}

# (a) wiki2000: ベースラインと同一条件（beam 32/48, gen=bigram, d=2.0, β=1.0）
run wiki_base data/eval/wiki2000.jsonl wiki 32 48 bigram 2.0 1.0,2.0,4.0

# (b) wiki2000: 上限確認（beam 128/256, gen=bigram, d=2.0, β=1.0）
run wiki_beam128 data/eval/wiki2000.jsonl wiki 128 256 bigram 2.0 1.0,2.0

# (c) sns2000: ベースラインと同一条件
run sns_base data/eval/sns2000.jsonl sns 32 48 bigram 2.0 1.0,2.0,4.0

# (d) sns2000: 上限確認
run sns_beam128 data/eval/sns2000.jsonl sns 128 256 bigram 2.0 1.0,2.0

echo "done: logs/eval_v3_*.json"
