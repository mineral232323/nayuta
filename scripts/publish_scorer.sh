#!/usr/bin/env bash
# 学習済み int8 scorer を GitHub assets リリースへ公開する。
#
#   bash scripts/publish_scorer.sh [onnx] [tag]
#   # 既定: logs/scorer_rerank_mix70k_int8.onnx を model-v3-rerank へ
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-logs/scorer_rerank_mix70k_int8.onnx}"
TAG="${2:-model-v3-rerank}"
REPO="${NAYUTA_SCORER_REPO:-Mineral232323/nayuta-assets}"

if [[ ! -f "$SRC" ]]; then
    echo "error: $SRC がありません" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$SRC" "$tmp/scorer.onnx"
echo "== sha256 =="
sha256sum "$tmp/scorer.onnx"

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    gh release upload "$TAG" "$tmp/scorer.onnx" -R "$REPO" --clobber
else
    gh release create "$TAG" -R "$REPO" \
        --title "Nayuta rerank scorer" \
        --notes "listwise 再ランク学習済み int8 ONNX。CI の fetch_assets.sh が参照する。" \
        "$tmp/scorer.onnx"
fi
echo "published: https://github.com/$REPO/releases/tag/$TAG"
