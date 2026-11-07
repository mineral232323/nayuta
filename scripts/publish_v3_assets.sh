#!/usr/bin/env bash
# v3 の辞書 / bigram を GitHub の assets リポジトリへ公開する。
#
#   bash scripts/publish_v3_assets.sh          # 既定タグ v3 へアップロード
#   bash scripts/publish_v3_assets.sh v3
#   NAYUTA_ASSETS_REPO=owner/repo bash scripts/publish_v3_assets.sh
#
# 事前に android/app/src/main/assets/dict/ に lexicon.ndict と bigram.bin を
# 用意しておく（export_dict / export_bigram で生成）。
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="${NAYUTA_ASSETS_REPO:-Mineral232323/nayuta-assets}"
TAG="${1:-v3}"
LEX=android/app/src/main/assets/dict/lexicon.ndict
BIG=android/app/src/main/assets/dict/bigram.bin

for f in "$LEX" "$BIG"; do
    if [[ ! -f "$f" ]]; then
        echo "error: $f がありません（先に辞書/bigram を書き出してください）" >&2
        exit 1
    fi
done

echo "== sha256 =="
sha256sum "$LEX" "$BIG"

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    gh release upload "$TAG" "$LEX" "$BIG" -R "$REPO" --clobber
else
    gh release create "$TAG" -R "$REPO" \
        --title "Nayuta v3 assets" \
        --notes "v3 辞書（50万読み）と 20万語 bigram。CI の fetch_assets.sh が参照する。" \
        "$LEX" "$BIG"
fi
echo "published: https://github.com/$REPO/releases/tag/$TAG"
