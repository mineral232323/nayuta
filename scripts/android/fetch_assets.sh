#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — 学習済みアセットを配布 APK から取り出す
#
# モデルと辞書（合計 ~27MB）はリポジトリに入っていない。学習から作り直すには
# Wikipedia ダンプと GPU が必要なので、開発環境のブートストラップ用に
# リリース済み debug APK の中の assets/ を展開する。
#
# 使い方:
#   bash scripts/android/fetch_assets.sh
#   NAYUTA_RELEASE_TAG=v0.2.4-pre bash scripts/android/fetch_assets.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSET_DIR="$REPO_ROOT/android/app/src/main/assets"
TAG="${NAYUTA_RELEASE_TAG:-v0.2.5-pre}"
CACHE_DIR="$REPO_ROOT/.cache/release"

# origin から owner/repo を取る（fork でも動くように）
REMOTE="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
SLUG="$(printf '%s' "$REMOTE" | sed -E 's#.*github\.com[/:]##; s#\.git$##')"
if [[ -z "$SLUG" ]]; then
    echo "error: git remote origin から GitHub リポジトリを特定できません" >&2
    exit 1
fi

APK_NAME="nayuta-${TAG#v}-debug.apk"
# タグ名と APK 名が一致しないリリース（v0.2.5-pre → nayuta-v0.2.5-debug.apk）を吸収
if [[ "$TAG" == *-pre ]]; then
    APK_NAME="nayuta-${TAG%-pre}-debug.apk"
fi
URL="https://github.com/$SLUG/releases/download/$TAG/$APK_NAME"

mkdir -p "$CACHE_DIR"
APK="$CACHE_DIR/$APK_NAME"
if [[ ! -f "$APK" ]]; then
    echo "==> APK を取得中: $URL"
    curl -sS -L --fail -o "$APK" "$URL"
else
    echo "==> APK は取得済み: $APK"
fi

echo "==> assets/ を展開中..."
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
unzip -q "$APK" 'assets/*' -d "$tmp_dir"
if [[ ! -d "$tmp_dir/assets" ]]; then
    echo "error: APK に assets/ がありません" >&2
    exit 1
fi
mkdir -p "$ASSET_DIR"
cp -r "$tmp_dir/assets/." "$ASSET_DIR/"

echo "==> 展開しました: $ASSET_DIR"
find "$ASSET_DIR" -type f -printf '    %P\t%s bytes\n' | sort
