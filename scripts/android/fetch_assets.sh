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

# APK 名の候補（新形式を優先し、旧形式 v付き も試す）
#   v0.2.5-pre → nayuta-0.2.5-release.apk（新・署名付き） / nayuta-v0.2.5-release.apk
#              → nayuta-0.2.5-debug.apk / nayuta-v0.2.5-debug.apk（旧・debug 署名）
BASE_NEW="${TAG#v}"; BASE_NEW="${BASE_NEW%-pre}"
BASE_OLD="${TAG%-pre}"

mkdir -p "$CACHE_DIR"
APK_NAME=""
for cand in "nayuta-${BASE_NEW}-release.apk" "nayuta-${BASE_OLD}-release.apk" \
            "nayuta-${BASE_NEW}-debug.apk" "nayuta-${BASE_OLD}-debug.apk"; do
    APK="$CACHE_DIR/$cand"
    if [[ -f "$APK" ]]; then
        APK_NAME="$cand"
        echo "==> APK は取得済み: $APK"
        break
    fi
    URL="https://github.com/$SLUG/releases/download/$TAG/$cand"
    echo "==> APK を取得中: $URL"
    if curl -sS -L --fail -o "$APK" "$URL"; then
        APK_NAME="$cand"
        break
    fi
    rm -f "$APK"
done
if [[ -z "$APK_NAME" ]]; then
    echo "error: タグ $TAG の APK が見つかりません（release / debug 両形式を試しました）" >&2
    exit 1
fi
APK="$CACHE_DIR/$APK_NAME"

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
