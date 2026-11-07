#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — 学習済みアセットを配布 APK から取り出す
#
# モデルと辞書（合計 ~27MB）はリポジトリに入っていない。学習から作り直すには
# Wikipedia ダンプと GPU が必要なので、開発環境のブートストラップ用に
# リリース済み APK（署名済み release / 旧 debug のどちらでも可）の中の assets/ を展開する。
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

fetch_url() {  # fetch_url <out> <url> — curl 失敗時は python urllib でフォールバック
    local out=$1 url=$2
    if curl -sS -L --fail -o "$out" "$url"; then
        return 0
    fi
    rm -f "$out"
    python3 - "$out" "$url" <<'PYEOF'
import sys
import urllib.request

urllib.request.urlretrieve(sys.argv[2], sys.argv[1])
PYEOF
}

hash_file() {  # sha256（sha256sum 無しの環境では shasum）
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

# origin から owner/repo を取る（fork でも動くように）
REMOTE="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
SLUG="$(printf '%s' "$REMOTE" | sed -E 's#.*github\.com[/:]##; s#\.git$##')"
if [[ -z "$SLUG" ]]; then
    echo "error: git remote origin から GitHub リポジトリを特定できません" >&2
    exit 1
fi

# APK 名の候補（署名済み release を優先し、debug 署名時代の配布物も後方互換で試す）
#   v0.2.6-pre → nayuta-0.2.6-release.apk / nayuta-v0.2.6-release.apk
#              → nayuta-0.2.6-debug.apk   / nayuta-v0.2.6-debug.apk（旧形式）
BASE_NEW="${TAG#v}"; BASE_NEW="${BASE_NEW%-pre}"
BASE_OLD="${TAG%-pre}"

mkdir -p "$CACHE_DIR"
APK_NAME=""
for flavor in release debug; do
    for base in "$BASE_NEW" "$BASE_OLD"; do
        cand="nayuta-${base}-${flavor}.apk"
        APK="$CACHE_DIR/$cand"
        if [[ -f "$APK" ]]; then
            APK_NAME="$cand"
            echo "==> APK は取得済み: $APK"
            break 2
        fi
        URL="https://github.com/$SLUG/releases/download/$TAG/$cand"
        echo "==> APK を取得中: $URL"
        if fetch_url "$APK" "$URL"; then
            APK_NAME="$cand"
            break 2
        fi
        rm -f "$APK"
    done
done
if [[ -z "$APK_NAME" ]]; then
    echo "error: タグ $TAG の APK が見つかりません（新旧両形式を試しました）" >&2
    exit 1
fi
APK="$CACHE_DIR/$APK_NAME"

echo "==> assets/dict と assets/model を展開中..."
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
unzip -q "$APK" 'assets/*' -d "$tmp_dir"
if [[ ! -d "$tmp_dir/assets/dict" && ! -d "$tmp_dir/assets/model" ]]; then
    echo "error: APK に assets/dict も assets/model もありません" >&2
    exit 1
fi

# AGP が生成する assets/dexopt/ などは取り込まない（release ビルドで重複するため）
mkdir -p "$ASSET_DIR"
for dir in dict model; do
    if [[ -d "$tmp_dir/assets/$dir" ]]; then
        mkdir -p "$ASSET_DIR/$dir"
        cp -r "$tmp_dir/assets/$dir/." "$ASSET_DIR/$dir/"
    fi
done

# ---------------------------------------------------------------------------
# 再ランク済みスコアラー overlay（GitHub: Mineral232323/nayuta-assets）
#
# 文頭 prior-wrong の hard negative で listwise 再ランク学習した int8 ONNX を
# 取得して model/scorer.onnx を差し替える。sha256 が一致しなければ失敗する。
#
#   無効化:   NAYUTA_V3_SCORER=0
#   差し替え: NAYUTA_SCORER_REPO / NAYUTA_SCORER_TAG / NAYUTA_SCORER_SHA256
# ---------------------------------------------------------------------------
if [[ "${NAYUTA_V3_SCORER:-1}" == 1 ]]; then
    scorer_repo="${NAYUTA_SCORER_REPO:-Mineral232323/nayuta-assets}"
    scorer_tag="${NAYUTA_SCORER_TAG:-model-v3-rerank}"
    scorer_url="https://github.com/$scorer_repo/releases/download/$scorer_tag/scorer.onnx"
    expected_sha="${NAYUTA_SCORER_SHA256:-40339a527b9faacf59199d7b873be9412bf43018abf3e875985e44a9ddb19dec}"

    echo "==> 再ランク済み scorer を取得中: $scorer_url"
    mkdir -p "$ASSET_DIR/model"
    if ! fetch_url "$ASSET_DIR/model/scorer.onnx" "$scorer_url"; then
        echo "error: scorer の取得に失敗しました" >&2
        exit 1
    fi
    actual_sha="$(hash_file "$ASSET_DIR/model/scorer.onnx")"
    if [[ -n "$expected_sha" && "$actual_sha" != "$expected_sha" ]]; then
        echo "error: scorer.onnx の sha256 が一致しません" >&2
        echo "       expected: $expected_sha" >&2
        echo "       got:      $actual_sha" >&2
        exit 1
    fi
    echo "    scorer.onnx sha256=$actual_sha"
fi

# ---------------------------------------------------------------------------
# v3 辞書 / bigram overlay（GitHub: Mineral232323/nayuta-assets の v3 リリース）
#
# 前回リリースの APK に入っている dict/ は旧世代（22万語辞書 / 10万語 bigram）。
# v3 の「50万読み辞書（mmap 用 lexicon.ndict）」と「20万語 bigram（bigram.bin）」を
# 取得して上書きする。sha256 が一致しなければ失敗する。
#
#   無効化:   NAYUTA_V3_ASSETS=0
#   差し替え: NAYUTA_ASSETS_REPO / NAYUTA_ASSETS_TAG
#             NAYUTA_LEXICON_SHA256 / NAYUTA_BIGRAM_SHA256
# ---------------------------------------------------------------------------
if [[ "${NAYUTA_V3_ASSETS:-1}" == 1 ]]; then
    assets_repo="${NAYUTA_ASSETS_REPO:-Mineral232323/nayuta-assets}"
    assets_tag="${NAYUTA_ASSETS_TAG:-v3}"
    assets_base="https://github.com/$assets_repo/releases/download/$assets_tag"
    mkdir -p "$ASSET_DIR/dict"
    fetch_v3_asset() {  # name expected_sha
        local name=$1 sha=$2 url="$assets_base/$1"
        echo "==> v3 アセットを取得中: $url"
        if ! fetch_url "$ASSET_DIR/dict/$name" "$url"; then
            echo "error: v3 アセットの取得に失敗しました: $name" >&2
            return 1
        fi
        local actual
        actual="$(hash_file "$ASSET_DIR/dict/$name")"
        if [[ -n "$sha" && "$actual" != "$sha" ]]; then
            echo "error: $name の sha256 が一致しません" >&2
            echo "       expected: $sha" >&2
            echo "       got:      $actual" >&2
            return 1
        fi
        echo "    $name sha256=$actual"
    }
    fetch_v3_asset lexicon.ndict "${NAYUTA_LEXICON_SHA256:-1a5225dd22e5cca9be15dceb97a222c3b61818096aec3dcb0a1db4dd50cc96d1}" || exit 1
    fetch_v3_asset bigram.bin "${NAYUTA_BIGRAM_SHA256:-1daa24e9dd430a54d254226813ad95581f6b951c5a1a6d253472ff5cc40dccc9}" || exit 1
    # 旧形式の辞書が残っていると APK に二重同梱されるので消す
    rm -f "$ASSET_DIR/dict/lexicon.tsv" "$ASSET_DIR/dict/lexicon.tsv.gz"
fi

# APK から展開したアセットは aapt が .gz を解凍した形（english.tsv など）になる。
# 元の .gz が残っているとリソース重複でビルドが失敗するので、解凍済みがあれば消す。
if [[ -f "$ASSET_DIR/dict/english.tsv" ]]; then
    rm -f "$ASSET_DIR/dict/english.tsv.gz"
fi

echo "==> 展開しました: $ASSET_DIR"
find "$ASSET_DIR/dict" "$ASSET_DIR/model" -type f -printf '    %P\t%s bytes\n' 2>/dev/null | sort
