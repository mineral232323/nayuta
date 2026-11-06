#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — Android ビルドラッパ
#
# このマシン固有の前提をここに閉じ込める:
#   - JDK は $JAVA_HOME → /usr/lib/jvm/java-21-openjdk → .cache/jdk21 の順
#   - Android SDK はリポジトリ内の .android-sdk
#   - Gradle のキャッシュは $HOME ではなく .cache/gradle（$HOME 側は読み取り専用）
#
# 使い方:
#   bash scripts/android/build.sh                 # assembleDebug
#   bash scripts/android/build.sh clean assembleDebug
#   bash scripts/android/build.sh lintDebug
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANDROID_DIR="$REPO_ROOT/android"
SDK_DIR="$REPO_ROOT/.android-sdk"

# shellcheck source=scripts/android/env.sh
source "$REPO_ROOT/scripts/android/env.sh"
if ! JAVA_HOME="$(resolve_java_home)"; then
    echo "error: JDK 21 が見つかりません。先に scripts/android/setup_jdk.sh を実行してください。" >&2
    exit 1
fi

if [[ ! -d "$SDK_DIR/platforms/android-35" ]]; then
    echo "error: Android SDK が未導入です。先に scripts/android/setup_sdk.sh を実行してください。" >&2
    exit 1
fi

export JAVA_HOME
export ANDROID_HOME="$SDK_DIR"
export ANDROID_SDK_ROOT="$SDK_DIR"
export GRADLE_USER_HOME="$REPO_ROOT/.cache/gradle"
export PATH="$JAVA_HOME/bin:$PATH"
mkdir -p "$GRADLE_USER_HOME"

# ---------------------------------------------------------------------------
# 辞書アセット（APK に同梱する読み→表記辞書）は生成物なので git 管理外。
# 無ければ学習側のスクリプトで書き出す。
# ---------------------------------------------------------------------------
ASSET_DIR="$ANDROID_DIR/app/src/main/assets"
DICT_ASSET_GZ="$ASSET_DIR/dict/lexicon.tsv.gz"
DICT_ASSET_TSV="$ASSET_DIR/dict/lexicon.tsv"
if [[ ! -f "$DICT_ASSET_GZ" && ! -f "$DICT_ASSET_TSV" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" && -f "$REPO_ROOT/data/lexicon.tsv" ]]; then
        echo "==> 辞書アセットが無いので生成します"
        ( cd "$REPO_ROOT" && ./.venv/bin/python -m train.export_dict )
    else
        echo "error: $ASSET_DIR/dict/lexicon.tsv(.gz) がありません。" >&2
        echo "  配布 APK から取り出す場合:" >&2
        echo "    bash scripts/android/fetch_assets.sh" >&2
        echo "  学習から作る場合:" >&2
        echo "    python -m train.export_dict" >&2
        exit 1
    fi
fi

if [[ ! -f "$ASSET_DIR/model/scorer.onnx" ]]; then
    echo "warn: model/scorer.onnx がありません。BERT 再ランクなし（unigram のみ）でビルドします。" >&2
fi

if [[ $# -eq 0 ]]; then
    set -- assembleDebug
fi

cd "$ANDROID_DIR"
./gradlew "$@"
echo
# 実際に生成された APK だけを表示する（assembleRelease でも debug のパスを出していた）
for variant in debug release; do
    apk="$ANDROID_DIR/app/build/outputs/apk/$variant/app-$variant.apk"
    if [[ -f "$apk" ]]; then
        echo "APK: $apk"
    fi
done
