#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — Android ビルドラッパ
#
# このマシン固有の前提をここに閉じ込める:
#   - JDK は /usr/lib/jvm/java-21-openjdk
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

JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk}"
if [[ ! -x "$JAVA_HOME/bin/java" ]]; then
    echo "error: JDK が見つかりません: $JAVA_HOME" >&2
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

if [[ $# -eq 0 ]]; then
    set -- assembleDebug
fi

cd "$ANDROID_DIR"
./gradlew "$@"
echo
echo "APK: $ANDROID_DIR/app/build/outputs/apk/debug/app-debug.apk"
