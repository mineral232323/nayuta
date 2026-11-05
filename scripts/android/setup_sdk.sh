#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — Android SDK セットアップ
#
# このマシンでは $HOME 側が読み取り専用なので、SDK はリポジトリ内の
# .android-sdk/ に入れる。何度実行しても壊れない（既にあるものは再取得しない）。
#
# 使い方:
#   bash scripts/android/setup_sdk.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDK_DIR="$REPO_ROOT/.android-sdk"
# shellcheck source=scripts/android/env.sh
source "$REPO_ROOT/scripts/android/env.sh"
CMDLINE_TOOLS_ZIP_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
CMDLINE_TOOLS_DIR="$SDK_DIR/cmdline-tools/latest"
SDKMANAGER="$CMDLINE_TOOLS_DIR/bin/sdkmanager"

# AGP / sdkmanager は JDK 17+ が必要。この環境では JDK 21 を使う。
if ! JAVA_HOME="$(resolve_java_home)"; then
    echo "error: JDK 21 が見つかりません。先に scripts/android/setup_jdk.sh を実行してください。" >&2
    exit 1
fi
export JAVA_HOME
export ANDROID_HOME="$SDK_DIR"
export ANDROID_SDK_ROOT="$SDK_DIR"
export PATH="$JAVA_HOME/bin:$PATH"

mkdir -p "$SDK_DIR"

# --- 1. cmdline-tools ------------------------------------------------------
if [[ ! -x "$SDKMANAGER" ]]; then
    echo "==> cmdline-tools を取得中..."
    tmp_zip="$(mktemp -d)/cmdline-tools.zip"
    curl -sS -L -o "$tmp_zip" "$CMDLINE_TOOLS_ZIP_URL"
    tmp_dir="$(mktemp -d)"
    unzip -q "$tmp_zip" -d "$tmp_dir"
    mkdir -p "$SDK_DIR/cmdline-tools"
    rm -rf "$CMDLINE_TOOLS_DIR"
    mv "$tmp_dir/cmdline-tools" "$CMDLINE_TOOLS_DIR"
else
    echo "==> cmdline-tools は導入済み: $CMDLINE_TOOLS_DIR"
fi

# --- 2. ライセンス同意 -----------------------------------------------------
echo "==> SDK ライセンスに同意..."
yes | "$SDKMANAGER" --sdk_root="$SDK_DIR" --licenses >/dev/null 2>&1 || true

# --- 3. パッケージ導入 -----------------------------------------------------
echo "==> platform-tools / platforms;android-35 / build-tools;35.0.0 を導入..."
"$SDKMANAGER" --sdk_root="$SDK_DIR" \
    "platform-tools" "platforms;android-35" "build-tools;35.0.0"

# --- 4. local.properties ---------------------------------------------------
cat > "$REPO_ROOT/android/local.properties" <<EOF
# 自動生成: scripts/android/setup_sdk.sh（git 管理外）
sdk.dir=$SDK_DIR
EOF
echo "==> android/local.properties に sdk.dir=$SDK_DIR を書きました"

echo "==> 完了"
"$SDKMANAGER" --sdk_root="$SDK_DIR" --list_installed
