#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — リポジトリ内 JDK 21 セットアップ
#
# sudo が使えない環境でも Android のビルドができるよう、Temurin の JDK 21 を
# .cache/jdk21 に展開する（.cache/ は git 管理外）。
#
# 使い方:
#   bash scripts/android/setup_jdk.sh
#   eval "$(bash scripts/android/setup_jdk.sh --print-env)"   # 現在の shell に反映
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/.cache/jdk21"
PRINT_ENV=0
if [[ "${1:-}" == "--print-env" ]]; then
    PRINT_ENV=1
fi

ADOPTIUM_URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"

java_major() {
    "$1/bin/java" -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p' | head -1
}

if [[ -x "$DEST/bin/java" ]]; then
    major="$(java_major "$DEST" || true)"
    if [[ "$major" == "21" ]]; then
        echo "==> JDK は導入済み: $DEST" >&2
    else
        echo "==> 既存の JDK が 21 ではない（$major）ので入れ直します: $DEST" >&2
        rm -rf "$DEST"
    fi
fi

if [[ ! -x "$DEST/bin/java" ]]; then
    echo "==> Temurin JDK 21 を取得中..." >&2
    mkdir -p "$REPO_ROOT/.cache"
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    curl -sS -L --fail -o "$tmp_dir/jdk.tar.gz" "$ADOPTIUM_URL"
    echo "==> 展開中..." >&2
    tar -xzf "$tmp_dir/jdk.tar.gz" -C "$tmp_dir"
    extracted="$(find "$tmp_dir" -maxdepth 1 -type d -name 'jdk-21*' | head -1)"
    if [[ -z "$extracted" ]]; then
        echo "error: 展開先に JDK が見つかりません" >&2
        exit 1
    fi
    rm -rf "$DEST"
    mv "$extracted" "$DEST"
fi

major="$(java_major "$DEST" || true)"
if [[ "$major" != "21" ]]; then
    echo "error: JDK 21 の検証に失敗しました: $DEST" >&2
    exit 1
fi

echo "JAVA_HOME=$DEST"
"$DEST/bin/java" -version 2>&1 | sed 's/^/    /' >&2

if [[ "$PRINT_ENV" == "1" ]]; then
    echo "export JAVA_HOME=\"$DEST\""
    echo "export PATH=\"$DEST/bin:\$PATH\""
fi
