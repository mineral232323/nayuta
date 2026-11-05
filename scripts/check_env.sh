#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — 開発環境のチェック
#
# 「ビルドできる状態か」を一通り確認する。足りないものがあれば直し方も出す。
#
# 使い方:
#   bash scripts/check_env.sh
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/android/env.sh
source "$REPO_ROOT/scripts/android/env.sh"

ok=0
ng=0
pass() { printf '  [ OK ] %s\n' "$1"; ok=$((ok + 1)); }
fail() { printf '  [ NG ] %s\n' "$1"; ng=$((ng + 1)); }
skip() { printf '  [ -- ] %s\n' "$1"; }

echo "== Python =="
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    pyver="$("$REPO_ROOT/.venv/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    pass ".venv の Python $pyver"
    if "$REPO_ROOT/.venv/bin/python" - <<'PY'
import importlib, sys
missing = []
for m in ("torch", "transformers", "pyarrow", "sudachipy", "onnx", "onnxruntime"):
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(f"{m} ({e})")
if missing:
    print("missing: " + ", ".join(missing))
    sys.exit(1)
PY
    then
        pass "学習・評価用の依存（torch / transformers / sudachipy / onnxruntime など）"
    else
        fail "依存が不足しています → uv sync"
    fi
else
    fail ".venv がありません → uv sync"
fi

echo "== JDK =="
if JAVA_HOME="$(resolve_java_home)"; then
    major="$("$JAVA_HOME/bin/java" -version 2>&1 | sed -n 's/.*version "\([0-9]*\).*/\1/p' | head -1)"
    if [[ "$major" == "21" ]]; then
        pass "JDK 21 ($JAVA_HOME)"
    else
        fail "JDK $major が見つかりました（21 が必要）→ bash scripts/android/setup_jdk.sh"
    fi
else
    fail "JDK が見つかりません → bash scripts/android/setup_jdk.sh"
fi

echo "== Android SDK =="
SDK="$REPO_ROOT/.android-sdk"
[[ -d "$SDK/platforms/android-35" ]] &&
    pass "platforms;android-35" || fail "platforms;android-35 が無い → bash scripts/android/setup_sdk.sh"
[[ -d "$SDK/build-tools/35.0.0" ]] &&
    pass "build-tools;35.0.0" || fail "build-tools;35.0.0 が無い → bash scripts/android/setup_sdk.sh"
[[ -x "$SDK/platform-tools/adb" ]] &&
    pass "platform-tools (adb)" || fail "platform-tools が無い → bash scripts/android/setup_sdk.sh"
[[ -f "$REPO_ROOT/android/local.properties" ]] &&
    pass "android/local.properties" || fail "android/local.properties が無い → bash scripts/android/setup_sdk.sh"

echo "== 学習済みアセット =="
ASSETS="$REPO_ROOT/android/app/src/main/assets"
if [[ -f "$ASSETS/dict/lexicon.tsv.gz" || -f "$ASSETS/dict/lexicon.tsv" ]]; then
    pass "辞書 (dict/lexicon.tsv[.gz])"
else
    fail "辞書が無い → bash scripts/android/fetch_assets.sh"
fi
for f in dict/bigram.bin dict/english.tsv model/scorer.onnx model/vocab.txt; do
    [[ -f "$ASSETS/$f" ]] && pass "$f" || fail "$f が無い → bash scripts/android/fetch_assets.sh"
done

echo "== 成果物 =="
APK="$REPO_ROOT/android/app/build/outputs/apk/debug/app-debug.apk"
if [[ -f "$APK" ]]; then
    pass "APK ($(du -h "$APK" | cut -f1))"
else
    skip "APK 未ビルド → bash scripts/android/build.sh"
fi

echo "== 端末 =="
if command -v adb >/dev/null && adb devices | awk 'NR>1 && $2=="device"' | grep -q .; then
    pass "adb 接続端末あり"
else
    skip "adb 接続端末なし（実機確認は後回しで可）"
fi

echo
echo "結果: OK=$ok NG=$ng"
[[ "$ng" -eq 0 ]]
