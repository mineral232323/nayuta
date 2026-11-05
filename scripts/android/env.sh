#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Nayuta — JDK 解決の共通処理
#
# build.sh / setup_sdk.sh から source して使う。優先順位:
#   1. 呼び出し側が明示した JAVA_HOME
#   2. システムの JDK 21（/usr/lib/jvm/java-21-openjdk）
#   3. リポジトリ内に置いた JDK（scripts/android/setup_jdk.sh が入れる）
# ---------------------------------------------------------------------------

# resolve_java_home: 使える JAVA_HOME を stdout に返す。見つからなければ 1。
resolve_java_home() {
    if [[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/java" ]]; then
        printf '%s\n' "$JAVA_HOME"
        return 0
    fi
    local candidate
    for candidate in \
        /usr/lib/jvm/java-21-openjdk \
        "${REPO_ROOT:-$PWD}/.cache/jdk21"
    do
        if [[ -x "$candidate/bin/java" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}
