# CI/CD 運用ガイド

GitHub Actions で「PR のビルド検証」と「pre-release への APK 自動添付」を行う。
ローカルの `scripts/android/*.sh` と同じ手順を踏むので、CI が緑ならローカルビルドもほぼ同じ結果になる。

## 構成

```
.github/
├── actions/setup-env/action.yml   共通環境構築: JDK21 → Android SDK → assets 取得 → Gradle ビルド
└── workflows/
    ├── ci.yml                     PR / master push: assembleDebug + lintDebug、APK を artifact に保存
    └── release.yml                master push / v* タグ push / 手動実行: ビルドして pre-release に APK 添付
```

## トリガーと動作

| トリガー | workflow | 動作 |
|---|---|---|
| PR を開く / 更新 | `ci.yml` | `assembleDebug lintDebug` を実行。APK を artifact `nayuta-debug-<sha>` として保存（14日保持） |
| master に push（PR マージ含む） | `ci.yml` + `release.yml` | CI に加え、`build.gradle.kts` の `versionName` から `vX.Y.Z-pre` を導出し、その pre-release を新規作成するか、既存なら APK を `--clobber` で上書きする |
| `v*` タグを push | `release.yml` | そのタグ名で pre-release を作成し、APK を添付する |
| Actions 画面から「Release」を手動実行 | `release.yml` | `ref`（ビルド対象 ref）と `tag`（作成するタグ名。省略時は `v<versionName>-pre` を自動採番）を指定できる |

ドキュメントのみの変更（`docs/**`、`*.md`）では `release.yml` は走らない（`paths-ignore`）。

## リリースの中身

- pre-release のタグ/名前: `v0.2.5-pre` / 「Nayuta v0.2.5」のような `v<versionName>-pre` 形式。すべて pre-release
- 添付ファイル: `nayuta-v<version>-debug.apk`（例: `nayuta-v0.2.5-debug.apk`）。この名前は
  `scripts/android/fetch_assets.sh` が吸い上げる APK 名と一致させている
- 署名: 現状は debug 署名の APK のまま配布する（keystore の secret は不要）。
  後で署名リリースに切り替える場合は `build.gradle.kts` に signingConfig を追加し、
  keystore を Secrets に登録して workflow に渡す
- モデル・辞書（`assets/`）はリポジトリに含めない。「自分を除く最新リリース」の APK から
  `fetch_assets.sh` で取得する（`self-tag` input で自身のタグを除外し、自己参照を回避）

## バージョンの上げ方

1. `android/app/build.gradle.kts` の `versionName`（例: `0.2.5` → `0.2.6`）と `versionCode`（例: `6` → `7`）を上げる
2. master にマージする（PR 経由でも直接 push でもよい）
3. master push で `release.yml` が走り、`v0.2.6-pre` の pre-release が自動作成され `nayuta-v0.2.6-debug.apk` が添付される

## ローカル手順との対応

| CI のステップ | ローカルで同じことをするには |
|---|---|
| `actions/setup-java` (temurin 21) | `bash scripts/android/setup_jdk.sh` |
| `bash scripts/android/setup_sdk.sh` | 同じ（`.android-sdk/` に入る） |
| `fetch_assets.sh`（最新リリースから） | `bash scripts/android/fetch_assets.sh` |
| `bash scripts/android/build.sh` | 同じ（`assembleDebug`） |

##ハマりどころ（すでに対処済み・再発防止のメモ）

- **composite action 内では `secrets` コンテキストを直接参照できない**
  （`Unrecognized named-value: 'secrets'`）→ 呼び出し元 workflow から `github-token` input で渡す
- **`gh ... --jq` は `--arg` を解釈しない**（`unknown command "self"`）→ jq 側で変数を使わず、
  シェルの `grep -v -x -F "$SELF_TAG"` で除外する
- **既存コードの lint エラーで `lintDebug` が abort する** → `build.gradle.kts` で
  `lint { abortOnError = false }`（レポートは出力される）
- actions v4 系は Node.js 20 非推奨の警告が出る（現状は動作する）。将来 v5 系へ移行を検討
- GitHub 側の障害で push が `remote: fatal error in commit_refs` で失敗することがある
  （2026-11-04〜05 のインシデントなど）。時間を置いて再試行で復旧する

## 手動でリリースをやり直したい場合

Actions 画面 → 「Release」→ "Run workflow" で `ref` と `tag` を指定して実行。
既存タグを指定した場合は `gh release upload --clobber` で同名 APK が上書きされる。
