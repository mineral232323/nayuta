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
- 添付ファイル: `nayuta-v<version>-release.apk`（例: `nayuta-v0.2.6-release.apk`）。リリース用の
  署名 Secrets が未設定の場合は `nayuta-v<version>-debug.apk`（debug 署名）にフォールバックする。
  これらの名前は `scripts/android/fetch_assets.sh` が吸い上げる APK 名と一致させている
- 署名: 下記「リリース署名（keystore）」の Secrets が設定されていれば release ビルドを正規の鍵で署名する。
  未設定の場合は debug 署名の APK を配布する（keystore の secret は不要）
- モデル・辞書（`assets/`）はリポジトリに含めない。「自分を除く最新リリース」の APK から
  `fetch_assets.sh` で取得する（`self-tag` input で自身のタグを除外し、自己参照を回避）

## リリース署名（keystore）

リリース APK は正規の鍵で署名する。鍵は GitHub Secrets 経由でのみ CI に渡り、リポジトリには置かない。

### 1. keystore を作る（1 回だけ・ローカルで実行）

```bash
keytool -genkeypair -v \
  -keystore nayuta-release.jks -alias nayuta -keyalg RSA -keysize 4096 -validity 10000
```

- 途中で聞かれるパスワード（keystore と key は同じでよい）と名前を入力する
- **`nayuta-release.jks` は絶対にコミットしない・紛失しないこと**。紛失すると同じ鍵での
  アップデート配布が二度とできなくなる（別アプリとして再インストールになる）
- バックアップは安全な場所（パスワードマネージャ等）に保管する

### 2. GitHub Secrets に登録する

リポジトリの Settings → Secrets and variables → Actions に以下を登録する
（base64 は `base64 -w0 nayuta-release.jks` で作れる）:

| Secret 名 | 内容 |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | `nayuta-release.jks` を base64 エンコードした文字列 |
| `ANDROID_KEYSTORE_PASSWORD` | keystore のパスワード |
| `ANDROID_KEY_ALIAS` | エイリアス（上の例なら `nayuta`） |
| `ANDROID_KEY_PASSWORD` | key のパスワード（keystore と同じなら同じ値） |

### 3. 動作

- `release.yml` が Secrets を復元して環境変数（`NAYUTA_KEYSTORE_PATH` 等）経由で Gradle に渡し、
  `assembleRelease` を実行する。`build.gradle.kts` の `signingConfigs.release` がこれらを読む
- 署名済み APK は `apksigner verify --print-certs` で検証してからリリースに添付される
- Secrets が未設定の場合は自動的に debug 署名にフォールバックする（CI が赤くならない）
- ローカルで署名付き release ビルドをしたい場合は環境変数を設定して
  `bash scripts/android/build.sh assembleRelease` を実行する

## バージョンの上げ方

1. `android/app/build.gradle.kts` の `versionName`（例: `0.2.5` → `0.2.6`）と `versionCode`（例: `6` → `7`）を上げる
2. master にマージする（PR 経由でも直接 push でもよい）
3. master push で `release.yml` が走り、`v0.2.6-pre` の pre-release が自動作成され
   `nayuta-v0.2.6-release.apk`（署名 Secrets 未設定なら `nayuta-v0.2.6-debug.apk`）が添付される

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
