# CI/CD 運用ガイド

GitHub Actions で「PR のビルド検証」と「pre-release への APK 自動添付」を行う。
ローカルの `scripts/android/*.sh` と同じ手順を踏むので、CI が緑ならローカルビルドもほぼ同じ結果になる。

## 構成

```
.github/
├── actions/setup-env/action.yml   共通環境構築: JDK21 → Android SDK → assets 取得 → (署名鍵の復号) → Gradle ビルド
└── workflows/
    ├── ci.yml                     PR / master push: assembleDebug + lintDebug、APK を artifact に保存
    └── release.yml                master push / v* タグ push / 手動実行: 署名付き release APK を pre-release に添付
```

## トリガーと動作

| トリガー | workflow | 動作 |
|---|---|---|
| PR を開く / 更新 | `ci.yml` | `assembleDebug lintDebug` を実行。APK を artifact `nayuta-debug-<sha>` として保存（14日保持）。debug 署名 |
| master に push（PR マージ含む） | `ci.yml` + `release.yml` | CI に加え、`build.gradle.kts` の `versionName` から `vX.Y.Z-pre` を導出し、その pre-release を新規作成するか、既存なら APK を `--clobber` で上書きする |
| `v*` タグを push | `release.yml` | そのタグ名で pre-release を作成し、APK を添付する |
| Actions 画面から「Release」を手動実行 | `release.yml` | `ref`（ビルド対象 ref）と `tag`（作成するタグ名。省略時は `v<versionName>-pre` を自動採番）を指定できる |

ドキュメントのみの変更（`docs/**`、`*.md`）では `release.yml` は走らない（`paths-ignore`）。

## リリースの中身

- pre-release のタグ/名前: `v0.2.5-pre` / 「Nayuta v0.2.5」のような `v<versionName>-pre` 形式。すべて pre-release
- 添付ファイル: `nayuta-v<version>-release.apk`（例: `nayuta-v0.2.6-release.apk`）。この名前は
  `scripts/android/fetch_assets.sh` が吸い上げる APK 名と一致させている
- 署名: リリース用 keystore（PKCS12 / RSA 4096）で **v2 + v3 署名**する。署名直後に証明書の
  SHA-256 指紋を照合し、鍵の取り違えや debug 署名へのフォールバックを検出して失敗させる
- Secret が未登録の場合、`release.yml` はビルド前に失敗する（debug 署名の APK を配布しないため）
- モデル・辞書（`assets/`）はリポジトリに含めない。「自分を除く最新リリース」の APK から
  `fetch_assets.sh` で取得する（`self-tag` input で自身のタグを除外し、自己参照を回避）

## 署名鍵（Secrets）

`release.yml` は次の2つの Secret を使う。どちらもリポジトリの Secrets に登録する。

| Secret | 内容 | 登録方法 |
|---|---|---|
| `NAYUTA_KEYSTORE_BASE64` | `.cache/nayuta-release.p12` を base64 化したもの | `base64 -w0 .cache/nayuta-release.p12` の出力を `gh secret set NAYUTA_KEYSTORE_BASE64` に渡す |
| `NAYUTA_STORE_PASSWORD` | キーストアのパスワード（PKCS12 なので鍵パスワードも兼ねる） | `gh secret set NAYUTA_STORE_PASSWORD`（対話入力） |

`setup-env` が Secret を `.cache/nayuta-release.p12` に復号し、`NAYUTA_*` 環境変数を
`$GITHUB_ENV` に書いて Gradle に渡す。`build.gradle.kts` は環境変数 > `keystore.properties` の順で読み、
どちらも無ければ debug 署名にフォールバックする（`release.yml` のガードで CI では未然に防いでいる）。

**`ci.yml`（PR 検証）は Secret を使わず debug 署名のまま。** fork からの PR に鍵を渡さないための意図的な分離。

ローカルでは `android/keystore.properties`（git 管理外）が同じ役割を果たす。書式は
`android/keystore.properties.example` を参照。

## 承認制（Environment）

`release.yml` の `release` ジョブは `environment: release` を指定している。この Environment には
Required reviewers が設定済みで、**承認されるまでジョブは開始しない**。ビルドも鍵の復号も承認後
なので、署名鍵が人の確認なしに使われることはない。

| 設定 | 値 | 理由 |
|---|---|---|
| Required reviewers | `mineral232323` | 実行前に人の確認を挟む |
| `can_admins_bypass` | `false` | 管理者でも承認を飛ばせない |
| `prevent_self_review` | `false` | 自分の push を自分で承認できる（1人運用のため） |
| Deployment branches | 未設定 | 任意のブランチ/タグから実行可能。master と `v*` に絞ることも可能 |

承認待ちの間は Actions の実行一覧に「Waiting for review」と表示され、Review deployments から
承認すると開始する。設定画面: `Settings → Environments → release`。

`ci.yml`（PR 検証）は Environment を使わないので、承認なしで従来どおり走る。

## バージョンの上げ方

1. `android/app/build.gradle.kts` の `versionName`（例: `0.2.5` → `0.2.6`）と `versionCode`（例: `6` → `7`）を上げる
2. master にマージする（PR 経由でも直接 push でもよい）
3. master push で `release.yml` が走り、`v0.2.6-pre` の pre-release が自動作成され `nayuta-v0.2.6-release.apk` が添付される

## ローカル手順との対応

| CI のステップ | ローカルで同じことをするには |
|---|---|
| `actions/setup-java` (temurin 21) | `bash scripts/android/setup_jdk.sh` |
| `bash scripts/android/setup_sdk.sh` | 同じ（`.android-sdk/` に入る） |
| `fetch_assets.sh`（最新リリースから） | `bash scripts/android/fetch_assets.sh` |
| Gradle ビルド（PR 検証） | `bash scripts/android/build.sh assembleDebug` |
| Gradle ビルド（リリース） | `bash scripts/android/build.sh assembleRelease`（`android/keystore.properties` が必要） |

##ハマりどころ（すでに対処済み・再発防止のメモ）

- **composite action 内では `secrets` コンテキストを直接参照できない**
  （`Unrecognized named-value: 'secrets'`）→ 呼び出し元 workflow から input で渡す
  （`github-token` / `keystore-base64` / `store-password`）
- **composite action の step 間で環境変数は引き継がれない**（step ごとに別シェル）→
  復号した鍵のパスやパスワードは `$GITHUB_ENV` に書いて後続 step に渡す
- **`gh ... --jq` は `--arg` を解釈しない**（`unknown command "self"`）→ jq 側で変数を使わず、
  シェルの `grep -v -x -F "$SELF_TAG"` で除外する
- **配布 APK の名前を変えると `fetch_assets.sh` が壊れる** → 添付名（`-release.apk`）と
  `fetch_assets.sh` の候補リストは必ず同時に更新する。旧 `-debug.apk` は後方互換で残している
- **既存コードの lint エラーで `lintDebug` が abort する** → `build.gradle.kts` で
  `lint { abortOnError = false }`（レポートは出力される）
- actions v4 系は Node.js 20 非推奨の警告が出る（現状は動作する）。将来 v5 系へ移行を検討
- GitHub 側の障害で push が `remote: fatal error in commit_refs` で失敗することがある
  （2026-11-04〜05 のインシデントなど）。時間を置いて再試行で復旧する

## 手動でリリースをやり直したい場合

Actions 画面 → 「Release」→ "Run workflow" で `ref` と `tag` を指定して実行。
既存タグを指定した場合は `gh release upload --clobber` で同名 APK が上書きされる。
