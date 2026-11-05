# 開発環境のセットアップ

このリポジトリを clone した直後から「APK をビルドして実機に入れる」ところまでの手順。
想定環境は Linux x86_64（CachyOS / Arch 系で確認）。**sudo 不要**で、必要なものは
すべてリポジトリ内（`.venv/` / `.cache/` / `.android-sdk/`）に閉じる。

> 学習データとモデルは git 管理外。手っ取り早く動かす場合は
> [配布 APK からアセットを取り出す](#4-学習済みアセットを用意する)のが速い。

## 0. まとめ（コピペ用）

```bash
# 1) Python（PyTorch は CPU 版）
uv sync

# 2) JDK 21（プロジェクト内に入れる。システムに無い場合）
bash scripts/android/setup_jdk.sh

# 3) Android SDK（cmdline-tools / platform 35 / build-tools 35）
bash scripts/android/setup_sdk.sh

# 4) 学習済みモデルと辞書（配布 APK から展開）
bash scripts/android/fetch_assets.sh

# 5) ビルド
bash scripts/android/build.sh

# 6) 確認
bash scripts/check_env.sh
```

## 1. Python（学習・評価・書き出し）

```bash
uv sync
```

- `.python-version` により **Python 3.12** が uv 管理で入る（システムは 3.14 でも可）。
- 依存は [pyproject.toml](../pyproject.toml) に定義。`train/` はパッケージとして
  インストールせず、リポジトリ root から `python -m train.<script>` で実行する。
- このマシンには NVIDIA GPU が無いため **CPU 版 torch** を入れる設定にしてある
  （[pyproject.toml](../pyproject.toml) の `tool.uv.sources`）。
  CUDA 版が要る場合は `pytorch-cpu` の URL を `https://download.pytorch.org/whl/cu128`
  に変えて `uv sync` し直す。

- Hugging Face のキャッシュをリポジトリ内に置く場合:

  ```bash
  export UV_CACHE_DIR=$PWD/.cache/uv
  export HF_HOME=$PWD/.cache/huggingface
  ```

## 2. JDK 21

AGP 8.9 / `sdkmanager` は JDK 17+ を要求する。システムに JDK 21 が無い場合は
Temurin 21 を `.cache/jdk21` に展開する:

```bash
bash scripts/android/setup_jdk.sh
# 現在の shell に反映する場合
eval "$(bash scripts/android/setup_jdk.sh --print-env)"
```

`build.sh` / `setup_sdk.sh` は次の順で `JAVA_HOME` を解決する
（[scripts/android/env.sh](../scripts/android/env.sh)）:

1. 明示された `JAVA_HOME`
2. `/usr/lib/jvm/java-21-openjdk`
3. `.cache/jdk21`

## 3. Android SDK

```bash
bash scripts/android/setup_sdk.sh
```

- `.android-sdk/` に cmdline-tools / `platforms;android-35` / `build-tools;35.0.0` /
  `platform-tools` を入れる。
- `android/local.properties` に `sdk.dir` を書く（git 管理外）。
- 何度実行しても既存は再取得しない。

## 4. 学習済みアセットを用意する

モデルと辞書（約 27MB）はリポジトリに含まれない。2 通りある。

### A. 配布 APK から取り出す（推奨・数秒）

```bash
bash scripts/android/fetch_assets.sh
```

`v0.2.4-pre` の debug APK を `.cache/release/` に落とし、中の
`assets/dict/*` と `assets/model/*` を `android/app/src/main/assets/` に展開する。
別のバージョンを使う場合:

```bash
NAYUTA_RELEASE_TAG=v0.2.3-pre bash scripts/android/fetch_assets.sh
```

### B. 学習から作り直す（GPU 推奨）

```bash
python -m train.prep_data          # data/raw に Wikipedia dump が必要
python -m train.build_lex_v2 --variant-penalty 4.5
python -m train.build_english_lexicon
python -m train.train --arch scratch-small --lexicon lexicon_v2.tsv --alt-prob 0.25
python -m train.export_dict
python -m train.build_bigram
python -m train.export_bigram
python -m train.export_onnx_online
```

詳細は [トップの README](../README.md#2-学習済みアセットを生成) を参照。
CPU のみだと学習は非現実的なので、この経路は GPU のあるマシン向け。

## 5. ビルド

```bash
bash scripts/android/build.sh
# => android/app/build/outputs/apk/debug/app-debug.apk （約 54MB）
```

`build.sh` は JDK / SDK / Gradle キャッシュ（`.cache/gradle`）の設定をまとめて行う。
辞書アセットが無い場合はエラーにせず、生成できるなら `train.export_dict` を試す。

エミュレーター（x86_64）で動かす場合は ABI を切り替える:

```bash
bash scripts/android/build.sh assembleDebug -PnayutaAbi=x86_64
```

## 6. 実機に入れる

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell ime enable com.nayuta.ime/.NayutaImeService
adb shell ime set    com.nayuta.ime/.NayutaImeService
adb shell am start -n com.nayuta.ime/.TestInputActivity
```

端末が接続されていない場合は、ビルドとアセットの検証までで止めてよい。

## 7. 環境チェック

```bash
bash scripts/check_env.sh
```

Python / JDK / SDK / アセット / APK / adb 接続をまとめて確認する。

## 8. エミュレーターでの確認（任意）

実機が無いときは x86_64 の AVD で確認できる。`setup_sdk.sh` には含まれないので、
使う場合だけ追加で入れる（イメージと AVD で約 11GB）。

```bash
export JAVA_HOME=$PWD/.cache/jdk21 PATH=$PWD/.cache/jdk21/bin:$PATH
SDK=$PWD/.android-sdk

$SDK/cmdline-tools/latest/bin/sdkmanager \
    "emulator" "system-images;android-36;google_apis;x86_64"
echo no | $SDK/cmdline-tools/latest/bin/avdmanager create avd \
    -n nayuta36 -k "system-images;android-36;google_apis;x86_64" -d pixel_6

export ANDROID_AVD_HOME=$PWD/.cache/android/avd
$SDK/emulator/emulator -avd nayuta36 \
    -no-window -no-audio -no-boot-anim -gpu swiftshader_indirect
```

起動後は x86_64 用の APK をビルドして入れる:

```bash
bash scripts/android/build.sh assembleDebug -PnayutaAbi=x86_64
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell ime enable com.nayuta.ime/.NayutaImeService
adb shell ime set    com.nayuta.ime/.NayutaImeService
adb shell am start -n com.nayuta.ime/.TestInputActivity
```

不要になったらイメージと AVD を消す:

```bash
$SDK/cmdline-tools/latest/bin/sdkmanager --uninstall "system-images;android-36;google_apis;x86_64"
rm -rf .cache/android
```

## 生成物とサイズの目安

| パス | 内容 | サイズ |
|---|---|---|
| `.venv/` | Python 3.12 + 依存（CPU torch） | 約 1.5GB |
| `.cache/jdk21/` | Temurin JDK 21 | 約 300MB |
| `.cache/gradle/` | Gradle / AGP / 依存キャッシュ | 約 1GB |
| `.android-sdk/` | Android SDK | 約 450MB（+エミュレーター約 8GB） |
| `android/app/src/main/assets/` | 辞書・bigram・モデル | 約 27MB |

いずれも `.gitignore` 対象。`git status` を汚さない。

## 詰まりやすい点

- **sudo が使えない** — JDK は `.cache/jdk21` に入れる。システム側を触らない。
- **`uv sync` が Python 3.14 を使う** — `.python-version` があるので通常は 3.12 が選ばれる。
  手動で指定するなら `uv sync --python 3.12`。
- **APK は作れるが起動時に落ちる** — 辞書アセットが無いと `LatticeConverter` が
  辞書を読めない。`scripts/check_env.sh` でアセットを確認する。
- **実機で BERT が効かない** — `model/scorer.onnx` が無いと unigram のみに
  フォールバックする（ログ `NayutaBert` に出る）。
- **GPU が無い** — 学習・評価は CPU でも動くが、`train.train` は現実的な時間で終わらない。
  推論と CLI の確認は CPU で十分。
