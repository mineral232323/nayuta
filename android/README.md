# Nayuta Android IME

BERT 採点つきの日本語 IME 本体（Kotlin）。プロジェクト全体は
[トップの README](../README.md) を参照。

- `InputMethodService` ベース。キーボードは**自前 View で描画**（フリック判定・候補バー・配色も自前）
- 変換は端末内で完結（ラティス + Viterbi + BERT 再ランク + bigram）
- 設定画面だけ Material 3（`MaterialCardView` / `ExposedDropdownMenu` など）を使っている

## ビルド

```bash
# 初回のみ: SDK を .android-sdk/ に入れる（ライセンス同意込み）
bash ../scripts/android/setup_sdk.sh

# ビルド（JDK21 / SDK / Gradle キャッシュの設定はラッパが面倒を見る）
bash ../scripts/android/build.sh
```

`scripts/android/build.sh` がやっていることを手で書くと:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
export ANDROID_HOME=$PWD/../.android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export GRADLE_USER_HOME=$PWD/../.cache/gradle
cd android
./gradlew assembleDebug
```

学習済みモデルと辞書（`app/src/main/assets/`、約 30MB）は**リポジトリに含まれない**。
`train/` の手順で生成する（[トップの README](../README.md#2-学習済みアセットを生成)）。

## 構成

```
app/src/main/java/com/nayuta/ime/
├── NayutaImeService.kt          IME 本体（状態遷移・ライブ変換・確定・学習呼び出し）
├── NayutaSettings.kt            設定（SharedPreferences）
├── SettingsActivity.kt          設定画面（Material 3）
├── TestInputActivity.kt         動作確認用の入力画面
├── converter/
│   ├── Converter.kt             変換エンジンのインターフェース
│   ├── Converters.kt            生成点（ここで実装を選ぶ）
│   ├── KanaConverter.kt         フォールバック（辞書なしで読みを返す）
│   ├── bert/
│   │   ├── BertScorer.kt        ONNX 推論（採点 / phi）
│   │   ├── OnlineLearner.kt     薄い層のオンライン学習
│   │   └── CharVocab.kt         文字語彙
│   └── dict/
│       ├── LatticeConverter.kt  本命の変換（ラティス + Viterbi + 再ランク）
│       ├── Dictionary.kt        辞書（読み → 候補）
│       ├── BigramLM.kt          単語 bigram
│       ├── EnglishLexicon.kt    カタカナ語 → 英単語
│       └── UserDictionary.kt    ユーザー辞書
└── keyboard/
    ├── KeyboardLayout.kt        KeySpec / FlickKey / FunctionKey / SpacerKey
    ├── KeyboardLayouts.kt       かな / 英字(QWERTY) / 数字テンキー / 記号 の配列
    ├── FlickKeyboardView.kt     描画・フリック判定・長押し・ポップアップ
    ├── CandidateView.kt         横スクロールの候補バー
    └── KeyboardColors.kt        配色（ライト / ダーク）
```

## キーボード配列

### かな（12 キー + 補助列）

```
1@  あ  か  さ  ⌫
◀   た  な  は  ▶
数字 ま  や  ら  変換
あa 小゛ わ 、。  ✓
```

フリックは母音の並びに合わせて **左=い / 上=う / 右=え / 下=お**:

```
       う
  い ← あ → え
       お
```

や行・わ行は未使用方向が無入力。`小゛` の長押しで半濁点。

### 英字（QWERTY）

```
1 2 3 4 5 6 7 8 9 0     ← 設定で表示/非表示
q w e r t y u i o p
 a s d f g h j k l
⇧ z x c v b n m  ⌫
あa  ,   空白   .  123  ✓
```

1 行の高さは 0.82 倍にして、かなより少し低くしてある。

### 数字・記号（`1@` / `123` から）

```
1 2 3 4 5 6 7 8 9 0
@ # ¥ % & - + ( ) /
ABC * " ' : ; ! ?  ⌫
あa  ,   空白   .   ⏎
```

かなの `数字` キーは従来の 12 キーテンキー（`1`〜`9` `0` `.` `,` など）。

## デザイン

見た目は Gboard と Material 3 に合わせている。色は
[keyboard/KeyboardColors.kt](app/src/main/java/com/nayuta/ime/keyboard/KeyboardColors.kt)
に Material のカラーロール名でまとまっている。

| ロール | 値 | 用途 |
|---|---|---|
| surfaceContainer | `#F1F3F4` | キーボードの地 |
| surface | `#F8F9FA` | 候補ストリップの地 |
| surfaceContainerLowest | `#FFFFFF` | キー（角丸 10dp） |
| surfaceContainerHigh | `#E6E8EB` | 機能キー |
| primary (state layer 12%) | `#E8F0FE` | 押下中のキー |
| primary | `#0B57D0` | 押下中のフリック方向のサブラベル |

- タイポグラフィ: かな 20sp、機能キー 13.5sp、サブラベル 11sp、候補 16sp
- 寸法: 4dp グリッド。行の基準高は設定（44 / 50 / 58dp）
- 押下中はキーが primary の状態レイヤーで薄く青くなり、上に角丸ポップアップが出る
- ダークモードは端末設定またはアプリ設定（システム / ライト / ダーク）に従う

## 実機で使う

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell ime enable com.nayuta.ime/.NayutaImeService
adb shell ime set    com.nayuta.ime/.NayutaImeService
```

挙動確認は `adb shell am start -n com.nayuta.ime/.TestInputActivity` の
テスト入力画面で行う（個人情報を含むアプリを開かなくて済む）。

## デバッグ

```bash
adb logcat -s NayutaConv NayutaDict NayutaBert NayutaOnline
```

| タグ | 内容 |
|---|---|
| `NayutaConv` | 変換結果と所要時間 |
| `NayutaDict` | 辞書・bigram の読み込み |
| `NayutaBert` | モデルの読み込み |
| `NayutaOnline` | オンライン学習の更新ログ |
