# Nayuta Android IME

BERT 採点による日本語 IME の Android 側（土台 / M2）。現時点の変換エンジンは
「読みをそのまま返す」スタブで、UI・状態遷移・ビルド基盤が動くところまで。

設計は [../docs/design.md](../docs/design.md) を参照。

## ビルド

```bash
# 初回のみ: SDK を .android-sdk/ に入れる（ライセンス同意込み）
bash ../scripts/android/setup_sdk.sh

# ビルド（JDK21 / SDK / Gradle キャッシュの設定はラッパが面倒を見る）
bash ../scripts/android/build.sh assembleDebug
```

`scripts/android/build.sh` がやっていることを手で書くと:

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
export ANDROID_HOME=$PWD/../.android-sdk
export ANDROID_SDK_ROOT=$ANDROID_HOME
export GRADLE_USER_HOME=$PWD/../.cache/gradle   # $HOME 側は読み取り専用のため
cd android
./gradlew assembleDebug
```

APK は `android/app/build/outputs/apk/debug/app-debug.apk` に出力される。

## 構成

```
app/src/main/java/com/nayuta/ime/
├── NayutaImeService.kt          IME 本体（状態遷移・変換呼び出し・確定）
├── converter/
│   ├── Converter.kt             変換エンジンとの唯一のインターフェース
│   ├── KanaConverter.kt         スタブ（読みをそのまま返す）
│   └── Converters.kt            ★差し替え点（ここ 1 箇所だけ書き換える）
└── keyboard/
    ├── KeyboardLayout.kt        KeySpec / FlickKey / FunctionKey / KeyboardRow
    ├── KeyboardLayouts.kt       12 キー配列のテーブル（かな / 英数）
    ├── FlickKeyboardView.kt     描画・フリック判定・ポップアップ
    ├── CandidateView.kt         横スクロールの候補バー
    └── KeyboardColors.kt        配色
```

## 変換エンジンの差し替え

`com.nayuta.ime.converter.Converters.create()` の戻り値を変えるだけ。

```kotlin
private fun create(context: Context): Converter {
    return KanaConverter()   // ← ここを LatticeBertConverter(...) に
}
```

`Converter.convert(reading, context)` は「ひらがなの読み + 直前の文脈文字列」を
受け取り、優先順の候補リストを返す。呼び出しはサービス側の単一スレッド executor
から行われるので、実装はブロッキングしてよい（ONNX 推論をそのまま書ける）。

## キーボード配列（かなモード）

```
あ(あいうえお)  か(かきくけこ)  さ(さしすせそ)
た(たちつてと)  な(なにぬねの)  は(はひふへほ)
ま(まみむめも)  や(や ゆ よ)    ら(らりるれろ)
わ(わをんー、)  小(っゃゅょぁ)  ゛(゛゜、。・)
[英数] [空白] [変換] [削除]
```

フリック方向は 上=2 番目 / 右=3 番目 / 左=4 番目 / 下=5 番目。
や行は「や・ゆ・よ」しかないため上下左右の未使用方向は無入力。
英数モードは電話機の 12 キー英字配列（ABC/DEF/GHI …）。

## 実機で使う

実機接続後:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
# 設定 → システム → 言語と入力 → 画面キーボード → Nayuta を有効化
adb shell ime enable com.nayuta.ime/.NayutaImeService
adb shell ime set com.nayuta.ime/.NayutaImeService
```
