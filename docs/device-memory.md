# 実機メモリの内訳（2026-11-07）

> 端末: SC-51E / Android 17。`TestInputActivity` の `--es mem` フックで
> コンポーネントを 1 つずつロードし、GC 後の Java ヒープとネイティブヒープを記録した。

## 計測方法

```bash
adb shell am start -n com.nayuta.ime/.TestInputActivity --es mem 1
adb logcat -s NayutaMem
adb shell dumpsys meminfo com.nayuta.ime
```

## コンポーネント別（辞書パック化 **前**: new500k 辞書 + 20万語 bigram）

| ステージ | Java ヒープ | ネイティブ | 差分 |
|---|---|---|---|
| 起動直後 | 3MB | 10MB | — |
| 辞書ロード後 | 55MB | 10MB | **辞書 +52MB** |
| bigram ロード後 | 90MB | 10MB | **bigram +35MB** |
| BERT ロード後 | 90MB | 35MB | **BERT +25MB** |

- 辞書: 500,000 読み / 950,827 エントリを `Array<String>` + `IntArray` で保持（文字列が支配的）
- bigram: 200,000 語 / 3,447,509 連接。CSR 配列（rights 13.8MB + costs 6.9MB）+ `HashMap<String,Int>` と語文字列
- BERT: ONNX Runtime のセッション＋int8 モデル（14MB）で約 25MB のネイティブ

## 辞書のパック化（2026-11-07 実施）

`train/export_dict.py` が `assets/dict/lexicon.ndict`（パック済みバイナリ）も書き出し、
`Dictionary` はそれを **mmap** で読むように変更した（Gradle で `.ndict` を非圧縮同梱）。
旧 `lexicon.tsv.gz` しか無い場合は TSV から同じレイアウトに詰め直すフォールバックあり。

| 項目 | 変更前 | 変更後 |
|---|---|---|
| 辞書ロード時間 | 1804ms | **1ms**（mmap のみ） |
| 辞書の Java ヒープ | +52MB | **+0MB**（mapped ページ） |
| Java Heap（PSS） | 100.6MB | **49.6MB** |
| TOTAL PSS（ベンチ画面） | 240.9MB | **203.1MB** |
| 初回変換（コールド） | 2323ms | **784ms**（BERT+bigram ロードが支配） |
| APK | 86.6MB | 86.9MB（ndict 24MB を非圧縮同梱、旧 tsv は削除） |

残っている大きいもの: bigram の Java ヒープ **+35MB**、BERT のネイティブ **+25MB**、
テスト画面の Graphics 45MB、debug ビルドの Code 32MB。

## プロセス全体（PSS）

| 内訳 | PSS |
|---|---|
| Java Heap | 100.6MB |
| Native Heap | 42.8MB |
| Code（.apk 18.1 / .dex 10.3 / .so 4.8 / .jar 2.2） | 32.5MB |
| Graphics（EGL/GL mtrack） | 44.6MB |
| Dalvik Other / .art / その他 | 20.4MB |
| **TOTAL** | **240.9MB** |

- Graphics 44.6MB はテスト入力画面（フルスクリーン）の描画バッファ。IME のキーボード表示時とは異なる。
- 旧アセット（10万語 bigram）では bigram が約 18MB（−17MB）、PSS 253MB 前後だった。

## 比較

| IME | PSS | 備考 |
|---|---|---|
| Sumire（`com.kazumaproject.markdownhelperkeyboard`） | **50〜64MB** | zenz v3.2 Xsmall の GGUF 21MB を APK に同梱。zenz 推論のロードは未確認（maps に gguf 無し） |
| Samsung 標準（HoneyBoard） | 266MB | 既定キーボード |
| Nayuta（新アセット） | 241〜293MB | Java 100MB + Native 43MB がエンジン分 |

## 重いもの・削減候補

1. ~~**辞書 52MB（Java ヒープ）**~~ → **mmap 化で解消**（2026-11-07）。
2. **bigram 35MB（Java ヒープ）** — 20万語版。10万語に戻すと約 −17MB。
   mmap/ネイティブ化すれば Java ヒープから外せる（ページ回収も可能）。
3. **BERT 25MB（ネイティブ）** — ORT セッション + 14MB int8。ORT mobile 版やセッション設定の見直し余地。
4. **Code 32.5MB** — debug ビルド由来（.apk 18MB / .dex 10MB）。release + R8 で減る。
5. **Graphics 44.6MB** — テスト画面のフルスクリーン描画。IME 実使用時は別途測る必要あり。

## まとめ

- Nayuta のエンジンだけで **Java 100MB + Native 43MB ≒ 143MB**。
- Sumire は同種の IME としては極端に軽く、Samsung 標準と比べれば Nayuta は同程度。
- 「Sumire より軽くする」なら、**辞書と bigram を Java ヒープから外す**のが最短。
