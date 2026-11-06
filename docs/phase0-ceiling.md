# Phase 0 — 天井の分解測定（探索律速か辞書律速か）

> 実施日: 2026-11-06 / 状態: 下書き（Lead が最終化する前提）
> 評価ツール: `train/evaluate_v2.py`（新規。`train/evaluate_bigram.py` と同一の生成・再ランク手順）
> 固定評価セット:
> * `data/eval/wiki2000.jsonl`（`data/valid.jsonl` 先頭 2,000 文、以後不変）
> * `data/eval/sns2000.jsonl`（国会会議録由来 2,000 文、Lead 作成）
> 生データ: `logs/eval_v2_wiki_baseline.json`, `logs/eval_v2_sns_baseline.json`,
> `logs/phase0_ceiling.json`, `logs/phase0_ceiling_256.json`, `logs/phase0_search_*.json`

## 0. 結論（先に）

1. **天井 70.70% は「辞書欠落」だけのせいではない。ビーム刈り（探索）の寄与が大きい。**
   ラティス天井（正解経路が辞書・候補内に存在）= **84.95%** に対し、
   現行ビーム (32,48) の N-best 天井は **70.70%（−14.25pt）**。
   beam を 4 倍の (128,256) にすると **78.00%（+7.30pt）** まで戻る。
2. **辞書欠落はスパン単位で 1.78% だが、文単位では 15.05% の文がどこかのスパンで欠ける。**
   1 文あたり平均 9.7 スパンあるため、1.78% のスパン欠落が文単位の天井を大きく削る。
   Phase 1（辞書増強）の上限指標として `gold span coverage` を監視する。
3. **再ランクの選択損失は 13.30pt**（bigram 天井 66.85% → 実力 53.55%）。
   天井を上げても再ランクが選べなければ精度は上がらない。実際、beam を
   (128,256) に広げると天井 78.00% に対して実力 55.40% で、**選択損失は
   22.60pt に拡大**する（acc@3 は 67.05%）。天井を活かす次のレバーは再ランク。
4. **探索拡大のコスト**: 53.55% → 55.40%（+1.85pt）に対して
   p50 8.7 → 36.0ms（4.1 倍）・p95 176.8 → 240.7ms（1.36 倍）。
   端末目標 p95 < 20ms には厳しい。**妥協点は beam=64 / want=128（+1.2pt、p50 約 1.8 倍）**。
5. 判定: **探索律速寄り（ただし辞書も 15pt 分の余地）**。推奨設定は §7。

## 1. 測定環境（アセット一式）

「アセット一式が揃った状態」で測るため、辞書・bigram・ONNX・vocab の
パスと sha256 を JSON に記録している。今回の測定は現行 v2 アセット。

| アセット | パス | sha256（先頭 12） | サイズ |
|---|---|---|---|
| 辞書 | `data/lexicon_v2.tsv` | `7733405c9072` | 7.17 MB |
| bigram | `data/bigram.tsv` | `b92df2a8fe83` | 11.0 MB |
| bigram 語彙 | `data/bigram_words.tsv` | `ae0317e6d385` | 1.14 MB |
| モデル (int8 ONNX) | `android/app/src/main/assets/model/scorer.onnx` | `7280670f791a` | 14.2 MB |
| vocab | `android/app/src/main/assets/model/vocab.txt` | — | 27.9 KB |

- 推論: ONNX Runtime 1.31.1 / CPUExecutionProvider / **intra_op 4 threads**（端末 proxy）
- 評価セット: `data/eval/wiki2000.jsonl` 2,000 文
  sha256 `fb5a60a98f1c0b9148008faa0e3551cd3176ebd787a978b4c2e53957e74039a9`
  （`data/eval/wiki2000_legacy.jsonl` とバイト一致。`data/valid.jsonl` 先頭 2000 行）
- 候補数 k = 8（端末 `MAX_CANDIDATES=8` と同一）、スパン長上限 16。

## 2. ベースライン再現（現行アセット）

`PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 --data data/eval/wiki2000.jsonl \
 --domain wiki --limit 2000 --deltas 2.0 --betas 1.0,2.0,4.0,6.0 \
 --json-out logs/eval_v2_wiki_baseline.json`

| 構成（beam=32 / want=48） | 文正解率 | CER(micro) | CER(文平均) | 境界F1 | 境界P | 境界R | acc@3 |
|---|---|---|---|---|---|---|---|
| unigram ビームのみ | 35.95% | 12.61% | 13.51% | 0.9569 | 0.9578 | 0.9561 | - |
| unigram ビーム + BERT（bigram 採点なし, β=2.0） | 45.70% | 10.78% | 11.33% | 0.9556 | 0.9571 | 0.9541 | 56.75% |
| bigram ビームのみ (d=2.0) | 51.80% | 8.73% | 9.60% | 0.9714 | 0.9602 | 0.9828 | - |
| bigram+BERT (d=2.0, β=1.0) | **53.55%** | 8.37% | 9.04% | 0.9715 | 0.9623 | 0.9809 | 62.30% |
| bigram+BERT (d=2.0, β=2.0) | 53.15% | 8.45% | 9.17% | 0.9711 | 0.9615 | 0.9808 | 61.70% |
| bigram+BERT (d=2.0, β=4.0) | 52.55% | 8.60% | 9.44% | 0.9697 | 0.9594 | 0.9801 | 60.50% |
| bigram+BERT (d=2.0, β=6.0) | 51.85% | 8.76% | 9.70% | 0.9687 | 0.9577 | 0.9798 | 59.95% |

- README 記載値（35.95 / 45.45 / **53.45** / 天井 70.70）と整合。
  文正解率は **53.55%**（README は 53.45%）、BERT 単独も **45.70%**
  （README は 45.45%、β の探索範囲の違い）。差は ±0.1〜0.25pt で、
  Lead が走らせた `logs/baseline_eval.log`（53.55%）とも一致しており、
  アセット・beam・β の不一致ではない。
- **β は 1.0 が最良**（現行 README の探索範囲 2〜6 より小さい側）。
- 天井（N-best oracle）: unigram 生成 **70.70%**、bigram(d=2.0) 生成 **66.85%**。
- CER micro 8.37% / 文平均 9.04%。**文正解率と CER の順位はほぼ一致**
  （完全一致 53.55%、文字誤り率 8.37%）。
- 境界 F1 は 0.9715 と高い。**現行の誤りは「境界の切り方」よりも
  「表記（漢字）選択」と「経路が候補に無い」ことが主因**。
  （正解境界 8.68 個/文に対し予測 8.85 個/文で、やや過分割。）

### 入力長別（ベースライン、primary config = bigram+BERT d=2.0 β=1.0）

| 読み長 | 件数 | 文正解率 | CER(micro) |
|---|---|---|---|
| 0–5 | 1 | 0.00% | 100.00% |
| 6–9 | 39 | 61.54% | 22.54% |
| 10–13 | 209 | 69.38% | 12.15% |
| 14–18 | 387 | 63.82% | 11.48% |
| 19–24 | 370 | 59.19% | 8.79% |
| 25+ | 994 | 43.86% | 7.34% |

長文は CER が低いのに文正解率が低い（1 文字でも違うと不正解のため）。
評価セットは 25 文字以上が半数を占める。**短い文だけを見ると精度は 6〜7 割**あり、
「長文の 1 語誤り」を減らすのが効く。

## 3. 天井の 3 層分解

```
ラティス天井（正解経路が辞書×候補内に存在）         84.95%
  └─ ビーム刈り（探索）                         −14.25pt
N-best 天井（unigram 生成 beam=32/want=48）        70.70%   ← README の「到達可能な上限」
  └─ bigram 誘導コストによる刈り                − 3.85pt
N-best 天井（bigram d=2.0 生成 beam=32/want=48）   66.85%
  └─ 再ランクの選択ミス                        −13.30pt
最終（bigram+BERT β=1.0）                          53.55%
```

### 3.1 ラティス天井 = 辞書カバレッジ（gold span coverage）

| 指標 | 値 |
|---|---|
| 正解スパン総数 | 19,362 |
| 読みが辞書スパンとして存在 | 98.98%（19,164） |
| ＋ 正解表記が候補上位 8 に入る | **98.22%**（19,018） |
| 全文の正解経路が存在する文 | **84.95%**（1,699 / 2,000） |

欠落スパンの理由内訳:

| 理由 | スパン数 | 例 |
|---|---|---|
| 読みが辞書に無い | 187 | ワッシュバーン, エコエコアザラク, インターハイ, フルリニューアル |
| 読みはあるが正解表記が上位 8 候補外 | 146 | 細（さい）, 紹（しょう）, 垣（かき）, 登茂（とも） |
| スパン長上限 16 超 | 11 | — |

- カタカナ固有名詞の欠落が目立つ。単漢字の稀読みは「辞書にはあるが
  `MAX_CANDIDATES=8` の枠に入らない」パターン。Phase 1 では
  「読みはあるが候補落ち」も改善対象（候補数を増やすか、頻度・コストの再校正）。
- 文単位の天井 84.95% は **「1.78% のスパン欠落 × 1 文 9.7 スパン」** の結果。
  Phase 1 の効果は `--coverage-only` の `sentence_path_rate` で直接追跡できる。

## 4. beam/want スイープ（Phase 0 の主測定）

現行アセット、`--sweep-only`（BERT なし・天井のみ）、2,000 文。
`--beam-sweep "32,48;64,128;128,256;32,256"`。

| beam | want | unigram 1位 | unigram 天井 | bigram(d=2) 1位 | bigram(d=2) 天井 |
|---|---|---|---|---|---|
| 32 | 48 | 35.95% | **70.70%** | 51.80% | **66.85%** |
| 32 | 256 | 35.95% | 72.90% | 51.80% | 67.95% |
| 64 | 128 | 35.95% | 75.35% | 52.15% | 70.80% |
| 128 | 256 | 35.95% | **78.00%** | 52.60% | **74.20%** |
| 256 | 512 | 35.70% | 80.40% | 52.20%※ | 76.20% |

※ (256,512) は計算時間の都合で **最初の 1,000 文**で測定（同じ 1,000 文での
(128,256) は unigram 77.30% / bigram 73.20%、ラティス天井 87.00%）。
1,000 文サブセットでも beam 128→256 で **+3.1pt / +3.0pt** 上がっており、
天井はまだ飽和していない。

読み取れること:

- **beam を増やすと天井は単調に上がる**（70.70 → 75.35 → 78.00）。
  (32,48)→(64,128) で +4.65pt、(64,128)→(128,256) で +2.65pt。
- **want だけ増やしても +2.20pt**（32,48→32,256）。ビーム幅は同じで、
  最終リストの打ち切りを緩めただけ。**刈りの主因は beam 側**。
- **bigram 誘導ビームは unigram ビームより天井が低い**（74.20 vs 78.00）。
  delta 付き連接コストで経路生成すると、正解経路がビームから落ちやすい。
  正解経路は unigram コストでは上位に来やすいが、bigram コストでは順位が下がる。
- **unigram の 1 位精度は beam を増やしても 35.95% で不変**
  （コスト最良経路は変わらない）。天井の改善は「BERT 再ランクして初めて価値になる」。
- ラティス天井 84.95% との差: (128,256) でも 78.00% なので、
  残り 6.95pt はまだ探索で落ちている。一方、辞書由来の 15.05pt も残っている。

## 5. 探索を広げると最終精度は上がるか（rerank 後）

天井が上がっても再ランクが選べなければ精度は上がらない。同じ 2,000 文・
同じアセット・同じ CPU int8 で、探索設定を変えて**最終文正解率**を測った
（生データ: `logs/phase0_*.json`、全実行 2,000 文）。

| 生成 | beam/want | 再ランク | 文正解率 | CER(micro) | 境界F1 | acc@3 | 天井 | p50 / p95 (ms) |
|---|---|---|---|---|---|---|---|---|
| bigram | 32 / 48 | bigram+BERT β=1.0 | 53.55% | 8.37% | 0.9715 | 62.30% | 66.85% | 8.7 / 176.8 |
| unigram | 32 / 48 | bigram+BERT β=1.0 | 53.10% | 8.85% | 0.9684 | 63.50% | 70.70% | 6.7 / 168.4 |
| unigram | 32 / 48 | BERT のみ β=2.0 | 45.70% | 10.78% | 0.9556 | 56.75% | 70.70% | 7.2 / 172.0 |
| bigram | 64 / 128 | bigram+BERT β=1.0 | 54.55% | 8.17% | 0.9722 | 64.35% | 70.80% | 16.3 / 195.6 |
| unigram | 64 / 128 | bigram+BERT β=1.0 | 54.75% | 8.47% | 0.9698 | 65.90% | 75.35% | 15.2 / 198.3 |
| bigram | 128 / 256 | bigram+BERT β=1.0 | 55.35% | 8.04% | 0.9722 | 66.10% | 74.20% | 44.6 / 251.9 |
| unigram | 128 / 256 | bigram+BERT β=1.0 | **55.40%** | 8.29% | 0.9708 | 67.05% | 78.00% | 36.0 / 240.7 |
| unigram | 128 / 256 | BERT のみ β=4.0 | 46.55% | 10.80% | 0.9521 | 56.75% | 78.00% | 43.7 / 263.7 |

bf1 = 境界 F1。`--gen unigram` は bigram を使わないビーム、`+bigram+BERT` は
生成した N-best に bigram 連接 + BERT で再ランクする構成。

読み取れること:

1. **探索を広げると最終精度も上がる**が、伸びは天井ほどではない。
   53.55 → **54.55/54.75（+1.0〜1.2pt, 64/128）** → **55.35/55.40（+1.8pt, 128/256）**。
   天井は +7.3pt 上がっているので、**広げた分の 6 割は再ランクが選べずに失われる**。
   選択損失は 13.30pt（32/48）→ 15.65pt（64/128）→ 18.85〜22.60pt（128/256）に拡大。
2. **生成は unigram の方がわずかに良い**（128/256 で 55.40 vs 55.35）。
   天井は一貫して unigram 生成が高い（64: 75.35 vs 70.80、128: 78.00 vs 74.20）。
   また bigram 生成はビーム内で連接表を引くため**遅い**（p50 44.6 vs 36.0ms）。
   ただし狭いビームでは bigram 誘導が効く（32/48: 53.55 vs 53.10）。
   → 端末では「狭いビーム + bigram 生成」、サーバ/高精度モードでは
   「unigram 生成 + bigram+BERT 再ランク」が良い。
3. **bigram 採点の寄与は依然大きい**: BERT のみ 45.70%（32/48）・46.55%（128/256）
   に対し bigram+BERT は 53.10%・55.40%。**+7.4〜8.9pt**。
4. **acc@3 は 62.30% → 67.05%** に上がる。top-3 に入るのに top-1 で外す文が
   11.65pt あり、ここが再ランク改善の直接の余地。
5. β は探索を広げても **1.0 が最良**（bigram+BERT）。BERT のみの構成だけは
   β=2〜4 が最良で、bigram 採点の有無で最適 β が変わる。

### 5.1 SNS / 話し言葉ドメイン（国会会議録 2,000 文、v2 アセット）

`data/eval/sns2000.jsonl`（Lead 作成。国会会議録由来、5 行まとめて文脈付与、
ctx 付与率 80.7%・平均 45.2 字、Wikipedia 評価セットと重複なし）。

| 条件 | 文正解率 | CER(micro) | 境界F1 | acc@3 | 天井 | p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| beam 32/48, d=2.0, β=1.0 | 42.45% | 7.06% | 0.9815 | 49.40% | 50.75% | 11.2 / 177.5 |
| beam 128/256, gen=bigram, d=2.0, β=1.0 | **45.50%** | 6.48% | 0.9830 | 57.10% | 61.80% | 60.8 / 254.8 |
| beam 128/256, gen=unigram, bigram+BERT β=1.0 | 41.90% | 7.83% | 0.9747 | 57.20% | 69.65% | 44.4 / 232.7 |

- ラティス天井は **91.95%** と wiki（84.95%）より高い。gold span coverage も
  surface 99.41%。**辞書カバレッジは SNS の方が良い**のに、実力は 42.45% で
  wiki より **11.1pt 低い**。→ ドメイン差は辞書ではなく探索・再ランク側。
- **探索損失が大きい**: ラティス天井 91.95% → unigram 天井 57.10%（−34.85pt）。
  wiki では −14.25pt だったので、SNS はビーム 32 では天井が低すぎる。
  beam 128 で unigram 天井は 69.65% まで上がり、**最終精度も +3.05pt**
  （wiki の +1.80pt より大きい）。
- **bigram 誘導の損失も大きい**: bigram 天井 50.75%（32/48）・61.80%（128/256）
  に対し unigram 天井 57.10%・69.65%。差 **6.4〜7.9pt** は wiki（3.85pt）より大きい。
  Wikipedia コーパスで学習した現行 bigram が話し言葉に合っていない可能性があり、
  Phase 4 の bigram 再構築（SNS コーパス込み）で改善余地がある。
- **SNS では unigram 生成が悪化する**（45.50% → 41.90%、−3.6pt）。
  一方 acc@3 は 57.20% と同等で、**上位 3 本には正解が入るのに top-1 の
  並べ替えが合わない**。bigram 生成の N-best は bigram+BERT 再ランクと整合しており、
  wiki と逆の傾向。**「unigram 生成が有利」は wiki 限定**で、ドメインで
  使い分けるか、生成と再ランクの整合（bigram 項の扱い）を見直す必要がある。
- 文脈の寄与: 参考として、文単位（ctx なし）で作った別サンプルでは 25.25% だった。
  同一アセット・同一ビームで約 **−17pt**。文脈付き評価セットの採用は妥当。
- 境界 F1 は 0.98 と高く、SNS の誤りも「表記選択」が主。読み長 10〜13 字では
  69.6% だが、25 字以上では 33.5% まで落ちる（長文の 1 語誤り）。

## 6. レイテンシ（CPU int8 / 4 threads / 端末 proxy）

ベースライン（beam=32, want=48, d=2.0, β=1.0、wiki）の 1 文あたり内訳:

| フェーズ | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| ラティス構築 | 0.20 | 0.18 | 0.36 | 0.44 | 0.98 |
| ビーム探索（主 delta） | 20.78 | 5.92 | 172.73 | 190.04 | 213.72 |
| BERT 推論 | 2.45 | 2.27 | 4.01 | 4.73 | 10.06 |
| 再ランク | 0.22 | 0.18 | 0.46 | 0.56 | 2.70 |
| **合計** | **23.65** | **8.74** | **176.76** | **194.50** | **218.54** |

探索を広げたときの合計（wiki、同条件）:

| 設定 | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| bigram 32/48 | 8.7 | 176.8 | 194.5 | 23.7 |
| bigram 64/128 | 16.3 | 195.6 | 212.3 | 48.2 |
| unigram 64/128 + bigram | 15.2 | 198.3 | 216.3 | 47.8 |
| bigram 128/256 | 44.6 | 251.9 | 274.6 | 103.2 |
| unigram 128/256 + bigram | 36.0 | 240.7 | 257.2 | 97.5 |
| unigram 128/256 BERT のみ | 43.7 | 263.7 | 289.1 | 103.3 |

- **p50 は 2〜4 倍になるが p95 は 1.36 倍**。長文（読み 48 文字まで）の探索コストが
  すでに支配的で、分布が二峰性のため。
- BERT 推論は 2.3〜3.9ms p50 でほぼ一定。ボトルネックは常に探索側。
- **トレードオフ**: +1.85pt のために p50 4.1 倍・p95 1.36 倍。端末目標
  p95 < 20ms を Python の数字で判断すると、beam 32 でも長文で超過する
  （実機 Kotlin はこれより速い可能性が高いので、実機計測が必要）。
  速度制約が厳しければ **beam 64/128（+1.0〜1.2pt、p50 約 1.8 倍）** が妥協点。
- beam を増やす場合は、探索の打ち切り（未到達コスト下界による枝刈り）や
  ラティスの arc 削減など、アルゴリズム側の改善がセットで必要。

## 7. 判定と推奨

**判定: 探索律速寄り（辞書も無視できない）。**

| 要因 | 損失 | 根拠 |
|---|---|---|
| 辞書・候補カバレッジ | 15.05pt | ラティス天井 84.95%（gold span coverage 98.22%） |
| 探索（ビーム刈り） | 14.25pt | unigram 天井 70.70% vs ラティス天井 84.95% |
| bigram 誘導コスト | 3.85pt | bigram 天井 66.85% vs unigram 天井 70.70% |
| 再ランクの選択 | 13.30pt | 実力 53.55% vs bigram 天井 66.85% |

推奨（現行 v2 アセット。v3 アセットでも同じ CLI で再測定する）:

| 用途 | 推奨設定 | 文正解率 | p50 / p95 |
|---|---|---|---|
| 端末・速度優先（現行） | gen=bigram, beam=32, want=48, d=2.0, β=1.0 | 53.55% | 8.7 / 176.8 |
| 端末・バランス | gen=bigram, **beam=64, want=128**, d=2.0, β=1.0 | 54.55% | 16.3 / 195.6 |
| 高精度（wiki） | **gen=unigram, beam=128, want=256**, bigram+BERT, β=1.0 | 55.40% | 36.0 / 240.7 |
| 高精度（SNS） | **gen=bigram, beam=128, want=256**, d=2.0, β=1.0 | 45.50% | 60.8 / 254.8 |

- **β = 1.0**（bigram+BERT）。BERT のみの構成だけ β=2〜4。
- 生成方式はドメイン依存: wiki は unigram 生成が最良（+0.05pt・天井 +3.8pt）、
  SNS は bigram 生成が最良（+3.6pt）。SNS では生成と再ランクの整合が崩れると
  acc@3 は同じでも top-1 が落ちる。
- 辞書増強（Phase 1）の目標は `sentence_path_rate` 84.95% → 90%+。
  v3 辞書（costtuned）では **90.85%** を確認済み（§9）。
- 次のレバーは再ランク。beam 128/256 では天井 78.00% に対し実力 55.40%、
  acc@3 67.05%。**「top-3 には入るが top-1 で外す」11.65pt の解消**が最短。

## 8. 再現コマンド

```bash
cd /home/ryouta/products/Nayuta

# ベースライン（全指標 + JSON）
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
  --deltas 2.0 --betas 1.0,2.0,4.0,6.0 \
  --json-out logs/eval_v2_wiki_baseline.json

# Phase 0: 天井の beam/want スイープ（BERT なし・高速）
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 --sweep-only \
  --gen bigram --deltas 2.0 \
  --beam-sweep "32,48;64,128;128,256;32,256" \
  --json-out logs/phase0_ceiling.json

# (256,512) は 1,000 文で
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 1000 --sweep-only \
  --gen bigram --deltas 2.0 --beam-sweep "128,256;256,512" \
  --json-out logs/phase0_ceiling_256.json

# 探索を広げた最終精度（例: unigram 生成 + bigram+BERT 再ランク）
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
  --gen unigram --beam 128 --want 256 --deltas 2.0 --betas 1.0,2.0,4.0 \
  --json-out logs/phase0_search_unigram128_bigram.json

# SNS（v2 アセット）
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/sns2000.jsonl --domain sns --limit 2000 \
  --deltas 2.0 --betas 1.0,2.0,4.0,6.0 \
  --json-out logs/eval_v2_sns_baseline.json
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/sns2000.jsonl --domain sns --limit 2000 \
  --gen bigram --beam 128 --want 256 --deltas 2.0 --betas 1.0,2.0 \
  --json-out logs/eval_v2_sns_baseline_beam128.json

# 辞書カバレッジのみ（Phase 1 の効果確認用・数秒）
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
  --coverage-only --json-out logs/coverage_wiki.json
# v3 辞書のプレビュー
PYTHONPATH=. .venv/bin/python -m train.evaluate_v2 \
  --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 --coverage-only \
  --lexicon data/lexicon_v3_frozen.tsv --json-out logs/coverage_wiki_v3dict.json
```

## 9. 新アセット（v3）評価の準備

v3 辞書 `data/lexicon_v3_frozen.tsv`（costtuned, md5 `00a25a03...`）で
**ラティス天井 84.95% → 90.85%** を確認（同じ 2,000 文・同じ評価コード）。
gold span coverage は span 99.62% / surface 98.96% に改善。

| 指標 | v2 辞書 | v3 辞書（frozen, costtuned） |
|---|---|---|
| span_in_dict | 98.98% | 99.62% |
| surface_in_candidates | 98.22% | 98.96% |
| sentence_path_rate（ラティス天井） | 84.95% | **90.85%** |
| 欠落 span（no dict / 候補外 / 長すぎ） | 187 / 146 / 11 | 62 / 129 / 11 |

- 辞書の密度: 読み 144.6k → 500k、max_len 24 → 44、arc 数/位置
  13.4 → 16.1（+20%）。ラティス構築自体は 0.03→0.04ms で軽いが、
  **ビーム探索の展開数が増えるため v3 ではレイテンシが悪化する可能性**がある。
  最終評価では p50/p95 も併記して確認する。

新 bigram（`data/bigram_v3.tsv`）と新 ONNX（12 層 teacher → 6 層 distill、
`models/scorer_v3.pt` → int8 ONNX）が揃ったら、以下の 4 通りを
**ベースラインと同一条件**で測る（コマンドは `scripts/eval_v3_assets.sh`）:

```bash
bash scripts/eval_v3_assets.sh
# 引数でアセットを差し替え可:
#   bash scripts/eval_v3_assets.sh <lexicon> <bigram> <bigram_words> <onnx>
```

| # | データ | 設定 | 出力 |
|---|---|---|---|
| a | `data/eval/wiki2000.jsonl` | beam 32/48, gen=bigram, d=2.0, β=1.0 | `logs/eval_v3_wiki_base.json` |
| b | `data/eval/wiki2000.jsonl` | beam 128/256, gen=bigram, d=2.0, β=1.0 | `logs/eval_v3_wiki_beam128.json` |
| c | `data/eval/sns2000.jsonl` | beam 32/48, gen=bigram, d=2.0, β=1.0 | `logs/eval_v3_sns_base.json` |
| d | `data/eval/sns2000.jsonl` | beam 128/256, gen=bigram, d=2.0, β=1.0 | `logs/eval_v3_sns_beam128.json` |

各 JSON の `assets.*.sha256` に使用アセットの指紋が入る。比較基準は
v2 アセットの `logs/eval_v2_wiki_baseline.json` と
`logs/eval_v2_sns_baseline.json`（wiki 53.55% / SNS 42.45%）。

## 10. 未確定・次のアクション

- **SNS 評価セット**: Lead 作成の `data/eval/sns2000.jsonl` を正式採用
  （`train/make_eval_set.py` はコーパスから固定評価セットを作る汎用ツールとして残す）。
- **v3 アセットの最終評価**は Lead の「評価開始」後に §9 の 4 通りを実行。
  `--deltas` は生成にしか効かない（再ランクは観測済み連接のみ使う）ため、
  `--gen unigram` のときは指定しても結果は変わらない（`evaluate_bigram.py` と同じ式）。
- bigram を SNS コーパス込みで再構築したら、§5.1 の bigram 天井（61.80%）の改善を確認。
- (256,512) は 1,000 文のみ（JSON に `limit` を記録）。
- 探索の打ち切り・arc 削減は Phase 6（速度）側の検討課題。
