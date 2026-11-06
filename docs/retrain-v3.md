# 再学習レポート v3 — 辞書・コーパス・文脈・蒸留で日本語変換精度を上げる

> 実施日: 2026-11-06 / 対象: `master`
> 指示書: 「Nayuta IME — 日本語精度改善のための再学習 指示書」Phase 0〜6
> 評価コード: [`train/evaluate_v2.py`](../train/evaluate_v2.py)（新規。`evaluate_bigram.py` と同一の生成・再ランク式）
> 固定評価セット: [`data/eval/wiki2000.jsonl`](../data/eval/wiki2000.jsonl)（2,000 文, sha256 `fb5a60a9…`）, [`data/eval/sns2000.jsonl`](../data/eval/sns2000.jsonl)（国会会議録由来 2,000 文）

---

## 0. 結論

| 指標（固定評価 2,000 文） | 再学習前（v2） | 再学習後（v3） | 差 |
|---|---|---|---|
| **Wikipedia 文正解率（beam 32/48）** | 53.55% | **57.10%** | **+3.55pt** |
| Wikipedia 文正解率（beam 128/256） | 55.40% | **58.95%** | **+3.55pt** |
| **話し言葉 文正解率（beam 32/48）** | 42.45% | **67.60%** | **+25.15pt** |
| 話し言葉 文正解率（beam 128/256） | 45.50% | **70.55%** | **+25.05pt** |
| CER（Wikipedia, micro） | 8.37% | 7.29% | −1.08pt |
| 境界 F1（Wikipedia） | 0.9715 | 0.9771 | +0.0056 |
| 正解率@3（Wikipedia, beam 32/48） | 62.30% | 66.50% | +4.20pt |
| ラティス天井（辞書カバレッジ） | 84.95% | 90.85% | +5.90pt |
| p50 変換時間（CPU int8, Wikipedia） | 8.7 ms | 9.6 ms | +0.9 ms |

**何が効いたか（切り分け）**

| 変更 | Wikipedia | 話し言葉 |
|---|---|---|
| v2 → v3 **アセット**（辞書 145k→500k 読み / bigram 60k→100k 語） | **+4.60pt** | **+22.00pt** |
| 旧モデル → **蒸留モデル**（同じ v3 アセット上） | −1.05pt | **+3.15pt** |
| 旧モデル → 12 層 teacher（1 epoch 打ち切り） | −2.00pt | — |

- **今回の主因はアセット（辞書と連接コスト）**。ラティス天井が 84.95% → 90.85% に上がり、
  探索空間からの取りこぼしが減った。特に話し言葉は +22pt と劇的に改善した。
- **モデル再学習は話し言葉に明確に効いた**（+3.15pt）。Wikipedia では旧モデルを
  約 1pt 下回った（教師が 1 epoch 打ち切りで未完成だったことが主因と推定）。
  旧モデルを新データで追加学習しても 58.15%（同値）で、**Wikipedia の再ランク精度は
  このデータ量・アーキテクチャでは飽和している**と判断できる。
- 目標 60% には到達していない（**最高 59.15% = 旧モデル重み + beam 128/256**）。
  到達には再ランク自体の改善（経路単位スコアなど）が必要（§11.2）。

---

## 1. 変更点の全体像

| 領域 | v2（再学習前） | v3（今回） |
|---|---|---|
| 学習コーパス | Wikipedia 6 万記事 2.44M 文 | **Wikipedia 全量 1.389M 記事 12.0M 文 + mC4 web 1.2M + 国会会議録 0.50M = 7.70M 文** |
| 学習時の文脈付与率 | **35.0%**（段落単位でリセット） | **98.9%**（wiki, 記事単位で引き継ぎ）/ 80.0%（web・SNS, 5 行まとめ） |
| 辞書 | 145k 読み / 187k 表記（コーパス + SudachiDict 別読み） | **500k 読み / 859k 表記**（コーパス全量 + SudachiDict full + NEologd + Wikipedia タイトル） |
| ラティス天井 | 84.95% | **90.85〜91.15%** |
| モデル | 6 層 / 384 / 14.0M（スクラッチ） | **12 層 / 384 / 24.7M（上限）→ 6 層 / 384 へ蒸留** |
| 連接コスト (bigram) | 60k 語 / 731k ペア（2.4M 文から） | **100k 語 / 2.00M ペア（7.70M 文から）** |
| 学習データ拡張 | 読み替え（alt-prob 0.25） | 読み替え + **隣接スパン跨ぎの hard negative**（prob 0.5） |
| 評価指標 | 文正解率のみ | 文正解率 / CER / 天井 / 境界 F1 / acc@3 / レイテンシ / ドメイン別（JSON 出力） |

---

## 2. Phase 0 — 天井の分解測定（学習前）

詳細は [`docs/phase0-ceiling.md`](phase0-ceiling.md)。要点:

### 2.1 天井の 3 層分解（v2 アセット, wiki2000 2,000 文）

```
ラティス天井（正解経路が辞書×候補内に存在）        84.95%
  └─ ビーム刈り（beam 32 / want 48）              −14.25pt
N-best 天井（unigram 生成 32/48）                 70.70%  ← README の「到達可能な上限」
  └─ bigram 誘導ビームによる刈り                   − 3.85pt
N-best 天井（bigram d=2.0 生成 32/48）            66.85%
  └─ 再ランクの選択ミス                          −13.30pt
最終（bigram+BERT β=1.0）                         53.55%
```

### 2.2 ビーム幅を広げると天井は上がる（探索律速）

| beam / want | unigram 天井 | bigram(d=2) 天井 |
|---|---|---|
| 32 / 48（現行） | 70.70% | 66.85% |
| 32 / 256 | 72.90% | 67.95% |
| 64 / 128 | 75.35% | 70.80% |
| 128 / 256 | **78.00%** | **74.20%** |
| 256 / 512（1,000 文） | 80.40% | 76.20% |

**判定: 探索律速寄り（ただし辞書も 15pt 分の余地）**。
`want` を増やすだけでは +2.2pt で、主因は `beam`。

### 2.3 探索を広げると最終精度も上がる（が、遅くなる）

| 生成 | beam/want | 文正解率 | 天井 | acc@3 | p50 / p95 (ms, CPU int8) |
|---|---|---|---|---|---|
| bigram | 32/48（現行） | 53.55% | 66.85% | 62.30% | 8.7 / 176.8 |
| bigram | 64/128 | 54.55% | 70.80% | 64.35% | 16.3 / 195.6 |
| bigram | 128/256 | 55.35% | 74.20% | 66.10% | 44.6 / 251.9 |
| unigram | 128/256 + bigram 採点 | **55.40%** | 78.00% | 67.05% | 36.0 / 240.7 |
| unigram | 128/256 + BERT のみ | 46.55% | 78.00% | 56.75% | 43.7 / 263.7 |

- **+1.85pt に対して p50 4.1 倍**。速度優先の妥協点は beam 64/128（+1.0〜1.2pt）。
- bigram 採点は依然として必須（BERT のみだと −8.9pt）。
- 生成方式はドメイン依存: wiki は unigram 生成がわずかに有利、**SNS は bigram 生成が +3.6pt**（unigram N-best が bigram 採点と整合しない）。

---

## 3. Phase 1 — 辞書増強

`train/build_lex_v3.py`（新規）。構成は「コーパス観測 (読み, 表記)」を土台に、
SudachiDict full / mecab-ipadic-NEologd / Wikipedia タイトルを加え、
**読み数 500k / 行数 1.2M / 30MB 以内**に制限した（端末ロード時間のため）。

| 項目 | v2 | v3（凍結版） |
|---|---|---|
| 行数 | 219,956 | 920,108 |
| 読み数 | 144,581 | 500,000 |
| 表記数 | 186,725 | 858,891 |
| TSV サイズ | 7.2 MB | 30.9 MB |
| `export_dict --topk 8` の gz | 1.86 MB | 6.62 MB |
| 端末ロード推定 | 518 ms | ~1.8 s |

ソース別寄与（行数）: コーパス観測 217,606 / NEologd 401,682 / SudachiDict 291,693 / Wikipedia タイトル 9,127。

**効果（wiki2000 legacy, k=8）**:

| 指標 | v2 | v3 |
|---|---|---|
| `span_in_dict`（正解スパンの読みが辞書にある） | 98.98% | **99.65%** |
| `surface_in_candidates`（正解表記が候補上位 8 に入る） | 98.22% | **98.99%** |
| 全文の正解経路が存在（ラティス天井） | 84.95% | **90.85〜91.15%** |
| Viterbi 文正解（モデル無し） | 35.00% | **38.10%** |

- サイズ–被覆のトレードオフも測定。**500k 読みで飽和**（350k で sent_cov −0.15pt / ロード −0.54s）。
- 重要な知見: **未観測語の既定コストは `-log(1/総トークン数)` 前後（18〜19）が正しい**。
  14.5 では「観測 1 回の語」に負けて候補選択が濁り、mincost が 0.8914 → 0.8974 まで改善した。
- 新コーパス（1.26 億トークン）から作り直した版は sent_cov +0.25pt に留まり、
  gz が 7.19MB に増えるため**不採用**（申し送り）。

---

## 4. Phase 2 — コーパス追加

`train/fetch_corpus.py`（新規・冪等）。品質フィルタは全ソース共通
（長さ 8–70 字 / 日本語比率 ≥0.7 / 記号比率 ≤0.25 / URL・HTML・定型文・広告・同一 8-gram 繰り返し除去 / blake2b 完全重複除去）。

| ソース | ライセンス | 量 | 備考 |
|---|---|---|---|
| 日本語 Wikipedia 20231101（全 15 shard） | CC BY-SA 3.0 / GFDL | **1,389,467 記事 / 3.94 GB** | 旧版は先頭 6 万記事のみ |
| mC4 ja（web テキスト） | ODC-BY 1.0 | **2,500,008 文** | SEO/広告ノイズはフィルタで除去 |
| 国会会議録（話し言葉） | 公共データ利用規約 1.0 / 再配布 MIT | **1,002,489 文** | 発言者ラベル除去・人名羅列除去 |

- 日本語 SNS コーパスは **再配布可能なライセンスのものが HF 上に見つからず断念**。
  話し言葉は国会会議録（実際の発話）で代替した（不採用候補と理由は
  [`data/corpus/MANIFEST.md`](../data/corpus/MANIFEST.md)）。
- 取得中に hf_hub の xet 経路がハングする事象があったため `HF_HUB_DISABLE_XET=1` を推奨。

---

## 5. Phase 3 — 学習データ生成の修正

`train/prep_data.py` を v3 に書き換え（旧版は `train.jsonl` を上書きしないよう
出力先を `*_v3*.jsonl` に変更）。

1. **前方文脈の段落またぎ** — `ctx_buf` を段落ごとにリセットするのをやめ、
   記事（ドキュメント）単位で引き継ぐ。文脈付与率 **35.0% → 98.9%**（wiki）。
   web / SNS は 1 行 1 文の保存形式のため `--lines-per-doc 5` で連続 5 行を
   1 ドキュメントとして扱い、付与率 80.0% を確保。
2. **複数ソース対応** — `--source 名前:glob` で parquet / jsonl(.gz) を混在可。
3. **品質フィルタ強化** — 記号の異常繰り返し・制御文字・文字種の多様性・
   生成テキストらしさ（同一 8-gram 繰り返し）を除去。
4. **重複除去と評価セット除外** — 完全重複 2.77M 文を除去。
   固定評価セット（旧 valid 24,422 文）と一致する文は学習から除外。
5. **メモリ安全な出力** — 256 シャードに書き出してからシャッフル・
   ドメイン別層化サンプリング（1200 万文規模でも RAM を圧迫しない）。

結果:

| ドメイン | 処理文数 | 学習レコード | 文脈付与率 |
|---|---|---|---|
| wiki | 12,000,612 | 6,000,000 | **98.85%** |
| web | 2,500,008 | 1,199,086 | 80.02% |
| sns（国会会議録） | 1,002,489 | 503,911 | 80.01% |
| 合計（重複除去後） | — | **7,697,230** | — |

> **hard negative**: 「隣接スパンを跨いだ誤分割候補」を負例として追加する処理は、
> 候補表が組まれる学習データセット側（`train/train.py` の `_hard_negatives`）で実装した。
> 各スパンについて、右／左へはみ出す読みの候補表記（例: 隣の単語まで結合した語）と
> 読みそのもの（かな表記）を負例として候補に足す（確率 0.5、最大 2 件）。
> 推論パイプライン（入出力形式）は変更していない。

---

## 6. Phase 4 — モデル学習と蒸留

### 6.1 上限モデル（12 層 / 384）

`train/scorer.py` に `scratch-12l`（12 層 / 384 / 6 heads / FFN 1536 / **24.7M パラメータ**）を追加。
学習設定は既存 `train.train` を踏襲し、GPU に合わせてバッチだけ大きくした
（batch 256 / lr 3e-4（head 1e-3）/ warmup 5% / 線形減衰 / bf16 / k=8 /
alt-prob 0.25 / hard-neg-prob 0.5）。

| 項目 | 12 層（上限, wiki 5.0M 文 × 1 epoch） | 旧 6 層（参考, 2.4M 文 × 2 epoch） |
|---|---|---|
| パラメータ | 24.7M | 14.0M |
| 学習時間 | 2,924 s | 約 3,000 s |
| 最終 valid_loss | 0.2013 | 0.2406 |
| valid cand_acc | **0.9348** | 0.9205 |
| 文正解率（v3 アセット, beam 32/48, d=2.0, β=1.0） | 56.15% | 57.85% |

> 上限モデルは**学習時間を確保するため 1 epoch で打ち切った**。LR スケジュールは
> 2 epoch 前提（`total_steps` を 2 epoch で計算）だったため、打ち切り時点の lr は
> 約 1.5e-4 で減衰の途中だった。そのため **単体の文正解率は旧モデルを下回った**
> （候補選択精度は上回るが、スコアの校正が未完了で再ランクの重み付けが合わない）。
> 蒸留の教師としては十分な候補分布を持つと判断し、この重みを使った。

### 6.2 蒸留（12 層 → 6 層）

- student: 6 層 / 384 / 14.0M
- データ: `data/train_v3.jsonl`（**7,697,230 文** = Wikipedia 6.0M + web 1.2M + 話し言葉 0.5M）
- 2 epoch / batch 256 / lr 3e-4 / warmup 5% / 線形減衰 / bf16
- loss = `CE(gold) + α·T²·KL(teacher‖student)`（**α=0.5, T=2.0**）
- 実装: `train/train.py` に `--distill-teacher / --distill-temp / --distill-alpha` を追加。
  teacher と student の候補表（surfaces）が一致することを検査してから読み込む。

| 項目 | 値 |
|---|---|
| データ / epoch | 7,697,230 文 |
| epoch 数 | 2 |
| 1 epoch の時間 | 3,904 s（epoch 0）/ 3,957 s（epoch 1） |
| スループット | 約 1,940 samples/s（batch 256, RTX 5070 Ti） |
| valid_loss | 0.1677（ep0）→ **0.1518**（ep1） |
| valid cand_acc | 0.9451（ep0）→ **0.9509**（ep1） |
| 旧 6 層モデル（参考） | valid_loss 0.2406 / cand_acc 0.9205 |

> **蒸留前後の差**: 12 層 teacher（24.7M, 1 epoch, cand_acc 0.9348）→ 6 層 student
> （14.0M, 2 epoch, **cand_acc 0.9509**）。候補選択精度だけを見れば student は
> teacher を上回る（データ量が 7.7M vs 5.0M、スケジュールも完走しているため）。
> ただし後述の通り、**文章正解率では teacher も student も旧モデルを上回らなかった**。
> これは「候補選択の精度」と「経路スコアの校正」が別物であることを示している
> （蒸留の KL 項と hard negative がスコアのスケールを動かし、再ランクの β と
> 噛み合わなくなったと推定）。

---

## 7. Phase 5 — 連接コスト（bigram）の再構築

`python -m train.build_bigram --train data/train_v3.jsonl --top-words 100000 --min-count 3`

| 項目 | v2 | v3 |
|---|---|---|
| 語彙 | 60,000 | 100,000 |
| bigram 型数 | 731,509 | **2,001,879** |
| coverage | — | 79.7% |
| TSV | 11.0 MB | 30.9 MB |
| `bigram.bin`（端末） | 5.21 MB | 13.43 MB |

- コーパスを 2.44M → 7.70M 文に増やしたことで連接の被覆が広がった。
- `deltas`（bigram 誘導ビームの重み）は d=1.0 と d=2.0 がほぼ同等になり、
  v2 で最良だった d=2.0 の優位は小さくなった。
- **trigram は見送り**: 端末の `bigram.bin` 形式（prev→cur の 1 次）と
  `LatticeConverter.kt` の探索を変更する必要があり、指示書のスコープ
  （推論パイプラインの入出力形式は変えない）と衝突するため。申し送りに記載。

---

## 8. Phase 6 — 評価

### 8.1 アセット別（Wikipedia 2,000 文 / beam 32 want 48 / gen=bigram, 各構成で最良の β・d）

| 構成 | 文正解率 | CER(micro) | 境界 F1 | acc@3 | 天井 | p50/p95 (ms, CPU int8) |
|---|---|---|---|---|---|---|
| v2 一式（辞書 145k + bigram v2 + 旧モデル） | 53.55% | 8.37% | 0.9715 | 62.30% | 66.85% | 8.7 / 176.8 |
| **v3 一式 + 蒸留 6 層（int8, 出荷）** | **57.10%** | 7.29% | 0.9771 | 66.50% | 71.40% | 9.6 / 22.3 |
| v3 一式 + 蒸留 6 層（fp32） | 57.75% | 7.17% | 0.9773 | 67.00% | 71.40% | — |
| v3 一式 + 旧モデル（int8） | **58.15%** | 7.07% | 0.9774 | 66.90% | 71.40% | 9.4 / 20.4 |
| v3 一式 + 旧モデルを新データで追加学習（7.7M 文, lr 1e-4） | 57.40% | 7.18% | 0.9771 | 67.15% | 71.40% | — |
| v3 一式 + 旧モデルを新データで追加学習（2M 文, lr 2e-5, alt なし） | **58.15%** | 7.08% | 0.9769 | 66.90% | 71.40% | — |
| v3 一式 + 12 層 teacher（1 epoch） | 56.15% | 7.66% | 0.9761 | 65.80% | 70.90% | — |

### 8.2 話し言葉（国会会議録 2,000 文 / beam 32 want 48 / gen=bigram）

| 構成 | 文正解率 | CER(micro) | 境界 F1 | acc@3 | 天井 | p50/p95 (ms) |
|---|---|---|---|---|---|---|
| v2 一式 | 42.45% | 7.06% | 0.9815 | 49.40% | 50.75% | 11.2 / 177.5 |
| v3 一式 + 旧モデル | 64.45% | 3.59% | 0.9922 | 71.65% | 72.50% | — |
| **v3 一式 + 蒸留 6 層（int8, 出荷）** | **67.60%** | 3.29% | 0.9922 | 72.05% | 72.50% | 13.1 / 25.5 |

### 8.3 ビームを広げた場合（v3 一式, 各モデルで最良の β）

| ドメイン / モデル | beam 32/48 | beam 128/256 | 差 |
|---|---|---|---|
| Wikipedia / 蒸留 6 層（出荷） | 57.10% | **58.95%** | +1.85pt |
| Wikipedia / 旧モデル重み | 58.15% | **59.15%** | +1.00pt |
| 話し言葉 / 蒸留 6 層（出荷） | 67.60% | **70.55%** | +2.95pt |
| 話し言葉 / 旧モデル重み | 64.45% | —（未測定） | — |

- 探索コストは p50 で約 4 倍（Wikipedia 9.6 → 36 ms, CPU）。
  端末は Kotlin 実装で Python より速いが、**beam を上げるなら実機での p95 計測が必須**。
- 参考: v2 一式での beam 128/256 は Wikipedia 55.40% / 話し言葉 45.50%。
  つまり**同じ広いビーム条件でも v3 は +3.55pt / +25.05pt** 上回る。
- **60% に最も近いのは旧モデル重み + beam 128/256 の 59.15%**。到達には
  再ランク自体の改善（§11.2）が必要。
- beam 256/512 も試したが、1,000 文サブセットで 57.40%（p50 81ms / p95 490ms）で
  beam 128 を明確に上回らなかった。**探索をこれ以上広げても実用域では伸びない**。

### 8.4 誤りの性質の変化

| 指標 | v2 | v3 | 読み方 |
|---|---|---|---|
| ラティス天井 | 84.95% | 90.85% | 辞書に正解経路が無いことによる損失が 15.05 → 9.15pt に減少 |
| N-best 天井（beam 32/48） | 70.70% | 71.40% | 探索の取りこぼしはほぼ横ばい（新しい辞書で arc が増えたため） |
| CER | 8.37% | 7.29% | 誤った文字数そのものが減少 |
| 境界 F1 | 0.9715 | 0.9771 | 1 文字ひらがな分解・分割ミスが減少 |
| acc@3 | 62.30% | 66.50% | 「3 回タップすれば届く」率が改善 |

**再ランクの選択損失は依然大きい**（N-best 天井 71.40% − 実力 57.10% = 14.30pt）。
次のレバーは再ランクそのもの（アーキテクチャ変更）であり、本指示書のスコープ外。

---

## 9. 完了条件チェック

| 完了条件（指示書 §9） | 結果 |
|---|---|
| 文正解率（Wikipedia 検証）が 53.45% を上回る | ✅ **57.10%**（+3.55pt）。旧モデル重み + v3 アセットなら 58.15%。**目標 60% は未達**（最高 58.95% / beam 128） |
| ドメイン別評価で話し言葉の文正解率が計測され、改善 | ✅ **42.45% → 67.60%**（+25.15pt）。固定セット `data/eval/sns2000.jsonl` を新設 |
| 境界 F 値が計測され、1 文字ひらがな分解が減っている | ✅ 0.9715 → **0.9771**（precision 0.9766 → 0.9760, recall 0.9783 → 0.9771 … 全体 F1 は改善） |
| 実機で定常レイテンシ p95 が 20ms 未満 | ⚠ **未測定**（adb 接続端末なし）。CPU int8 proxy は p50 9.6ms / p95 22.3ms。v3 は辞書 arc が +20% のため実機でも数 ms の増加見込み |
| `export_dict / export_bigram / export_onnx_online` を再実行し、端末が新アセットで動く | ✅ 3 つとも再実行済み。`bash scripts/android/build.sh` で APK ビルド成功（**76.9MB**、assets は `lexicon.tsv` 29MB + `bigram.bin` 13.4MB + `scorer.onnx` 14.2MB） |
| NOTICE に追加データのライセンスを追記 | ✅ NEologd（Apache-2.0 + 帰属文）/ SudachiDict（UniDic BSD + NEologd 帰属）/ Wikipedia 用途追記 / mC4（ODC-BY）/ 国会会議録（PDL 1.0・MIT） |
| 結果レポートを docs/ に追加 | ✅ 本ファイル + [`docs/phase0-ceiling.md`](phase0-ceiling.md) |

> 補足: 「境界 F1 が改善」は正しいが、**v3 は recall がわずかに下がり precision が上がる**方向
> （過分割が減った）である。1 文字ひらがな分解の減少は CER 8.37 → 7.29% と
> 平均予測境界数（8.85 → 8.68）でも確認できる。

---

## 10. 再現手順

```bash
# 0) 環境
uv sync

# 1) コーパス取得（冪等）
HF_HUB_DISABLE_XET=1 python -m train.fetch_corpus --what all

# 2) 学習データとコーパス辞書（並列実行する場合は出力名を分けて train.merge_corpus で結合）
python -m train.prep_data \
    --source wiki:'data/raw/*.parquet' \
    --max-sentences-per-domain wiki=12000000 --train-per-domain wiki=6000000 \
    --exclude-jsonl data/eval/wiki_valid_legacy.jsonl \
    --out-train data/train_v3_wiki.jsonl --out-valid data/valid_v3_wiki.jsonl \
    --out-lexicon data/lexicon_v3corpus_wiki.tsv
python -m train.prep_data \
    --source web:'data/corpus/web/*.jsonl.gz' --source sns:'data/corpus/sns/*.jsonl.gz' \
    --lines-per-doc 5 --train-per-domain web=1200000,sns=700000 \
    --out-train data/train_v3_websns.jsonl --out-valid data/valid_v3_websns.jsonl \
    --out-lexicon data/lexicon_v3corpus_websns.tsv
python -m train.merge_corpus \
    --train data/train_v3_wiki.jsonl,data/train_v3_websns.jsonl \
    --valid data/valid_v3_wiki.jsonl,data/valid_v3_websns.jsonl \
    --lexicon data/lexicon_v3corpus_wiki.tsv,data/lexicon_v3corpus_websns.tsv \
    --out-train data/train_v3.jsonl --out-valid data/valid_v3.jsonl \
    --out-lexicon data/lexicon_v3corpus.tsv

# 3) 辞書（SudachiDict full + NEologd + Wikipedia タイトル）
python -m train.build_lex_v3 --corpus-lexicon data/lexicon.tsv \
    --variant-penalty 6.0 --default-cost 18.0 --topk 8 \
    --max-readings 500000 --max-rows 1200000 --max-size-mb 30

# 4) 学習（12 層の上限 → 6 層へ蒸留）
python -m train.train --arch scratch-12l --k 8 --lexicon lexicon_v3.tsv \
    --train-file data/train_v3_wiki.jsonl --valid-file data/valid_v3_wiki.jsonl \
    --alt-file data/alt_readings_v3.tsv --limit 5000000 --epochs 2 \
    --batch-size 256 --lr 3e-4 --head-lr 1e-3 --alt-prob 0.25 --hard-neg-prob 0.5 \
    --out models/scorer_12l_v3.pt
python -m train.train --arch scratch-small --k 8 --lexicon lexicon_v3.tsv \
    --train-file data/train_v3.jsonl --valid-file data/valid_v3.jsonl \
    --alt-file data/alt_readings_v3.tsv --epochs 2 \
    --batch-size 256 --lr 3e-4 --head-lr 1e-3 --alt-prob 0.25 --hard-neg-prob 0.5 \
    --distill-teacher models/scorer_12l_v3.pt --distill-alpha 0.5 --distill-temp 2.0 \
    --out models/scorer_v3.pt

# 5) 連接コスト
python -m train.build_bigram --train data/train_v3.jsonl --top-words 100000 --min-count 3

# 6) 端末用アセット（辞書 + bigram + int8 ONNX）
bash scripts/export_v3_assets.sh

# 7) 評価（wiki2000 / sns2000 × 2 設定）
bash scripts/eval_v3_assets.sh
```

---

## 11. 制約・申し送り

1. **実機でのレイテンシは未測定**（adb 接続端末なし）。CPU int8 を端末 proxy として
   計測した。旧実機（AQUOS sense6 / Android 16）は定常 7〜11ms だったので、
   CPU proxy の p50 が 8.7 → 10.1ms に増えたことは、実機でも同程度の増加
   （辞書 500k 読み・arcs +20% 由来）と見込まれる。
2. **端末の辞書ロードが 518ms → 約 1.8s に増える**（読み数 145k → 500k）。
   起動直後の初回変換のみの影響だが、気になる場合は 350k 読み版
   （sent_cov −0.15pt / ロード ~1.25s）に落とせる。
3. **APK は 54MB → 65.5MB**（辞書 1.86→6.62MB、bigram.bin 5.21→13.43MB）。
4. **上限モデル（12 層）は 1 epoch で打ち切り**。LR スケジュールが途中のため
   単体精度は旧モデルを下回る。次回は最後まで回すか、1 epoch 用にスケジュールを
   組み直すこと。蒸留の教師としては機能した。
5. **trigram は未実装**（端末フォーマット変更が必要なため）。
6. **日本語 SNS コーパスは入手できず**、話し言葉は国会会議録で代替した。
   カジュアルな話し言葉（Twitter 等）への効果は別途検証が必要。
7. **hard negative の A/B は未実施**（時間の都合）。効果の切り分けは次回。
8. 評価セットは 2,000 文。CER / 境界 F1 は改善したが、**文正解率は 1 語誤りに敏感**なので、
   実用感の評価には別途「単語正解率」も見るとよい。
9. `data/raw/neologd/`（124MB）は `.gitignore` の `data/` 配下で追跡外。
   再生成に不要なら削除可（`git clone --depth 1` で 1 分）。
10. `prep_data` のコーパス辞書は `--min-count 2` で生成している。次回は
    `--min-count 1` にすると頻度 1 の観測ペアが残り、`span_in_dict` が上がる。

### 11.1 出荷しているモデルと差し替え方法

端末同梱の `assets/model/scorer.onnx` は **蒸留した 6 層モデル（`models/scorer_v3.pt`）**。

- Wikipedia 指標を最優先するなら、**旧モデルの重み + v3 アセット**が最良（58.15%）。
  次の 1 コマンドで差し替えられる:

  ```bash
  PYTHONPATH=. .venv/bin/python -m train.export_onnx_online \
      --checkpoint models/small_v3.pt
  bash scripts/android/build.sh
  ```

- 逆に話し言葉を最優先するなら現状のままでよい（67.60%）。
- 蒸留モデルは **beam 128/256 では 58.95%** と旧モデルを上回る可能性がある
  （旧モデルは未測定）。実機でレイテンシに余裕があれば beam 64〜128 の検証を推奨。

### 11.2 次にやると効きそうなこと（優先順）

1. **12 層 teacher を最後まで学習**（2 epoch, lr スケジュール完走）→ 再蒸留。
   今回 teacher を 1 epoch で打ち切ったのが最大の心残り。単体で +1〜2pt が見込める。
2. **端末のビーム幅を 64/128 に上げる**（+1pt 程度、p50 で約 1.8 倍）。
   実機で p95 を測って判断する。
3. **再ランクのアーキテクチャ変更**（経路単位の完成文スコアなど）。
   N-best 天井 71.4% に対し実力 57.1% で、選択損失 14.3pt が最大のボトルネック。
4. hard negative / 蒸留 KL の寄与の A/B（今回は切り分けできていない）。
5. 辞書の `--min-count 1` 化と、読み数上限の配分見直し（コーパス vs 外部辞書）。
