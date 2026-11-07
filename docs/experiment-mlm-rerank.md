# 「する」接続の分解と MLM 再ランク実験（P0 / P2）

> 実施日: 2026-11-07 / 状態: 実験（端末実装なし）

## 1. 目的

`帰社する` を `記者する` と変換してしまうように、連接の不自然さを判定できない問題を分解する。

- 探索（N-best）・再ランク・辞書 prior のどこで落ちているか
- 文レベル LM（事前学習 char BERT の MLM 疑似尤度）を足すと効くか

## 2. 評価セット（P0）

`train/make_suru_eval.py` で `data/valid_v3.jsonl` から生成する。

```bash
python -m train.make_suru_eval --limit 300
# => data/eval/suru_amb.jsonl （valid 297 文 + 手作り 3 文）
```

抽出条件:

- 直後が `する` の活用形（する / し / さ / せ / す / しろ / しよう）
- その名詞（正解表記）の Sudachi POS が「サ変可能」
- 同じ読みの候補（k=8）に「サ変可能でない表記」が入り、コスト差が 12 nat 以内
  （`記者 / 帰社`、`内臓 / 内蔵` のような競合ケース）

手作り 3 文（コーパスに gold が無い語の確認用）: `きしゃする`、`きしゃし`、`ないぞうする`。

## 3. 現状の分解（300 文）

| 指標 | 値 |
|---|---|
| 文正解率（bigram 生成 + BERT, β=0.5） | **46.67%** (140/300) |
| 同（BERT なし = 探索 1 位） | 46.00% |
| ラティス天井（正解経路が辞書内に存在） | 86.67% (260/300) |
| N-best oracle: unigram / bigram / union | **62.33% / 51.33% / 70.67%** |
| 候補上位 16 内の gold 順位 | 1位=139, 2–4位=10, 5–8位=3, 圏外=148 |

- 誤りの主因は再ランクの順位ではなく**候補生成（到達率）**。天井 86.7% に対して N-best は 51.3%。
- このセットでは **bigram 生成が unigram 生成より到達率を下げている**（51.3% vs 62.3%）。
- 上位 16 に入った正解のうち 139/152 はすでに 1 位。並べ替えで直せる余地は小さい。

## 4. MLM 再ランク（P2）

`train/experiment_mlm_rerank.py`。上位 K 仮説の**表記列**を、事前学習 char BERT
（`tohoku-nlp/bert-base-japanese-char-v3`、HF キャッシュ）の MLM 疑似尤度
（1 文字ずつマスクして logP を合計）で再ランクする。

```bash
python -m train.experiment_mlm_rerank --data data/eval/suru_amb.jsonl \
    --limit 300 --top-k 16 --gammas 0.05,0.1,0.2,0.4,0.8,1.5
```

| データ | n | baseline | 最良 γ | 再ランク後 | fixed / broken |
|---|---|---|---|---|---|
| suru_amb（top-16） | 300 | 46.33% | 0.4 | 47.33% | 5 / 2 |
| wiki2000 先頭（top-8） | 200 | 52.00% | 0.8 | 56.50% | 15 / 6 |
| wiki2000 201–700（held-out, top-8） | 500 | 55.60% | 0.8 | **61.60%** | 41 / 11 |

- **Wikipedia では明確に効く**（held-out で +6.0pt）。文レベル LM の信号自体は有効。
- 注意: 事前学習モデルは日本語 Wikipedia で学習済み。wiki2000 は Wikipedia 由来なので、
  この +6pt には事前学習データとの重複（リーク）が含まれる可能性がある。
  あくまで「蒸留で取りに行ける上限」として読む。
- ただし 110M パラメータ × 1 文字ごとの forward は端末では非現実的。
  この数字は「蒸留して取りに行ける headroom」として読む。
- `する` 接続セットでは +1pt 止まり。理由は 2 つ:
  1. 正解が上位 16 に入らないケースが約半数（到達率の問題）
  2. char MLM は `帰社する` を `記者する` より好む（PLL −20.3 vs −25.2）が、
     `騎射する`（−21.6）との差は 1.3 nat しかなく、辞書コスト差 5.3 nat を覆せない。
     稀語の文法性判定は文字 LM では弱い
- 手作りケース `きしゃする`: `帰社する` は 47 候補中 **25 位**。γ=2 で 2 位まで上がるが
  `騎射する` に負けたまま。**この1件は MLM 再ランクでは直らない**。

## 5. 予備: union N-best で到達率を上げる

`train/experiment_union_nbest.py`。unigram ビーム（端末の現行生成）と bigram ビームを
両方回し、表記で重複除去してマージする。

| データ | oracle unigram | bigram | union | sent_acc bigram+BERT | union+BERT |
|---|---|---|---|---|---|
| wiki2000 | 74.55% | 71.40% | **80.40%** | 57.05% | 56.30% |
| suru_amb 300 | 62.33% | 51.33% | **70.67%** | 46.67% | 51.33% |

- 到達率は大きく上がる（wiki +9.0pt / suru +19.3pt）。探索を 2 回回すだけで、学習は不要。
- ただし現行の再ランカでは wiki の文正解率は **−0.75pt**。候補を増やすだけでは活かせない。
  suru では +4.7pt。
- → 次の本命は**経路単位の再ランク学習**（union の候補を負例にした listwise / pairwise）。

## 6. 辞書 prior の予備

未観測語（coverage 0）のコストを 18.0 → 15.0 に丸めた辞書で比較。

| データ | frozen | cap15 |
|---|---|---|
| suru_amb 300 | 46.67% | 48.00% |
| wiki2000 先頭 1000 | 55.20% | 54.10% |

一律のコスト下げはトレードオフ。新コーパス辞書からの再生成（`帰社` は 19 回観測）と
bigram の語彙・バックオフ改善は、別途アセット側で検証する。

## 7. 結論

1. `記者する` の失敗は 1 つの原因ではなく、
   **到達率（25 位/47、上位16圏外 49%）→ 再ランク → char LM の文法判定力**の順に重なっている。
2. 文レベル LM は Wikipedia で +6pt の headroom がある。ただし蒸留前提。
3. union 生成は到達率を +9〜19pt 上げるが、現行再ランカでは wiki で活かせない。
4. 次の順序:
   - union 生成を評価系に組み込み、経路単位の listwise 再ランクを学習（gold vs union 負例）
   - MLM の文レベル信号を小さいモデルへ蒸留（14M のスパン採点器に追加特徴として統合）
   - `suru_amb` を回帰セットとして残す

## 8. 再現

```bash
# 評価セット生成（P0）
python -m train.make_suru_eval --limit 300

# 現状の精度（評価ハーネス）
python -m train.evaluate_v2 --data data/eval/suru_amb.jsonl --domain suru --limit 300 \
    --lexicon data/lexicon_v3_frozen.tsv --bigram data/bigram_v3.tsv \
    --bigram-words data/bigram_words_v3.tsv \
    --onnx android/app/src/main/assets/model/scorer.onnx \
    --gen bigram --beam 32 --want 48 --deltas 1.0 --betas 0.5

# MLM 再ランク（P2）
python -m train.experiment_mlm_rerank --data data/eval/wiki2000.jsonl \
    --skip 200 --limit 500 --top-k 8 --gammas 0.05,0.1,0.2,0.4,0.8,1.5

# union N-best
python -m train.experiment_union_nbest --data data/eval/wiki2000.jsonl --limit 2000
```
