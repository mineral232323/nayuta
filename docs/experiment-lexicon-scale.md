# 辞書規模の実験 — 増やすのに「学習」は必要か

> 実施日: 2026-11-07 / ブランチ: `experiment/lexicon-scale` / 状態: 実験（アセット未差し替え）

## 1. 目的

辞書を大きくするのにモデルの学習が必要かを確かめる。あわせて、効くのは
**読み数 / 行数（候補の深さ）/ 頻度の再集計** のどれかを見る。

前回の調査で、現行辞書 `data/lexicon_v3_frozen.tsv` は

- 読み 500,000 / 行 920,108 のうち、**コーパス頻度が付いているのは 217,606 行（23.6%）**
- 未観測（freq 0 / cost 18.00）が 702,502 行（76.4%）

であることが分かっている。また出荷辞書は旧コーパス（2,368 万トークン）から作られており、
新コーパス辞書 `data/lexicon_v3corpus.tsv`（533,901 行 / 382,218 読み）の頻度が入っていない。

## 2. 実験条件

`train/build_lex_v3.py` で 4 種を生成（すべて topk=8）。

| 名前 | corpus lexicon | max-readings | max-rows | max-size |
|---|---|---|---|---|
| frozen（既存） | `data/lexicon.tsv`（旧） | 500,000 | 1,200,000 | 30MB |
| old500k2m | 旧 | 500,000 | 2,000,000 | 60MB |
| old1m | 旧 | 1,000,000 | 3,000,000 | 60MB |
| new500k | `data/lexicon_v3corpus.tsv`（新） | 500,000 | 2,000,000 | 45MB |
| new1m | 新 | 1,000,000 | 3,000,000 | 60MB |

評価は `train.evaluate_v2`（bigram 生成 beam 32/want 48、int8 ONNX、CPU）。
wiki / suru は delta=1.0, beta=0.5、sns は beta=2.0（従来の比較条件に合わせた）。

```bash
python -m train.build_lex_v3 --corpus-lexicon data/lexicon_v3corpus.tsv \
    --out logs/lex_new_500k.tsv --alt-out logs/alt_new_500k.tsv \
    --max-readings 500000 --max-rows 2000000 --max-size-mb 45
python -m train.experiment_lexicon_scale --lexicon logs/lex_new_500k.tsv --gzip-test
python -m train.evaluate_v2 --data data/eval/wiki2000.jsonl ... --lexicon logs/lex_new_500k.tsv
```

## 3. サイズとロード

| 辞書 | 読み | 行 | 頻度付き行 | TSV | 端末辞書 gzip | Python load | 端末ロード推定※ |
|---|---|---|---|---|---|---|---|
| frozen | 500,000 | 920,108 | 217,606 | 30.9MB | 6.65MB | 3.96s | **1.80s** |
| old500k2m | 500,000 | 1,014,096 | 217,606 | 34.9MB | 7.18MB | 3.89s | 1.80s |
| old1m | 1,000,000 | 1,587,759 | 217,606 | 61.6MB | 12.53MB | 6.44s | **3.60s** |
| new500k | 500,000 | 950,827 | 503,231 | 34.4MB | 7.75MB | 3.96s | 1.80s |
| new1m | 1,000,000 | 1,608,085 | 503,231 | 61.6MB | 13.26MB | 6.57s | **3.60s** |

※ README の実測（14.5万読み=518ms / 50万読み=1.8s）から **1読み≒3.6µs** とみなした線形推定。
実機実測ではない。gzip は端末辞書（3列TSV）形式のサイズ見積もり。

- 読み数を2倍にすると、端末ロードは約2倍・辞書サイズは約2倍。**行数（候補の深さ）はサイズにしか効かない**。

## 4. 精度

| 辞書 | wiki2000 | sns2000 | suru_amb(300) | oracle wiki / sns / suru | 天井 wiki / sns / suru |
|---|---|---|---|---|---|
| frozen | 57.10% | 67.65% | 46.67% | 71.4 / 72.4 / 51.3 | 90.9 / 95.8 / 86.7 |
| old500k2m | 57.10% | — | 46.67% | 71.4 / — / 51.3 | 90.9 / — / 87.0 |
| old1m | **58.35%** | — | 47.67% | 72.7 / — / 53.3 | 91.0 / — / 87.0 |
| new500k | 56.70% | **69.65%** | **50.33%** | 70.4 / 75.0 / 54.7 | 90.5 / 98.1 / 93.3 |
| new1m | **58.40%** | **69.90%** | **51.00%** | 72.2 / 75.2 / 55.7 | 90.9 / 98.2 / 93.3 |

（相対比較のため全辞書で同じ条件。sns は beta=2.0、その他は beta=0.5）

## 5. わかったこと

1. **学習は不要だった。** 精度が動いた要因は「頻度の再集計（数え上げ）」と「読みの追加」だけ。
   モデルの再学習は一切していない。
2. **行数（候補の深さ）を増やしても精度は変わらない。** old500k2m は frozen と完全に同値。
   同一読みの候補を増やしても、コスト18.0の未観測候補は選ばれない。
3. **読み数を増やすと効く。** old1m で wiki +1.25 / suru +1.0。追加分は未観測だが、
   未知語の1文字処理（unk 25.0）より安いスパンとして分割を改善する。
4. **新コーパスの頻度が効く。** new500k はロード時間そのままで sns +2.0 / suru +3.66
   （wiki は −0.40）。new1m は wiki +1.30 / sns +2.25 / suru +4.33。
   suru の天井は 86.7% → 93.3% に上がった（`帰社` などの頻度が入るため）。
5. **1M 読みは効くが高い。** new1m は全指標で最良だが、端末ロード推定 3.6s・gzip 13.3MB。
   50万読みのままでも sns/suru は大きく改善するので、まずは **new500k が費用対効果の最適点**。

## 6. `きしゃする` は直らない

new500k でも 1位は `記者する`（`帰社` はコスト 15.90 で7位、`帰社する` は上位5に入らない）。
頻度の修正だけでは prior 差が残り、BERT も `記者` を好むため。前回の結論どおり、
**連接（bigram）と経路単位の再ランク**が別途必要。

## 7. 推奨

1. **`data/lexicon_v3corpus.tsv` から 50万読みで再生成**（学習不要・ロード増なし）
   → sns +2.0pt / suru +3.7pt。wiki は微減なので、wiki 重視なら 1M 読みか bigram 側の改善とセットで。
2. 1M 読みを使いたいなら、**辞書のバイナリ化 / mmap / 部分ロード**で 3.6s を短縮するのが先。
   行数を増やす投資は効果がないのでやらない。
3. 次は bigram の語彙拡張＋バックオフ（これもカウントのみ）。`帰社` の連接が引けるようになる。

## 8. 再現

```bash
# 辞書サイズ・ロード時間
python -m train.experiment_lexicon_scale --lexicon logs/lex_new_500k.tsv --gzip-test

# 精度
python -m train.evaluate_v2 --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
    --lexicon logs/lex_new_500k.tsv --bigram data/bigram_v3.tsv \
    --bigram-words data/bigram_words_v3.tsv \
    --onnx android/app/src/main/assets/model/scorer.onnx \
    --gen bigram --beam 32 --want 48 --deltas 1.0 --betas 0.5
```
