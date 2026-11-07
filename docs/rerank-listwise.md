# 文頭 hard negative を使った経路単位の再ランク学習

## 目的

文頭（左文脈なし）で頻度 prior に負ける誤変換を、**経路（path）単位のランキング学習**で直す。
`train.train` の文節単位 cross-entropy ではなく、N-best の中の
「正解経路 > 文頭が prior 側の経路」を直接学習する。

## 手順

### 1. N-best を作る

```bash
python -m train.build_rerank_nbest \
    --limit 200000 --head-only \
    --beam 16 --want 16 \
    --out logs/rerank_nbest_head.jsonl
```

- 文頭が prior-wrong の文だけを対象（`--head-only`）
- 正解経路が N-best に入っている文だけ残す
- 文頭が prior 側の経路も N-best に含まれることを要求（hard negative）
- `--general-prob 0.3` で prior-wrong でない文も混ぜられる

### 2. 学習

```bash
python -m train.train_listwise \
    --data logs/rerank_nbest_head.jsonl \
    --init models/scorer_v3.pt \
    --out models/scorer_rerank.pt \
    --epochs 1 --batch-sentences 8 --lr 1e-5 --loss listwise
```

- `--loss listwise`: 経路スコアの softmax に対する cross-entropy
- `--loss pairwise`: 正解経路と各負例の margin（`--margin`）
- 1 文 = 1 forward。文内の全スパン（N-best の和集合）をまとめて採点し、
  経路スコア = 各スパンの logP の和（端末の再ランクと同じ式）
- モデルは表記 ID を持たない（候補は文ごとに char 埋め込みから構築）ため、
  辞書を差し替えても学習できる

### 3. 評価

```bash
python -m train.evaluate_v2 --data data/eval/head_prior_wrong.jsonl \
    --domain head --limit 0 --checkpoint models/scorer_rerank.pt \
    --deltas 1.0 --betas 0.5
```

通常セット（wiki2000 / sns2000 / suru_amb）でも必ず確認する。
改善したら `train.export_onnx_online` で ONNX を書き出し、端末で再評価する。

## スモークテスト

CPU で 30 文の N-best 生成 → 20 文で 1 epoch 学習まで動作確認済み（数秒）。
本学習は GPU 推奨。

## 注意点

- gold が N-best に無い文は使わない（探索の天井の外）
- 経路スコアは分割の異なる経路間で logP の和を比較する（現行 rerank と同じ前提）
- 学習データ生成は CPU で約 4,000 文 / 7 秒。20 万文でも数分〜十数分
- 学習後の評価は「文頭語正解率」と通常セットの両方で行う
## 実験結果（2026-11-07, RTX 5070 Ti）

データ:

- 文頭 prior-wrong の N-best: 50,000 文（beam 8 / want 8、gold と prior 側の経路を含む）
- 一般文の N-best: 20,000 文（`--general-prob 1.0`）を混ぜた 70,000 文
- 初期化: `models/scorer_v3.pt` / 2 epochs / lr 1e-5 / listwise

評価（frozen 辞書 + bigram_v3、ONNX int8 量子化後）:

| セット | 指標 | 現行 v3 | 再ランク 70k |
|---|---|---|---|
| head_prior_wrong (337) | 文正解率 | 0.3709 | **0.4036** |
| head_prior_wrong (337) | **文頭語正解率** | 0.5401 | **0.5935** |
| head_prior_wrong (337) | CER | 0.1054 | **0.0995** |
| wiki2000 | 文正解率 | 0.5710 | **0.5825** |
| sns2000 | 文正解率 | 0.6760 | 0.6700 |
| suru_amb (300) | 文正解率 | 0.4667 | **0.4733** |

実機に近い構成（new500k 辞書 + 20万語 bigram + device プリセット = unigram 生成 β=1.2）:

| セット | 指標 | 現行 v3 | 再ランク 70k |
|---|---|---|---|
| head_prior_wrong | 文正解率 | 0.3234 | **0.3650** |
| head_prior_wrong | **文頭語正解率** | 0.5341 | **0.6083** |

例:

- `せんとうのもじのしっぱいりつたかくない` → `先頭の文字の失敗率高くない`（現行は 戦闘の文字…）
- `せんとうをはしる` → `先頭を走る`（現行は 戦闘を走る）
- `れつのせんとう` → まだ `列の戦闘`（左右の距離2 skip で直るが全体精度とのトレードオフがあり保留）

所感:

- 文頭 prior-wrong の改善は大きい（+5〜7pt）。一般セットは wiki/suru が微増、sns が −0.6pt。
- さらに伸ばすなら: データ量増（10〜50万文）、sns を混ぜた hard negative、pairwise との併用、
  β の再調整。デバイスは unigram 生成なので、bigram 生成への変更も別途検討。
