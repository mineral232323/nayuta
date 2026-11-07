# bigram の語彙拡張と補間バックオフ（実験）

> 実施日: 2026-11-07 / ブランチ: `experiment/bigram-backoff` / 状態: 実験（アセット未差し替え）

## 1. 目的

現行 bigram は

- 語彙が上位 **10 万語**だけ（`帰社` は 15.6 万位なので入っていない）
- 未観測の連接は **unigram コストへハードフォールバック**（生成は `+δ=1.0`、再ランクは素の unigram）

なので、辞書にあるが頻度の低い語は連接統計を持てない。語彙を 20 万語まで広げ、
**Jelinek-Mercer 補間**で未観測連接も滑らかに扱えるようにして、精度が上がるかを見る。

```
P(cur|prev) = (c(prev,cur) + α·P_uni(cur)) / (c(prev) + α)
```

## 2. 実装

- `train/build_bigram_backoff.py` — 語彙を辞書側（コーパス頻度）から取り、`train_v3` を
  1 パスして unigram / 連接の**生カウント**を出力（`words.tsv` / `pairs.tsv` / `meta.json`）
- `BackoffBigramLM`（`train/evaluate_bigram.py`） — α を受けて補間コストを計算。
  未観測でも `None` を返さずバックオフコストを返す
- `train/experiment_bigram_backoff.py` — baseline と比較評価

語彙 20 万語を選んだ理由: `帰社` のコーパス頻度順位が約 15.6 万位のため。

## 3. 結果

### 3.1 語彙拡張だけ（フォールバックは現行のまま = ハード）

`pairs.tsv` の生カウントを ML コスト `-log(c(prev,cur)/c(prev))` に変換し、
既存 `BigramLM`（観測済みのみ、未観測は unigram へ）で評価。

| bigram | wiki2000 | sns2000 | suru_amb |
|---|---|---|---|
| 現行 10 万語 | 57.10% | 67.65% | 46.67% |
| **20 万語に拡張** | **58.90%** | **68.40%** | **49.33%** |
| oracle（20万語） | 72.35 | 72.90 | 54.33 |

- **語彙拡張だけで wiki +1.80 / sns +0.75 / suru +2.66**。学習は不要（カウントのみ）。
- サイズ: `bigram.bin` 13.43MB → **23.68MB**（+10.25MB）。
  Python でのロード 0.82s → 1.49s、語彙 10万→20万、連接 2.00M→3.45M。

### 3.2 補間バックオフ（JM, α=100）

| モデル | wiki2000 | sns2000 | suru_amb |
|---|---|---|---|
| baseline（現行） | 57.10% | 67.65% | 46.67% |
| bo100k α=100 | 54.35% | 63.40% | 47.33% |
| bo200k α=100 | 54.50% | 63.65% | 46.67% |
| bo200k α=300 | 55.45% | — | 47.33% |
| bo200k α=1000 | 55.85% | — | 47.00% |
| bo200k ハイブリッド（生成は現行・再ランクのみ α=100） | 56.50% | 68.05% | 49.33% |

- 完全に置き換えると**大幅に悪化**（wiki −2.6〜−3.6pt、sns −4.0〜−4.3pt）。α を上げても回復しない。
  oracle も下がる（wiki 71.4 → 68.2 / sns 72.35 → 68.05）ので、**生成の到達率**が落ちている。
- 生成は現行（観測済みのみ＋δ）のまま、再ランクだけバックオフにすると
  suru +2.66 / sns +0.40 / wiki −0.60（oracle は 72.30 / 72.90 / 54.33 に上昇）。
  悪化はしないが、語彙拡張＋ハードフォールバック（wiki 58.90）には届かない。

### 3.3 新コーパス辞書との組み合わせ（語彙 20 万 + ハードフォールバック）

| 辞書 | bigram | wiki2000 | sns2000 | suru_amb |
|---|---|---|---|---|
| 現行 | 10万 | 57.10% | 67.65% | 46.67% |
| 現行 | 20万 | **58.90%** | 68.40% | 49.33% |
| new500k（新コーパス） | 10万 | 56.70% | 69.65% | 50.33% |
| new500k | 20万 | 58.40% | **70.80%** | **53.00%** |

- データセット別の最良: wiki は「現行辞書 + 20万語」、sns/suru は「new500k + 20万語」。
- 現行比では最大 **wiki +1.3 / sns +3.15 / suru +6.33**。

## 4. なぜバックオフが負けたか

- JM の未観測コストは `log((c(prev)+α)/α) − log P_uni(cur)`。
  頻出語（`の` など）の未観測連接は **10 nat 超**になり、現行の「unigram + δ(1.0)」より
  はるかに悲観的。
- 85M トークンのコーパスでは、正解経路にも未観測連接が多い。ビーム探索が
  「観測済み連接の多い経路」に寄り、**N-best から正解が落ちる**（oracle 低下）。
- α を上げると unigram に近づくが、今度は観測済み連接の識別力が消える。
  JM + 固定 α は頻出文脈に弱い（Witten-Bell / Kneser-Ney / 上限付きバックオフが必要）。

## 5. きしゃする は直らない

bo200k のバックオフでは `帰社→する` が 3.8 nat（α=100）と安くなるが、
初語の unigram 差（new500k で `帰社` 15.9 vs `記者` 9.8）を覆せず、
`帰社する` は 5 位（frozen 辞書では 5 位圏外）。1 位は `記者する` のまま。
語彙拡張＋ハードフォールバックでも同じ。

→ 連接統計の平滑化だけでは「記者はサ変でない」という語彙的な知識は入らない。
POS（サ変可能）や経路単位の再ランクが引き続き必要。

## 6. 結論・推奨

1. **bigram の語彙拡張（10万→20万）は有効**。学習不要、カウントだけで +0.75〜+2.66pt。
   まずこれを採用するのが良い（サイズ +10MB は要検討）。
2. **JM バックオフを生成コストに使うのは悪手**。現行の「unigram + δ」フォールバックの
   方が到達率を保てる。再ランク専用にすると suru は改善するが wiki は微減で、
   語彙拡張＋ハードフォールバックに総合では劣る。バックオフを活かすなら
   - unigram より悲観的にならない上限（cap）を付ける
   - Witten-Bell / Kneser-Ney を試す
3. 辞書と bigram の組み合わせは、wiki と sns/suru で最適点が違う。
   リリースではどちらを優先するかを決める必要がある。

## 7. 実機確認（SC-51E / Android 17）

`TestInputActivity` に追加した intent extra `bench` で、実機のロード時間と変換時間を計測した
（`adb shell am start -n com.nayuta.ime/.TestInputActivity --es bench "..." --ei bench_runs 5`）。
比較は「旧アセット（frozen 辞書 + 10万語 bigram）」と「新アセット（new500k 辞書 + 20万語 bigram）」。

| 項目 | 旧アセット | 新アセット |
|---|---|---|
| 辞書ロード（50万読み） | 1854ms | 1776ms |
| BERT ロード | 216ms | 204ms |
| bigram ロード | 184ms | **282ms** |
| 初回変換（コールド） | 2323ms | 2398ms |
| 短文（きしゃする） | 6.4〜7.8ms | 3.1〜3.6ms |
| 中文（13〜20字） | 8.7〜14ms | 9〜19ms |
| 長文（26字） | 18〜27ms | 21〜33ms |
| PSS | 253MB | **303MB（+50MB）** |
| APK | 76.9MB | **86.6MB（+9.6MB）** |

- ロード時間の支配要因は辞書（約1.8s）。bigram 拡張の追加は約 **+100ms**。
- ウォーム時の変換は長文で +3〜6ms、中文・短文はほぼ同等。
- 代償は **RAM +50MB / APK +9.6MB**。IML としては無視できない。
- `きしゃする` は新旧どちらも `記者する`。
- サンプル1文（`あしたのてんきははれです`）では旧 `晴れ` → 新 `母れ` と悪化した。
  1件では判断できないので、評価セットの数字で判断する。

## 8. 再現

```bash
# bigram を作る（語彙 20 万）
python -m train.build_bigram_backoff --vocab-size 200000 --out-dir logs/bigram_bo200k

# ハードフォールバック用の ML コストを作る
#   pairs.tsv の (prev,cur,count) から -log(count/c(prev)) を pairs_cost.tsv に書く

# 語彙拡張の評価
python -m train.evaluate_v2 --data data/eval/wiki2000.jsonl --domain wiki --limit 2000 \
    --lexicon data/lexicon_v3_frozen.tsv \
    --bigram logs/bigram_bo200k/pairs_cost.tsv --bigram-words logs/bigram_bo200k/words.tsv \
    --onnx android/app/src/main/assets/model/scorer.onnx \
    --gen bigram --beam 32 --want 48 --deltas 1.0 --betas 0.5

# バックオフの比較
python -m train.experiment_bigram_backoff --dirs logs/bigram_bo100k logs/bigram_bo200k \
    --alpha 100 --datasets wiki,sns,suru
```
