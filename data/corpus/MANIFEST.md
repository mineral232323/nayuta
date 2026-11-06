# コーパス一覧（MANIFEST）

最終更新: 2026-11-06 16:55 JST

本ディレクトリの日本語テキストは、かな漢字変換モデルの学習データ作成に使う。
各ソースの出典・ライセンス・文数は以下のとおり。NOTICE へはこの表を転記する。

## ソース一覧

| ソース | 出典 URL | ライセンス | 取得日 | パス | サイズ | 記事/行 | 文数 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| Wikipedia 日本語版 20231101.ja | https://huggingface.co/datasets/wikimedia/wikipedia | CC BY-SA 3.0 / GFDL（Wikipedia 日本語版） | 2026-11-06 16:38 JST | `data/raw/train-0000X-of-00015.parquet`（15 シャード） | 3.9 GB | 1389467 | - |
| mC4 (ja) web テキスト | https://huggingface.co/datasets/allenai/c4 | ODC-BY 1.0（C4 / mC4） | 2026-11-06 16:50 JST | `data/corpus/web/*.jsonl.gz`（5 ファイル） | 88.4 MB | 61810 | 2500008 |
| 国会会議録（話し言葉） | https://huggingface.co/datasets/Soma8622/kokkai_speech | MIT（リパッケージ）／原データは国会会議録検索システム（公共データ利用規約 1.0 ≒ CC BY 4.0） | 2026-11-06 16:51 JST | `data/corpus/sns/*.jsonl.gz`（2 ファイル） | 36.0 MB | 368000 | 1002489 |

### Wikipedia のファイル内訳

| ファイル | サイズ (bytes) | 記事数 |
| --- | ---: | ---: |
| `data/raw/train-00000-of-00015.parquet` | 611504422 | 92632 |
| `data/raw/train-00001-of-00015.parquet` | 383148798 | 92632 |
| `data/raw/train-00002-of-00015.parquet` | 304603238 | 92631 |
| `data/raw/train-00003-of-00015.parquet` | 252027908 | 92631 |
| `data/raw/train-00004-of-00015.parquet` | 240133299 | 92631 |
| `data/raw/train-00005-of-00015.parquet` | 240414553 | 92631 |
| `data/raw/train-00006-of-00015.parquet` | 233075429 | 92631 |
| `data/raw/train-00007-of-00015.parquet` | 217654475 | 92631 |
| `data/raw/train-00008-of-00015.parquet` | 224362017 | 92631 |
| `data/raw/train-00009-of-00015.parquet` | 234568965 | 92631 |
| `data/raw/train-00010-of-00015.parquet` | 217186356 | 92631 |
| `data/raw/train-00011-of-00015.parquet` | 213095924 | 92631 |
| `data/raw/train-00012-of-00015.parquet` | 201165492 | 92631 |
| `data/raw/train-00013-of-00015.parquet` | 192710096 | 92631 |
| `data/raw/train-00014-of-00015.parquet` | 176347554 | 92631 |

### web のファイル内訳

| ファイル | サイズ (bytes) | 文数 |
| --- | ---: | ---: |
| `data/corpus/web/web-00000.jsonl.gz` | 17473549 | 500000 |
| `data/corpus/web/web-00001.jsonl.gz` | 17742870 | 500000 |
| `data/corpus/web/web-00002.jsonl.gz` | 17692499 | 500000 |
| `data/corpus/web/web-00003.jsonl.gz` | 17703695 | 500000 |
| `data/corpus/web/web-00004.jsonl.gz` | 17756895 | 500008 |

### sns のファイル内訳

| ファイル | サイズ (bytes) | 文数 |
| --- | ---: | ---: |
| `data/corpus/sns/sns-00000.jsonl.gz` | 18002953 | 500000 |
| `data/corpus/sns/sns-00001.jsonl.gz` | 17992181 | 502489 |

## 品質フィルタ（web / 話し言葉 共通）

`train/fetch_corpus.py` の `ok_sentence()`。1 行 1 文の `{"text": ...}` 形式で保存。

| 条件 | 値 |
| --- | --- |
| 文字数 | 8〜70 文字（NFKC 正規化・空白除去後） |
| 日本語文字比率 | 0.7 以上（ひらがな・カタカナ・漢字・々〆） |
| 記号比率の上限 | 0.25 |
| 除去: URL / HTML タグ / 実体参照 | `URL_RE` / `HTML_RE` |
| 除去: 制御文字・ゼロ幅文字・BOM | `CTRL_RE` |
| 除去: 記号・絵文字の 3 連以上 / `wwww` | `SYMBOL_RUN_RE`、同一文字 3 連続 |
| 除去: 定型文・SEO/ナビ文言（続きを読む・ログイン・カート・ChatGPT 等） | `BOILERPLATE_RE` |
| 除去: 出会い系・アダルト広告の文言 | `AD_SPAM_RE` |
| 除去: 括弧・パイプ・`»` などのナビ表記 | `MARKUP_RE` |
| 除去: 同一 8-gram の繰り返し（生成文らしさ） | `has_repeated_ngram()` |
| 除去: 文書内で同一文が 30% 超を占める定型ページ | `fetch_web()` |
| 完全重複除去 | NFKC 正規化後の blake2b 64bit 一致を全コーパス共通で排除 |

### 話し言葉（国会会議録）だけの追加フィルタ

| 条件 | 内容 |
| --- | --- |
| 発言者ラベルの除去 | 行頭 `○役職（氏名）　本文` のラベルを落として本文だけ残す（`strip_speaker_prefix()`） |
| 進行メモ・日付見出しの除去 | `KOKKAI_META_RE`（平成/昭和/令和・出席者・議事日程・辞任 など） |
| 人名の羅列の除去 | `君` を含む文、ひらがな比率 10% 未満の文（`HIRA_MIN`） |

## 採用しなかったソースと理由

| ソース | ライセンス | 不採用の理由 |
| --- | --- | --- |
| CC-100 (ja) | 不明瞭 | `datasets` 5.x で script-based dataset の読み込みが不可。web は mC4 で代替 |
| community-datasets/covid_tweets_japanese | CC BY-ND 4.0 | ND（改変禁止）で学習利用の解釈が不明確、COVID 関連の偏りも大 |
| kubota/defamation-japanese-twitter | CC BY 4.0 | 誹謗中傷の収集データで、IME 学習用テキストとして不適切 |
| IngCrowd/japanese-themed-dialogue | CC BY-NC 4.0 | NC（非商用）で再配布条件が合わない |
| cyberagent/chatbot-arena-ja-calm2-*-experimental | CC BY 4.0 | gated（利用同意必須）で HF_TOKEN 無しでは取得不可 |
| shi3z/Japanese_Wikipedia_Conversation | CC BY-SA 4.0 | LLM 生成の対話で「生成テキスト除去」方針に反する |
| OmniAICreator/Japanese-Roleplay-Dialogues | Apache-2.0 | LLM 生成のロールプレイで文面が不適切なものが混在 |
| globis-university/aozorabunko-chats | CC BY 4.0 | 青空文庫からの生成対話で、やはり生成テキスト |

→ 日本語 SNS コーパスは、再配布可能なライセンスのものが HF 上に見つからなかった。
   話し言葉は国会会議録（議事録 = 実際の発話）で代替した。

## ローカルキャッシュ（学習には使わない）

| パス | 内容 |
| --- | --- |
| `data/cache/sns/all_speeches.parquet` | 国会会議録の元 parquet（3.1GB）。`--what sns` の再実行時に再ダウンロードを省くだけ |

## 再生成方法

```sh
# 全部（冪等。取得済みはスキップ）
python -m train.fetch_corpus --what all
# 個別
python -m train.fetch_corpus --what wiki
python -m train.fetch_corpus --what web --web-max-sentences 2500000
python -m train.fetch_corpus --what sns
# MANIFEST だけ作り直す
python -m train.fetch_corpus --what manifest
```
