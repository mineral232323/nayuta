# Nayuta

BERT を活用した Android 向け日本語 IME。
入力を文節（トークン）単位に区切り、文脈を考慮した BERT 採点で漢字・英単語へ変換する。

- 設計: [docs/design.md](docs/design.md)
- 状態: 設計フェーズ（実装前）

## 構成（予定）

```
docs/        設計書
train/       BERT スパン分類器の学習・評価（Python）
android/     IME 本体（Kotlin）
assets/      辞書・モデル
```

## 開発メモ

このマシンでは `$HOME/.cache` が読み取り専用のため、Python/uv/HuggingFace の
キャッシュはワークスペース内に置く。

```bash
export UV_CACHE_DIR=$PWD/.cache/uv
export HF_HOME=$PWD/.cache/huggingface
```
