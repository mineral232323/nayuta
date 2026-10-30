"""Wikipedia からかな漢字変換の学習データと辞書を作る。

入力: data/raw/*.parquet (wikimedia/wikipedia 20231101.ja)
出力:
  data/lexicon.tsv   読み(ひらがな) \t 表記 \t 頻度
  data/train.jsonl   {"ctx": 文脈, "read": 読み, "words": [[読み, 表記], ...]}
  data/valid.jsonl

使い方:
  python -m train.prep_data --limit-articles 60000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import is_japanese_char, kata2hira  # noqa: E402

MAX_READ_LEN = 48
CTX_LEN = 64
SPLIT_CHARS = "。！？!?"
CLOSE_CHARS = "」』）)】〉》\"'"

SKIP_PATTERNS = re.compile(r"(==|\{\{|\}\}|<ref|\|\||https?://|\|)")


def split_sentences(par: str) -> list[str]:
    out, buf, i, n = [], [], 0, len(par)
    while i < n:
        ch = par[i]
        buf.append(ch)
        i += 1
        if ch in SPLIT_CHARS:
            while i < n and par[i] in CLOSE_CHARS:
                buf.append(par[i])
                i += 1
            out.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def clean(s: str) -> str:
    s = s.replace("\u3000", " ").strip()
    return re.sub(r"\s+", "", s)


def usable_sentence(s: str) -> bool:
    if not (8 <= len(s) <= 70):
        return False
    if SKIP_PATTERNS.search(s):
        return False
    ja = sum(1 for c in s if is_japanese_char(c))
    if ja / len(s) < 0.7:
        return False
    if not any(0x3041 <= ord(c) <= 0x309F for c in s) and not any(
        0x4E00 <= ord(c) <= 0x9FFF for c in s
    ):
        return False
    return True


def make_tokenizer():
    from sudachipy import dictionary

    for kwargs in ({}, {"dict_type": "core"}, {"dict": "core"}):
        try:
            d = dictionary.Dictionary(**kwargs)
            return d.tokenizer() if hasattr(d, "tokenizer") else d.create()
        except Exception:
            continue
    raise RuntimeError("SudachiDict を初期化できませんでした")


KANA_OK = set(range(0x3041, 0x3097)) | set(range(0x30A1, 0x30FB)) | {0x30FC}


def is_kana_reading(s: str) -> bool:
    return bool(s) and all(ord(c) in KANA_OK for c in s)


def has_kanji(s: str) -> bool:
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--limit-articles", type=int, default=60000)
    ap.add_argument("--max-sentences", type=int, default=3_000_000)
    ap.add_argument("--valid-ratio", type=float, default=0.01)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = make_tokenizer()

    files = sorted(
        os.path.join(args.raw_dir, f)
        for f in os.listdir(args.raw_dir)
        if f.endswith(".parquet")
    )
    if not files:
        raise SystemExit("parquet がありません")

    word_counts: dict[str, Counter] = defaultdict(Counter)
    records: list[dict] = []
    n_sent = 0
    n_art = 0

    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=200, columns=["text"]):
            for text in batch["text"].to_pylist():
                n_art += 1
                if n_art > args.limit_articles:
                    break
                for para in text.split("\n"):
                    if SKIP_PATTERNS.search(para):
                        continue
                    ctx_buf = ""
                    for sent in split_sentences(para):
                        sent = clean(sent)
                        # 直前文脈は「(1) 段落内の全テキストの末尾」を使う。
                        # 文が 1 つ捨てられただけで文脈を失うと、学習例の 86% が
                        # 文脈なしになってしまうため、捨てた文もバッファには積む。
                        prev_ctx = ctx_buf[-CTX_LEN:]
                        ctx_buf = (ctx_buf + sent)[-CTX_LEN:]
                        if not usable_sentence(sent):
                            continue
                        ms = list(tok.tokenize(sent))
                        words: list[list[str]] = []
                        ok = True
                        for m in ms:
                            pos = m.part_of_speech()
                            if pos[0] == "記号" or (len(pos) > 1 and pos[1] == "数詞"):
                                continue
                            surf = m.surface()
                            if not any(is_japanese_char(c) for c in surf):
                                continue
                            rd = kata2hira(m.reading_form() or surf)
                            if not surf or not is_kana_reading(rd):
                                ok = False
                                break
                            words.append([rd, surf])
                        if (
                            not ok
                            or not words
                            or not any(has_kanji(w[1]) for w in words)
                            or not any(w[0] != w[1] for w in words)
                        ):
                            continue
                        read = "".join(w[0] for w in words)
                        if not (2 <= len(read) <= MAX_READ_LEN):
                            continue
                        for rd, surf in words:
                            word_counts[rd][surf] += 1
                        records.append(
                            {"ctx": prev_ctx, "read": read, "words": words}
                        )
                        n_sent += 1
                if n_art > args.limit_articles:
                    break
            if n_art > args.limit_articles:
                break
        print(f"  {os.path.basename(path)}: articles={n_art} sentences={n_sent}",
              flush=True)
        if n_sent >= args.max_sentences:
            break

    print(f"articles={n_art} sentences={n_sent} readings={len(word_counts)}")

    # 辞書（min_count 以上）
    lex_path = os.path.join(args.out_dir, "lexicon.tsv")
    n_entries = 0
    with open(lex_path, "w", encoding="utf-8") as f:
        for rd, cnt in word_counts.items():
            for surf, c in cnt.items():
                if c < args.min_count:
                    continue
                f.write(f"{rd}\t{surf}\t{c}\n")
                n_entries += 1
    print(f"lexicon: {n_entries} entries -> {lex_path}")

    random.seed(13)
    random.shuffle(records)
    n_valid = max(1, int(len(records) * args.valid_ratio))
    valid, train = records[:n_valid], records[n_valid:]
    for name, rows in (("train", train), ("valid", valid)):
        p = os.path.join(args.out_dir, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {p}")


if __name__ == "__main__":
    main()
