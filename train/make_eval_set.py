"""固定評価セットをコーパスから作る（Phase 6: ドメイン別評価の基盤）。

`train/prep_data.py` と同じ品質フィルタ・Sudachi 分割・文脈付与（直前方 64 文字）を
使い、学習データと同じ `{"ctx", "read", "words"}` スキーマの jsonl を書き出す。
学習データとの重複は「読み + 表記」の CRC32 完全一致で除く。

例::

    # SNS / 話し言葉の 2000 文を固定評価セットにする
    PYTHONPATH=. .venv/bin/python -m train.make_eval_set \
        --domain sns --glob 'data/corpus/sns/*.jsonl.gz' \
        --out data/eval/sns2000.jsonl --n 2000 \
        --exclude-jsonl data/valid.jsonl,data/eval/wiki2000.jsonl \
        --report-json logs/eval_set_sns.json

    # 学習データとの重複も確認する（train.jsonl は大きいので任意）
    PYTHONPATH=. .venv/bin/python -m train.make_eval_set \
        --domain web --glob 'data/corpus/web/*.jsonl.gz' \
        --out data/eval/web2000.jsonl --n 2000 --exclude-train data/train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import is_japanese_char, kata2hira  # noqa: E402
from train.prep_data import (  # noqa: E402
    CTX_LEN,
    MAX_READ_LEN,
    SKIP_PATTERNS,
    clean,
    crc32,
    has_kanji,
    is_kana_reading,
    iter_docs,
    make_tokenizer,
    split_sentences,
    usable_sentence,
)


def load_excluded(paths: Sequence[str]) -> Set[int]:
    """評価セット・学習データと完全一致する文のキー集合を作る。"""
    keys: Set[int] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            print(f"[warn] exclude 対象がありません: {path}", flush=True)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                words = rec.get("words") or []
                if not words:
                    continue
                gold = "".join(w[1] for w in words)
                read = "".join(w[0] for w in words)
                keys.add(crc32(read + "\t" + gold))
                n += 1
        print(f"exclude: {path} ({n} 文)", flush=True)
    return keys


def iter_records(
    domain: str,
    paths: Sequence[str],
    tok: Any,
    excluded: Set[int],
    stats: Dict[str, int],
    limit_docs: int = 0,
) -> Iterator[Dict[str, Any]]:
    """prep_data と同じ手順で評価候補レコードを列挙する。"""
    for path in paths:
        print(f"  read {path}", flush=True)
        for _title, text in iter_docs(path):
            if not text or len(text) > 400_000:
                continue
            stats["docs"] += 1
            ctx_buf = ""
            for para in text.split("\n"):
                if not para:
                    continue
                if domain.startswith("wiki") and SKIP_PATTERNS.search(para):
                    continue
                for sent in split_sentences(para):
                    sent = clean(sent)
                    if not sent:
                        continue
                    prev_ctx = ctx_buf[-CTX_LEN:]
                    ctx_buf = (ctx_buf + sent)[-CTX_LEN:]
                    stats["sentences"] += 1
                    if not usable_sentence(sent, domain):
                        stats["filtered_quality"] += 1
                        continue
                    ms = list(tok.tokenize(sent))
                    words: List[List[str]] = []
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
                        stats["filtered_tokenize"] += 1
                        continue
                    read = "".join(w[0] for w in words)
                    if not (2 <= len(read) <= MAX_READ_LEN):
                        stats["filtered_length"] += 1
                        continue
                    gold = "".join(w[1] for w in words)
                    key = crc32(read + "\t" + gold)
                    stats["candidates"] += 1
                    if key in excluded:
                        stats["dropped_overlap"] += 1
                        continue
                    excluded.add(key)  # 評価セット内の重複も除く
                    yield {"ctx": prev_ctx, "read": read, "words": words}
            if limit_docs and stats["docs"] >= limit_docs:
                return


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="コーパスから固定評価セット（ctx/read/words の jsonl）を作る"
    )
    ap.add_argument("--domain", required=True, help="sns / web / wiki など")
    ap.add_argument("--glob", required=True,
                    help="入力ファイルの glob（カンマ区切りで複数可）")
    ap.add_argument("--out", required=True, help="出力 jsonl")
    ap.add_argument("--n", type=int, default=2000, help="目標文数")
    ap.add_argument("--pool-factor", type=float, default=3.0,
                    help="目標文数の何倍を候補プールとして集めるか")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--exclude-jsonl", default="",
                    help="除外する既存データ（カンマ区切り。例: data/valid.jsonl）")
    ap.add_argument("--exclude-train", default="",
                    help="学習データ本体との重複も除く（大きいので任意）")
    ap.add_argument("--limit-docs", type=int, default=0, help="デバッグ用の記事数上限")
    ap.add_argument("--report-json", default="", help="作成形質の記録先")
    args = ap.parse_args()

    paths: List[str] = []
    for pat in args.glob.split(","):
        pat = pat.strip()
        if not pat:
            continue
        matched = sorted(glob.glob(pat))
        if not matched:
            print(f"[warn] {pat} に一致するファイルがありません", flush=True)
        paths.extend(matched)
    if not paths:
        raise SystemExit(f"入力が見つかりません: {args.glob}")

    excluded = load_excluded([p for p in args.exclude_jsonl.split(",") if p])
    if args.exclude_train:
        excluded |= load_excluded([args.exclude_train])

    tok = make_tokenizer()
    stats: Dict[str, int] = {
        "docs": 0,
        "sentences": 0,
        "filtered_quality": 0,
        "filtered_tokenize": 0,
        "filtered_length": 0,
        "candidates": 0,
        "dropped_overlap": 0,
    }
    target = int(args.n * args.pool_factor)
    pool: List[Dict[str, Any]] = []
    for rec in iter_records(args.domain, paths, tok, excluded, stats, args.limit_docs):
        pool.append(rec)
        if len(pool) >= target:
            break
        if len(pool) % 1000 == 0:
            print(f"    pool={len(pool)}", flush=True)
    if not pool:
        raise SystemExit("評価候補が 1 件も取れませんでした（フィルタが厳しすぎる可能性）")

    rng = random.Random(args.seed)
    n = min(args.n, len(pool))
    if n < args.n:
        print(
            f"[warn] 候補が足りません: pool={len(pool)} < n={args.n}。"
            f"{n} 文で作成します",
            flush=True,
        )
    sample = pool if n == len(pool) else rng.sample(pool, n)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lens = sorted(len(r["read"]) for r in sample)
    n_ctx = sum(1 for r in sample if r["ctx"])
    n_words = sum(len(r["words"]) for r in sample)
    n_single_hira = sum(
        1
        for r in sample
        for rd, surf in r["words"]
        if rd == surf and all(0x3041 <= ord(c) <= 0x309F for c in surf)
    )
    report = {
        "domain": args.domain,
        "out": args.out,
        "n": len(sample),
        "target_n": args.n,
        "pool": len(pool),
        "seed": args.seed,
        "globs": args.glob,
        "inputs": paths,
        "exclude_jsonl": args.exclude_jsonl,
        "exclude_train": args.exclude_train,
        "stats": stats,
        "sha256": sha256_file(args.out),
        "read_len": {
            "min": lens[0],
            "p50": lens[len(lens) // 2],
            "max": lens[-1],
            "mean": round(sum(lens) / len(lens), 2),
        },
        "ctx_coverage": round(n_ctx / len(sample), 4),
        "words": n_words,
        "words_per_sentence": round(n_words / len(sample), 2),
        "single_hiragana_words": n_single_hira,
        "single_hiragana_rate": round(n_single_hira / max(n_words, 1), 4),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.report_json:
        d = os.path.dirname(os.path.abspath(args.report_json))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.report_json}", flush=True)


if __name__ == "__main__":
    main()
