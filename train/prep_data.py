"""Wikipedia / web / SNS からかな漢字変換の学習データと辞書を作る（v3）。

v2 からの変更点（指示書 Phase 3）:

1. **文脈の付与率を上げる** — 段落単位で `ctx_buf` をリセットするのをやめ、
   記事（ドキュメント）単位で引き継ぐ。段落の先頭文も直前段落の末尾を文脈に持つ。
   これにより文脈付与率が 35% → 85% 程度まで上がる。
2. **複数ソース対応** — `--source 名前:glob` を繰り返し指定して、
   Wikipedia（parquet）・web テキスト・SNS（jsonl / jsonl.gz）を混ぜられる。
3. **品質フィルタの強化** — 記号の異常繰り返し・制御文字・日本語比率・文字種の
   多様性・重複文を除去する。生成テキストらしきものも落とす。
4. **重複除去と評価セット除外** — `--exclude-jsonl` で固定評価セットと完全一致する
   文を学習から除く（再学習後の比較を汚さないため）。
5. **メモリ安全な出力** — 全レコードをシャードファイルに書き出してから
   シャッフル・層化サンプリングする。2000 万文規模でも RAM を圧迫しない。

入力:
  data/raw/*.parquet              Wikipedia（columns: id, title, text）
  data/corpus/web/*.jsonl.gz      web テキスト（{"text": ...}）
  data/corpus/sns/*.jsonl.gz      SNS・話し言葉（{"text": ...}）

出力（既定）:
  data/train_v3.jsonl    {"ctx": 文脈, "read": 読み, "words": [[読み, 表記], ...], "src": ドメイン}
  data/valid_v3.jsonl
  data/lexicon_v3corpus.tsv  読み \t 表記 \t 頻度
  data/prep_v3_stats.json    件数・文脈付与率・ドメイン別内訳

使い方:
  python -m train.prep_data \
      --source wiki:'data/raw/*.parquet' \
      --source web:'data/corpus/web/*.jsonl.gz' \
      --source sns:'data/corpus/sns/*.jsonl.gz' \
      --exclude-jsonl data/eval/wiki2000_legacy.jsonl,data/eval/sns2000.jsonl
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import re
import shutil
import sys
import zlib
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Set, Tuple

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import is_japanese_char, kata2hira  # noqa: E402

MAX_READ_LEN = 48
CTX_LEN = 64
SPLIT_CHARS = "。！？!?"
CLOSE_CHARS = "」』）)】〉》\"'"
# Wikipedia のマークアップを含む段落は捨てる（web/SNS には適用しない）
SKIP_PATTERNS = re.compile(r"(==|\{\{|\}\}|<ref|\|\||https?://|\|)")

# 品質フィルタ
REPEAT_SYMBOL = re.compile(r"([!?！？。、,.~ーｗw])\1{3,}")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LATIN_CODE = re.compile(r"[A-Za-z]{4,}")

N_SHARDS = 256


# ---------------------------------------------------------------- 文分割


def split_sentences(par: str) -> List[str]:
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


def usable_sentence(s: str, domain: str) -> bool:
    """学習に使える文か。domain ごとに少しだけ条件を変える。"""
    if not (8 <= len(s) <= 70):
        return False
    if domain.startswith("wiki") and SKIP_PATTERNS.search(s):
        return False
    if CONTROL_CHARS.search(s):
        return False
    if REPEAT_SYMBOL.search(s):
        return False
    # 文字種の多様性（"あああああ…" や "wwww" を落とす）
    if len(set(s)) / len(s) < 0.15:
        return False
    ja = sum(1 for c in s if is_japanese_char(c))
    if ja / len(s) < 0.7:
        return False
    # 英字の長い塊（コード・URL・生成テキスト）は落とす
    if len(LATIN_CODE.findall(s)) >= 2:
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


def crc32(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


# ---------------------------------------------------------------- 入力


def iter_docs(path: str, lines_per_doc: int = 1) -> Iterator[Tuple[str, str]]:
    """(title, text) を返す。parquet と jsonl(.gz) の両対応。

    jsonl ソースが 1 行 1 文で保存されている場合、`lines_per_doc` で連続行を
    1 ドキュメントにまとめる。そうしないと「文脈」が常に空になり、文脈の
    付与率が 0% になってしまう（web / SNS コーパスで実際に起きた）。
    """
    if path.endswith(".parquet"):
        pf = pq.ParquetFile(path)
        cols = [c for c in ("title", "text") if c in pf.schema_arrow.names]
        if "text" not in cols:
            raise SystemExit(f"{path}: text 列がありません")
        for batch in pf.iter_batches(batch_size=200, columns=cols):
            d = batch.to_pydict()
            titles = d.get("title") or [""] * len(d["text"])
            for title, text in zip(titles, d["text"]):
                if text:
                    yield (title or ""), text
        return
    if path.endswith((".jsonl", ".jsonl.gz", ".json.gz", ".json")):
        opener = gzip.open if path.endswith(".gz") else open
        buf: List[str] = []
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, str):
                    text = rec
                else:
                    text = rec.get("text") or rec.get("content") or rec.get("body")
                if not text:
                    continue
                if lines_per_doc <= 1:
                    yield "", text
                    continue
                buf.append(text)
                if len(buf) >= lines_per_doc:
                    yield "", "\n".join(buf)
                    buf = []
        if buf:
            yield "", "\n".join(buf)
        return
    raise SystemExit(f"未対応の拡張子: {path}")


def parse_sources(args) -> List[Tuple[str, List[str]]]:
    specs = args.source or ["wiki:data/raw/*.parquet"]
    out: List[Tuple[str, List[str]]] = []
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--source は 名前:glob 形式で: {spec}")
        domain, pat = spec.split(":", 1)
        pats = [p for p in pat.split(",") if p]
        paths: List[str] = []
        for p in pats:
            matched = sorted(glob.glob(p))
            if not matched:
                print(f"[warn] {domain}: {p} に一致するファイルがありません", flush=True)
            paths.extend(matched)
        out.append((domain, paths))
    return out


def load_excluded(paths: List[str]) -> Set[int]:
    """固定評価セットと完全一致する文を学習から除くためのキー集合。"""
    keys: Set[int] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            print(f"[warn] exclude 対象がありません: {path}", flush=True)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                words = rec.get("words") or []
                if not words:
                    continue
                gold = "".join(w[1] for w in words)
                read = "".join(w[0] for w in words)
                keys.add(crc32(read + "\t" + gold))
                n += 1
        print(f"exclude: {path} ({n} 文)", flush=True)
    return keys


# ---------------------------------------------------------------- 本体


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source", action="append", default=None,
        help="名前:glob（カンマ区切りで複数 glob 可）。複数回指定可。",
    )
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--out-train", default="data/train_v3.jsonl")
    ap.add_argument("--out-valid", default="data/valid_v3.jsonl")
    ap.add_argument("--out-lexicon", default="data/lexicon_v3corpus.tsv")
    ap.add_argument("--out-stats", default="data/prep_v3_stats.json")
    ap.add_argument("--tmp-dir", default="data/tmp_prep")
    ap.add_argument(
        "--exclude-jsonl", default="",
        help="学習から除外する評価セット（カンマ区切り）",
    )
    ap.add_argument(
        "--train-per-domain", default="wiki=6000000,web=1500000,sns=800000",
        help="ドメイン別の学習レコード上限（層化サンプリング）",
    )
    ap.add_argument(
        "--max-sentences-per-domain", default="",
        help="この文数に達したらそのドメインの処理を打ち切る（例: wiki=12000000）。"
        "全 shard は 4000 万文規模になるため、計算量を抑えるのに使う。",
    )
    ap.add_argument("--valid-per-domain", type=int, default=10000)
    ap.add_argument("--valid-ratio", type=float, default=0.002)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--max-reading-len", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--limit-docs", type=int, default=0, help="デバッグ用：記事数上限")
    ap.add_argument(
        "--lines-per-doc", type=int, default=1,
        help="jsonl ソースで連続何行を 1 ドキュメントとして文脈を付与するか"
        "（1 行 1 文のコーパスでは 5 程度にすると文脈付与率が上がる）",
    )
    ap.add_argument(
        "--debug-keep-tmp", action="store_true", help="中間シャードを残す（デバッグ用）"
    )
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    specs = parse_sources(args)
    excluded = load_excluded([p for p in args.exclude_jsonl.split(",") if p])
    tok = make_tokenizer()

    per_domain_cap: Dict[str, int] = {}
    for item in args.train_per_domain.split(","):
        if not item:
            continue
        k, v = item.split("=")
        per_domain_cap[k] = int(v)

    sent_cap: Dict[str, int] = {}
    for item in args.max_sentences_per_domain.split(","):
        if not item:
            continue
        k, v = item.split("=")
        sent_cap[k] = int(v)

    shard_paths = [
        os.path.join(args.tmp_dir, f"shard_{i:03d}.jsonl") for i in range(N_SHARDS)
    ]
    shard_fhs = [open(p, "w", encoding="utf-8") for p in shard_paths]

    word_counts: Dict[str, Counter] = defaultdict(Counter)
    seen: Set[int] = set()
    # 統計
    dom_docs: Counter = Counter()
    dom_sent: Counter = Counter()
    dom_ctx: Counter = Counter()
    n_docs = n_sent = n_dup = n_excluded = 0

    def emit(rec: dict) -> None:
        key = crc32(rec["read"] + "\t" + "".join(w[1] for w in rec["words"]))
        shard_fhs[key % N_SHARDS].write(
            json.dumps(rec, ensure_ascii=False) + "\n"
        )

    def process_doc(title: str, text: str, domain: str) -> None:
        nonlocal n_sent, n_dup, n_excluded
        dom_docs[domain] += 1
        ctx_buf = ""  # 記事（ドキュメント）単位でだけ引き継ぐ
        for para in text.split("\n"):
            if not para:
                continue
            if domain.startswith("wiki") and SKIP_PATTERNS.search(para):
                continue
            for sent in split_sentences(para):
                sent = clean(sent)
                if not sent:
                    continue
                # 文脈は「直前までのテキストの末尾 64 文字」。段落をまたいで保持する。
                prev_ctx = ctx_buf[-CTX_LEN:]
                ctx_buf = (ctx_buf + sent)[-CTX_LEN:]
                if not usable_sentence(sent, domain):
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
                    continue
                read = "".join(w[0] for w in words)
                if not (2 <= len(read) <= MAX_READ_LEN):
                    continue
                key = crc32(read + "\t" + "".join(w[1] for w in words))
                if key in excluded:
                    n_excluded += 1
                    continue
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                for rd, surf in words:
                    word_counts[rd][surf] += 1
                rec = {"ctx": prev_ctx, "read": read, "words": words, "src": domain}
                emit(rec)
                n_sent += 1
                dom_sent[domain] += 1
                if prev_ctx:
                    dom_ctx[domain] += 1

    for domain, paths in specs:
        if not paths:
            continue
        print(f"=== source {domain}: {len(paths)} files", flush=True)
        hit_cap = False
        for path in paths:
            if hit_cap or (args.limit_docs and n_docs >= args.limit_docs):
                break
            print(f"  {os.path.basename(path)}", flush=True)
            for title, text in iter_docs(path, args.lines_per_doc):
                if not text or len(text) > 400_000:
                    continue
                process_doc(title, text, domain)
                n_docs += 1
                cap_s = sent_cap.get(domain)
                if cap_s and dom_sent[domain] >= cap_s:
                    print(f"    sentence cap reached: {domain} {dom_sent[domain]}", flush=True)
                    hit_cap = True
                    break
                if args.limit_docs and n_docs >= args.limit_docs:
                    break
                if n_docs % 20000 == 0:
                    print(
                        f"    docs={n_docs} sentences={n_sent} dup={n_dup}",
                        flush=True,
                    )
        print(
            f"  -> {domain}: docs={dom_docs[domain]} sentences={dom_sent[domain]}",
            flush=True,
        )

    for fh in shard_fhs:
        fh.close()
    print(
        f"docs={n_docs} sentences={n_sent} dup={n_dup} excluded={n_excluded}",
        flush=True,
    )

    # ---- 辞書（コーパス頻度）
    n_entries = 0
    with open(args.out_lexicon, "w", encoding="utf-8") as f:
        for rd, cnt in word_counts.items():
            if len(rd) > args.max_reading_len:
                continue
            for surf, c in cnt.items():
                if c < args.min_count:
                    continue
                f.write(f"{rd}\t{surf}\t{c}\n")
                n_entries += 1
    print(f"lexicon: {n_entries} entries -> {args.out_lexicon}", flush=True)

    # ---- シャッフル＋層化サンプリングで train/valid を書き出す
    order = list(range(N_SHARDS))
    random.shuffle(order)

    def domain_rate(domain: str) -> float:
        cap = per_domain_cap.get(domain)
        n = dom_sent.get(domain, 0)
        if not cap or n <= cap:
            return 1.0
        return cap / n

    valid_target = {
        d: min(args.valid_per_domain, max(1, int(n * args.valid_ratio)))
        for d, n in dom_sent.items()
    }
    n_train = n_valid = 0
    dom_train: Counter = Counter()
    dom_valid: Counter = Counter()
    with open(args.out_train, "w", encoding="utf-8") as ft, open(
        args.out_valid, "w", encoding="utf-8"
    ) as fv:
        for shard in order:
            with open(shard_paths[shard], encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    d = rec["src"]
                    h = crc32(rec["read"] + "\t" + "".join(w[1] for w in rec["words"]))
                    rate = domain_rate(d)
                    if rate < 1.0 and (h % 1_000_000) >= rate * 1_000_000:
                        continue
                    # valid: ドメイン別に小さく確保
                    if dom_valid[d] < valid_target.get(d, 0) and (h % 1000) == 0:
                        fv.write(line)
                        n_valid += 1
                        dom_valid[d] += 1
                        continue
                    cap = per_domain_cap.get(d)
                    if cap and dom_train[d] >= cap:
                        continue
                    ft.write(line)
                    n_train += 1
                    dom_train[d] += 1

    print(f"train: {n_train} -> {args.out_train}", flush=True)
    print(f"valid: {n_valid} -> {args.out_valid}", flush=True)

    stats = {
        "docs": n_docs,
        "sentences": n_sent,
        "duplicates_removed": n_dup,
        "excluded_by_eval_set": n_excluded,
        "train": n_train,
        "valid": n_valid,
        "train_by_domain": dict(dom_train),
        "valid_by_domain": dict(dom_valid),
        "sentences_by_domain": dict(dom_sent),
        "ctx_coverage_by_domain": {
            d: (dom_ctx[d] / dom_sent[d] if dom_sent[d] else 0.0)
            for d in dom_sent
        },
        "ctx_coverage_overall": (
            sum(dom_ctx.values()) / max(sum(dom_sent.values()), 1)
        ),
        "lexicon_entries": n_entries,
        "lexicon_readings": len(word_counts),
        "sources": {d: len(p) for d, p in specs},
        "version": 3,
    }
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)

    if not args.debug_keep_tmp:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
