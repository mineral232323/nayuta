"""読み→表記辞書 v3 を作る（Phase 1: 辞書増強）。

v2 (`train/build_lex_v2.py`) は「コーパスで観測した表記」に SudachiDict の別読みだけを
足していた。v3 はそれに加えて、コーパスに全く出てこない語彙（SudachiDict full の全語彙、
mecab-ipadic-NEologd の新語・固有名詞、Wikipedia 記事タイトル）を**既定コスト**で足し、
変換候補の到達可能な被覆（reachable ceiling）を広げる。

ただし無制限に入れると端末の辞書ロードが破綻し（実測 14.5 万読みで 518ms。読み数に
ほぼ比例する）、探索も濁る。そこで **3 つの上限（読み数 / 行数 / ファイルサイズ）** を
守りながら、優先度の高い順に採用する。

構成（採用優先度順。同じ (読み, 表記) は先勝ち）:
  (a) corpus   : `--corpus-lexicon` の観測 (読み, 表記, 頻度)。
                 コスト = -log(頻度 / 総トークン数)。**観測済みは全件残す**
                 （上限にどうしても収まらないときだけ頻度の低い順に落とす）。
  (b) variant  : 表記はコーパス観測済みで、その読みだけ未観測のもの（SudachiDict /
                 NEologd / タイトル由来）。コスト = -log(表記の総頻度/総トークン数)
                 + variant-penalty（v2 の式）。表記の頻度という裏付けがあるので、
                 未観測表記より優先して残す。
  (c) neologd  : mecab-ipadic-NEologd の seed/*.csv.xz を直接パース（MeCab ビルド不要）。
                 列は `表層形,左ID,右ID,コスト,品詞,品詞細分類1,2,3,活用型,活用形,原形,読み,発音`
                 の 13 列。表層形にカンマが入り得るので **右から 12 個で split** する。
                 読みはカタカナなのでひらがなに直す。記号/数詞/空白は除外。
                 採用順は POS（名詞を優先）→ MeCab の語コスト（= 頻度の代理）→ 表記長。

                 **ライセンス注意**: 本スクリプトが読むのは NEologd の `seed/*.csv(.xz)`
                 だけであり、MeCab 用辞書 (`mecab-ipadic`) 本体は読み込まないし再配布もしない。
                 seed CSV は NEologd の Apache License 2.0 の対象で、同梱の COPYING が
                 求める帰属表示（はてなキーワード / 日本郵便 / 日本全国駅名一覧 /
                 工藤拓氏 人名データ / Web クロール）は NOTICE に転記している。
  (d) title    : data/raw/*.parquet の title 列（Wikipedia 記事タイトル）。
                 読みは (1) 記事冒頭の「'''表題'''（よみ）」パターン、
                 (2) それが無ければ Sudachi で形態素分割し reading_form() を連結。
                 コストは他より少し高くし、採用順は記事長の降順（有名さの代理）。
  (e) sudachi  : sudachidict-full の全エントリ（約 290 万行）から (読み, 表記)。
                 コストは既定。採用順は POS → 表記長 → 読み長。最後の優先度。

コストの考え方（実測に基づく）:
  - 既定コスト default_cost は「コーパスに 1 回だけ出た語」より**大きく**しないと、
    未観測の固有名詞が観測済みの一般語を押しのけて候補選択が濁る。
    旧コーパス（2,368 万トークン）では -log(1/2.37e7) ≒ 16.9 なので
    default_cost=14.5 では足りず、18.0 で mincost_acc が 0.8914 → 0.8974 に改善した
    （span / surf@8 / sent_cov は 18.0 でも不変）。総トークン T に対して
    目安は `-log(1/T)` 前後。上限は未知語コスト（conv.Lexicon の unk_cost = 25/1 文字）
    なので 20 を超えると未観測語が未知語扱いに近づき候補として機能しなくなる。
  - variant-penalty は v2 と同様 `-log(表記の総頻度/総トークン数) + penalty`。
    2.5/3.5/4.5/6.0/8.0 を比較し、mincost_acc が 4.5=0.8914 → 6.0=0.8918 → 8.0=0.8920
    と単調に（ただし僅かに）改善。既定は 6.0（v2 は 4.5）。

語彙を増やしすぎないための制限:
  - `--min-surface-len` : 未観測表記の最小長（既定 2）。単漢字はコーパスがかなり拾う。
  - 1 文字表記      : 未観測の 1 文字は「漢字を含むもの」だけ許可（`日/手/気` は可、
                      `ぁ/あ/ん` は不可）。コーパス観測済みの 1 文字はそのまま残す
                      （`て/に/を` のような助詞が消えるとラティスが壊れるため）。
  - `--topk`        : 読みあたりの候補数上限。学習・評価・端末はいずれも k=8 なので
                      既定 8（12 まで許容）。
  - `--max-readings` / `--max-rows` / `--max-size-mb` : 出力の上限。
    端末ロード時間は読み数にほぼ比例する（14.5 万読みで 518ms の実測）。
  - NEologd の特定 seed は除外。形容詞の伸ばし表現 (`-exp-`, `-std-`) と非標準表記
    (ill-formed) は「表記ゆれを MeCab に吸収させる」ためのもので、読み→表記辞書に
    入れると「おそれおおーーいっ」のような誤変換候補になる。

出力:
  data/lexicon_v3.tsv       読み \\t 表記 \\t 頻度 \\t コスト*100  （train.load_lexicon 互換）
  data/alt_readings_v3.tsv  表記 \\t 読み1,読み2,...            （train.load_alt_readings 互換）

再現手順（RUNBOOK）:
  # 依存（pyproject.toml に追記済み）
  uv pip install sudachidict-full
  # NEologd の seed CSV だけ欲しい（MeCab のビルドは不要）
  git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd data/raw/neologd
  # コーパス辞書（Lead が Phase 3 で全コーパスから生成する）
  PYTHONPATH=. .venv/bin/python -m train.prep_data
  # 本スクリプト（既定で k=8 / 読み 50 万 / 120 万行 / 30MB を上限に選ぶ）
  PYTHONPATH=. .venv/bin/python -m train.build_lex_v3 \\
      --corpus-lexicon data/lexicon_v3corpus_wiki.tsv
  # 検証（gold span coverage / 候補数分布 / unigram 文正解率）
  PYTHONPATH=. .venv/bin/python -m train.build_lex_v3 --valid data/eval/wiki2000_legacy.jsonl
  # 端末辞書（Lead が最後に回す）
  PYTHONPATH=. .venv/bin/python -m train.export_dict --lexicon data/lexicon_v3.tsv --topk 8
"""

from __future__ import annotations

import argparse
import glob
import json
import lzma
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import kata2hira  # noqa: E402

# ------------------------------------------------------------------ 文字クラス

_KANA_START, _KANA_END = 0x3041, 0x3096
_KATA_START, _KATA_END = 0x30A1, 0x30FB
_KANA = set(range(_KANA_START, _KANA_END + 1)) | set(range(_KATA_START, _KATA_END)) | {0x30FC}
_KANJI = set(range(0x4E00, 0x9FFF)) | {0x3005, 0x3006}  # 漢字 + 々〆
_KATAKANA = set(range(0x30A1, 0x30FB)) | {0x30FC}
# 表記に許す文字: かな・漢字・中黒・ASCII 英数字と一部記号。
# 括弧・チルダ・罫線などは表記ゆれ/装飾なので落とす。
_SURFACE_EXTRA = {0x30FB} | set(range(0x30, 0x3A)) | set(range(0x41, 0x5B)) | set(range(0x61, 0x7B))

SKIP_POS = ("記号", "空白", "補助記号")

# 採用順に使う品詞の優先度（小さいほど優先）。名詞 → 用言 → その他。
POS_RANK = {
    "名詞": 0,
    "動詞": 1,
    "形容詞": 1,
    "形状詞": 1,
    "副詞": 2,
    "連体詞": 2,
    "接頭辞": 2,
    "接尾辞": 2,
    "接続詞": 3,
    "感動詞": 3,
    "助詞": 4,
    "助動詞": 4,
}

# NEologd の seed のうち、変換辞書に入れると害になるもの。
#   -exp- / -std- : 形容詞の伸ばし表現（おそれおおーーいっ 等）を MeCab に吸収させる用
#   ill-formed    : 非標準表記（ヤボッタィ 等）
NEOLOGD_EXCLUDE_PATTERNS = ("-exp-", "-std-", "ill-formed")

# 伸ばし表現の検出。コーパスの読みには「ーー」が 1 件も無い（22.7 万行中 0）ので、
# 連続長音は異常形とみなしてよい。
_RE_ELONG = re.compile(r"ー{2,}")
_RE_INTRO = re.compile(r"'''(.{1,80}?)'''")
_RE_KANA_PAREN = re.compile(r"^[（(]([ぁ-ゖァ-ヺー・、\s]{1,40})[)）]")
_RE_KANA_ONLY = re.compile(r"^[ぁ-ゖァ-ヺー・]+$")
_RE_TITLE_DISAMBIG = re.compile(r"^(.*?)\s*[（(][^)）]{1,40}[)）]\s*$")

# 表記長の上限（これを超える固有名詞は誤変換・探索の濁りの元）
MAX_SURFACE_LEN_DEFAULT = 24

# ソース別の採用優先度（小さいほど優先）。variant はソースより先に置く。
SRC_ORDER = {"corpus": 0, "sudachi": 1, "neologd": 2, "title": 3}
SRC_ORDER_VARIANT = {"sudachi": 0, "neologd": 1, "title": 2}


# ------------------------------------------------------------------ フィルタ


def kana_ok(s: str) -> bool:
    """ひらがな/カタカナ/長音のみで構成されているか。"""
    return bool(s) and all(ord(c) in _KANA for c in s)


def reading_ok(s: str, max_len: int) -> bool:
    """読みとして妥当か（かなのみ・長すぎない・異常な伸ばしが無い）。"""
    return kana_ok(s) and len(s) <= max_len and not _RE_ELONG.search(s)


def has_kanji(s: str) -> bool:
    return any(ord(c) in _KANJI for c in s)


def surface_ok(s: str, min_len: int, max_len: int) -> bool:
    """表記として妥当か。

    - 長さが範囲内
    - 漢字かカタカナを 1 文字以上含む（ひらがなだけの表記は変換結果として無意味）
    - 使える文字だけで構成されている（〜, 括弧, 罫線, 記号を落とす）
    """
    if not s or not (min_len <= len(s) <= max_len):
        return False
    if not (has_kanji(s) or any(ord(c) in _KATAKANA for c in s)):
        return False
    return all(ord(c) in _KANA or ord(c) in _KANJI or ord(c) in _SURFACE_EXTRA for c in s)


def one_char_ok(surf: str) -> bool:
    """未観測の 1 文字表記は漢字を含むものだけ許可する（ぁ/あ/ん を落とす）。"""
    return len(surf) > 1 or has_kanji(surf)


# ------------------------------------------------------------------ データ構造


@dataclass(frozen=True)
class Entry:
    """辞書 1 エントリ。

    count=0 はコーパス未観測。tier は採用順の大分類:
      0 = コーパス観測（必ず残す）
      1 = 別読み（表記は観測済み、読みだけ未観測）
      2 = 未観測（外部辞書にしか無い語彙。上限に達したらここから削る）
    rank は tier=2 の並び順の材料 `(漢字か, ソース順, 表記長, ソース固有の信号)`。
    """

    count: int
    cost: float
    source: str
    tier: int
    src_order: int
    rank: Tuple


# ------------------------------------------------------------------ ソース読み込み


def load_corpus(
    path: str, min_freq: int
) -> Tuple[Dict[Tuple[str, str], int], int, Counter]:
    """(a) コーパス観測分。(読み, 表記) -> 頻度 と 総トークン数 と 表記別頻度。"""
    counts: Dict[Tuple[str, str], int] = {}
    surf_total: Counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rd, surf, c = parts[0], parts[1], int(parts[2])
            if c < min_freq:
                continue
            counts[(rd, surf)] = counts.get((rd, surf), 0) + c
            surf_total[surf] += c
    total = sum(counts.values()) or 1
    return counts, total, surf_total


def make_sudachi_dict(dict_type: str):
    """SudachiDict を開く。full が無ければ core にフォールバックする。"""
    from sudachipy import dictionary

    try:
        return dictionary.Dictionary(dict=dict_type), dict_type
    except Exception as exc:  # pragma: no cover - 環境依存
        if dict_type != "core":
            print(f"[warn] SudachiDict({dict_type}) を開けません ({exc}); core を使います")
            return dictionary.Dictionary(dict="core"), "core"
        raise


def iter_sudachi(
    dict_type: str, max_reading_len: int, min_surface_len: int, max_surface_len: int
) -> Iterator[Tuple[str, str, str]]:
    """(b) SudachiDict の全エントリから (読み, 表記, 品詞1) を列挙する。"""
    d, used = make_sudachi_dict(dict_type)
    print(f"[info] SudachiDict: {used}", flush=True)
    for m in d.entries():
        pos = m.part_of_speech()
        if pos[0] in SKIP_POS:
            continue
        if len(pos) > 1 and pos[1] == "数詞":
            continue
        surf = m.surface()
        if not surface_ok(surf, min_surface_len, max_surface_len):
            continue
        reading = m.reading_form()
        if not reading or reading == "*":
            continue
        rd = kata2hira(reading)
        if not reading_ok(rd, max_reading_len):
            continue
        yield rd, surf, pos[0]


def _neologd_seed_files(seed_dir: str, exclude: Sequence[str]) -> List[str]:
    files: List[str] = []
    for name in sorted(os.listdir(seed_dir)):
        if not (name.endswith(".csv") or name.endswith(".csv.xz")):
            continue
        if any(p in name for p in exclude):
            continue
        files.append(os.path.join(seed_dir, name))
    return files


def _open_seed(path: str):
    if path.endswith(".xz"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_neologd(
    seed_dir: str, max_reading_len: int, min_surface_len: int, max_surface_len: int,
    exclude: Sequence[str],
) -> Iterator[Tuple[str, str, str, int]]:
    """(c) NEologd の seed CSV から (読み, 表記, 品詞1, MeCab 語コスト) を列挙する。

    列: 表層形,左ID,右ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,
        活用型,活用形,原形,読み,発音 の 13 列。
    表層形にカンマが含まれ得るので右から 12 個で split する。
    MeCab の語コストは「小さいほど高頻度」なので採用順の信号に使う。
    """
    files = _neologd_seed_files(seed_dir, exclude)
    if not files:
        print(f"[warn] NEologd の seed CSV がありません: {seed_dir}")
        return
    for path in files:
        n = 0
        with _open_seed(path) as f:
            for line in f:
                parts = line.rstrip("\n").rsplit(",", 12)
                if len(parts) != 13:
                    continue
                surf, pos1, pos2, yomi = parts[0], parts[4], parts[5], parts[11]
                if pos1 in SKIP_POS or pos1 == "数詞" or pos2 == "数詞":
                    continue
                if not surface_ok(surf, min_surface_len, max_surface_len):
                    continue
                if not yomi or yomi == "*":
                    continue
                rd = kata2hira(yomi)
                if not reading_ok(rd, max_reading_len):
                    continue
                try:
                    wcost = int(parts[3])
                except ValueError:
                    wcost = 99999
                n += 1
                yield rd, surf, pos1, wcost
        print(f"[info] neologd {os.path.basename(path)}: kept={n:,}", flush=True)


def title_reading_from_intro(title: str, text: str) -> Optional[str]:
    """記事冒頭の「'''表題'''（よみ）」から読みを取る。"""
    head = text[:600]
    m = _RE_INTRO.search(head)
    if not m:
        return None
    after = head[m.end(): m.end() + 80].lstrip()
    m2 = _RE_KANA_PAREN.match(after)
    if not m2:
        return None
    # 「にほん、にっぽん」のように複数読みが並ぶことがある。先頭を採る。
    for cand in re.split(r"[、,／/]", m2.group(1)):
        cand = cand.strip().replace(" ", "").replace("\u3000", "")
        if cand and _RE_KANA_ONLY.match(cand) and not _RE_ELONG.search(cand):
            return kata2hira(cand)
    return None


def title_reading_by_sudachi(title: str, tok) -> Optional[str]:
    """タイトルを Sudachi で分割し、reading_form() を連結して読みにする。"""
    try:
        ms = list(tok.tokenize(title))
    except Exception:
        return None
    if not ms:
        return None
    out: List[str] = []
    for m in ms:
        pos = m.part_of_speech()
        if pos[0] in SKIP_POS:
            return None
        r = m.reading_form()
        if not r or r == "*":
            return None
        out.append(r)
    rd = kata2hira("".join(out))
    return rd or None


def clean_title(title: str) -> str:
    """曖昧さ回避の括弧を落とし、前後空白を除く。"""
    t = title.replace("\u3000", " ").strip()
    m = _RE_TITLE_DISAMBIG.match(t)
    if m:
        t = m.group(1).strip()
    return t


def iter_titles(
    raw_dir: str, max_titles: int, max_reading_len: int,
    min_surface_len: int, max_surface_len: int,
) -> Iterator[Tuple[str, str, int]]:
    """(d) Wikipedia タイトルから (読み, 表記, 記事長) を列挙する。

    記事長は「採用順を決めるときの有名さの代理」に使う。
    """
    files = sorted(
        p for p in glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True)
        # HF のダウンロード途中キャッシュ等は除外する
        if not any(part.startswith(".") for part in os.path.relpath(p, raw_dir).split(os.sep))
    )
    if not files:
        print(f"[warn] parquet がありません: {raw_dir}")
        return

    import pyarrow.parquet as pq

    d, _ = make_sudachi_dict("full")
    tok = d.tokenizer()

    seen: set = set()
    rows: List[Tuple[int, str, str]] = []
    n_art = 0
    for path in files:
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:  # ダウンロード途中の壊れたファイルは飛ばす
            print(f"[warn] skip {path}: {exc}")
            continue
        try:
            for batch in pf.iter_batches(batch_size=256, columns=["title", "text"]):
                for title, text in zip(batch["title"].to_pylist(), batch["text"].to_pylist()):
                    n_art += 1
                    surf = clean_title(title or "")
                    if not surface_ok(surf, min_surface_len, max_surface_len):
                        continue
                    rd = title_reading_from_intro(surf, text or "") or \
                        title_reading_by_sudachi(surf, tok)
                    if not rd or not reading_ok(rd, max_reading_len):
                        continue
                    key = (rd, surf)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append((len(text or ""), rd, surf))
        except Exception as exc:  # 途中まで読めた分は使う
            print(f"[warn] partial {path}: {exc}")
        print(f"[info] titles {os.path.basename(path)}: artifacts={n_art:,} "
              f"pairs={len(rows):,}", flush=True)

    if max_titles > 0 and len(rows) > max_titles:
        # 記事長の降順で上限まで（長い記事 = 有名な項目という代理指標）
        rows.sort(key=lambda r: (-r[0], r[1], r[2]))
        rows = rows[:max_titles]
    for _, rd, surf in rows:
        yield rd, surf


# ------------------------------------------------------------------ マージと取捨選択


def build_entries(
    corpus: Dict[Tuple[str, str], int], total: int, surf_total: Counter,
    args: argparse.Namespace,
) -> Dict[Tuple[str, str], Entry]:
    """各ソースをマージして (読み, 表記) -> Entry を作る（上限適用前の全件）。"""
    entries: Dict[Tuple[str, str], Entry] = {}

    def add_variant_or_new(rd: str, surf: str, source: str, extra: float,
                           tail: Tuple) -> None:
        key = (rd, surf)
        if key in entries:
            return  # 先勝ち（優先度の高いソースを残す）
        if not one_char_ok(surf):
            return
        if surf in surf_total:
            # 表記は観測済み、この読みだけ未観測 -> variant-penalty 付き
            cost = -math.log(surf_total[surf] / total) + args.variant_penalty + extra
            entries[key] = Entry(0, cost, source, 1, SRC_ORDER_VARIANT.get(source, 9), ())
            return
        if len(surf) < args.min_surface_len:
            return
        cost = args.default_cost + extra
        # rank = (漢字を含むか, ソース順, 表記長, ソース固有の信号)。
        # 実際の大小関係は --rank-policy で決める（select_entries を参照）。
        rank = (0 if has_kanji(surf) else 1, SRC_ORDER[source], len(surf), tail)
        entries[key] = Entry(0, cost, source, 2, SRC_ORDER[source], rank)

    # (a) コーパス観測分: cost = -log(頻度/総トークン数)。観測済みは全件残す。
    for (rd, surf), c in corpus.items():
        entries[(rd, surf)] = Entry(c, -math.log(c / total), "corpus", 0, 0, ())

    n_corpus = len(entries)
    print(f"[info] corpus: pairs={n_corpus:,}", flush=True)

    if not args.no_neologd:
        n0 = len(entries)
        for rd, surf, pos1, wcost in iter_neologd(
            args.neologd_dir, args.max_reading_len, 1,
            args.max_surface_len, args.neologd_exclude,
        ):
            tail = (POS_RANK.get(pos1, 5), wcost)
            add_variant_or_new(rd, surf, "neologd", args.neologd_offset, tail)
        print(f"[info] neologd: +{len(entries) - n0:,}", flush=True)

    if not args.no_sudachi:
        n0 = len(entries)
        for rd, surf, pos1 in iter_sudachi(
            args.sudachi_dict, args.max_reading_len, 1, args.max_surface_len,
        ):
            tail = (POS_RANK.get(pos1, 5), len(rd))
            add_variant_or_new(rd, surf, "sudachi", 0.0, tail)
        print(f"[info] sudachi: +{len(entries) - n0:,}", flush=True)

    if not args.no_titles:
        n0 = len(entries)
        for rd, surf in iter_titles(args.raw_dir, args.max_titles, args.max_reading_len,
                                    1, args.max_surface_len):
            add_variant_or_new(rd, surf, "title", args.title_offset, (0,))
        print(f"[info] titles: +{len(entries) - n0:,}", flush=True)

    print(f"[info] candidates(all): pairs={len(entries):,}", flush=True)
    return entries


def selection_key(surf: str, e: Entry, policy: str) -> Tuple:
    """採用順のソートキー（小さいほど優先）。

    tier 0: コーパス観測。頻度の多い順（上限に収まらないときだけ後ろから削られる）。
    tier 1: 別読み。ソース順（Sudachi → NEologd → タイトル）。
    tier 2: 未観測。policy で並べ替える。
      balanced : 漢字あり → 表記長 → ソース → ソース固有信号
                 （表記長を先に見るので、短い一般語がソース横断で先に入る）
      source   : ソース優先（NEologd → Sudachi → タイトル）→ 漢字 → 表記長
      length   : 漢字あり → 表記長 のみ（ソースは最後）
    """
    if e.tier == 0:
        return (0, -e.count, 0, 0, (), surf)
    if e.tier == 1:
        return (1, e.src_order, 0, 0, (), surf)
    kanji_flag, src_order, length, tail = e.rank
    if policy == "source":
        return (2, src_order, kanji_flag, length, tail, surf)
    if policy == "length":
        return (2, kanji_flag, length, tail, src_order, surf)
    return (2, kanji_flag, length, src_order, tail, surf)


def select_entries(
    entries: Dict[Tuple[str, str], Entry], args: argparse.Namespace,
    max_readings: Optional[int] = None, max_size_mb: Optional[float] = None,
    topk: Optional[int] = None, policy: Optional[str] = None,
) -> Tuple[Dict[str, List[Tuple[str, Entry]]], Dict[str, int]]:
    """採用順に上限（読み数 / 行数 / サイズ）まで残し、読みごとに topk へ切る。

    上限: 端末の辞書ロードは読み数にほぼ比例する（14.5 万読みで 518ms）。
    """
    max_readings = args.max_readings if max_readings is None else max_readings
    max_size_mb = args.max_size_mb if max_size_mb is None else max_size_mb
    topk = args.topk if topk is None else topk
    policy = args.rank_policy if policy is None else policy
    max_bytes = int(max_size_mb * 1e6) if max_size_mb > 0 else 1 << 62

    order = sorted(entries.items(), key=lambda kv: selection_key(kv[0][1], kv[1], policy))
    kept: List[Tuple[str, str, Entry]] = []
    readings: set = set()
    per_reading: Dict[str, int] = defaultdict(int)
    n_rows = 0
    n_bytes = 0
    dropped_readings = dropped_rows = dropped_size = dropped_topk = 0
    for (rd, surf), e in order:
        if per_reading[rd] >= topk:
            dropped_topk += 1
            continue
        if n_rows >= args.max_rows:
            dropped_rows += 1
            continue
        new_reading = rd not in readings
        if new_reading and len(readings) >= max_readings:
            dropped_readings += 1
            continue
        # 実際の行のバイト数: 読み \t 表記 \t 頻度 \t コスト*100 \n
        est = (len(rd.encode("utf-8")) + len(surf.encode("utf-8"))
               + len(str(e.count)) + len(str(int(round(e.cost * 100)))) + 3)
        if n_bytes + est > max_bytes:
            dropped_size += 1
            continue
        kept.append((rd, surf, e))
        readings.add(rd)
        per_reading[rd] += 1
        n_rows += 1
        n_bytes += est
    print(f"[info] selected: rows={n_rows:,} readings={len(readings):,} "
          f"bytes~{n_bytes / 1e6:.2f}MB (policy={policy}, topk={topk}; drop "
          f"rows={dropped_rows:,} readings={dropped_readings:,} "
          f"size={dropped_size:,} topk={dropped_topk:,})", flush=True)

    by_reading: Dict[str, List[Tuple[str, Entry]]] = defaultdict(list)
    for rd, surf, e in kept:
        by_reading[rd].append((surf, e))
    out: Dict[str, List[Tuple[str, Entry]]] = {}
    for rd, lst in by_reading.items():
        # Lexicon / export_dict と同じ「コスト昇順」に並べ、読みあたり topk に切る。
        lst.sort(key=lambda it: (it[1].cost, -it[1].count, it[0]))
        out[rd] = lst[:topk]
    stats = {"rows": n_rows, "readings": len(readings), "bytes": n_bytes,
             "drop_rows": dropped_rows, "drop_readings": dropped_readings,
             "drop_size": dropped_size, "drop_topk": dropped_topk}
    return out, stats


# ------------------------------------------------------------------ 書き出し


def write_outputs(
    by_reading: Dict[str, List[Tuple[str, Entry]]], out_path: str, alt_path: str
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_rows = 0
    src_rows: Counter = Counter()
    with open(out_path, "w", encoding="utf-8") as f:
        for rd in sorted(by_reading):
            for surf, e in by_reading[rd]:
                cost = int(round(e.cost * 100))
                f.write(f"{rd}\t{surf}\t{e.count}\t{cost}\n")
                src_rows[e.source] += 1
                n_rows += 1

    # 表記 -> 読み一覧（学習時の読み替え拡張用）。2 読み以上ある表記だけ。
    by_surf: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    for rd, lst in by_reading.items():
        for surf, e in lst:
            by_surf[surf].append((e.cost, rd))
    n_alt = 0
    with open(alt_path, "w", encoding="utf-8") as f:
        for surf in sorted(by_surf):
            rds = [rd for _, rd in sorted(by_surf[surf])]
            if len(rds) < 2:
                continue
            f.write(f"{surf}\t{','.join(rds)}\n")
            n_alt += 1

    print(f"rows={n_rows:,} readings={len(by_reading):,} surfaces={len(by_surf):,} "
          f"alt_surfaces={n_alt:,}")
    print("source breakdown:", dict(src_rows.most_common()))
    print(f"-> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    print(f"-> {alt_path} ({os.path.getsize(alt_path) / 1e6:.2f} MB)")


# ------------------------------------------------------------------ 検証


def report_valid(lex_path: str, valid_path: str, topk: int, limit: int) -> None:
    """gold span coverage と候補選択精度（モデル無しの unigram ベース）を出す。

    - span_in_dict          : 正解スパンの (読み,表記) が辞書にある率
    - surface_in_candidates : 正解表記が k 候補に入る率
    - mincost_acc           : 正解分割を与えたとき最小コスト候補が正解と一致する率
                              （train/evaluate.py の base_hit / cover と同じ定義）
    - viterbi               : 読み全体を Viterbi で最小コスト探索した文正解率
    """
    from train.conv import Lexicon, build_arcs, to_surfaces, viterbi

    ents: Dict[str, List[Tuple]] = defaultdict(list)
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            if len(p) >= 4:
                ents[p[0]].append((p[1], int(p[2]), int(p[3]) / 100.0))
            else:
                ents[p[0]].append((p[1], int(p[2])))
    lex = Lexicon(dict(ents), topk=topk)
    pairs = {(rd, s) for rd, lst in ents.items() for s, *_ in lst}

    n = n_tok = n_span = n_surf = n_cand = n_mincost = 0
    n_sent = n_sent_cov = n_vit = 0
    cand_counts: Counter = Counter()
    for rd in lex.entries:
        cand_counts[min(len(lex.entries[rd]), 99)] += 1
    for line in open(valid_path, encoding="utf-8"):
        if n >= limit:
            break
        rec = json.loads(line)
        n += 1
        read, words = rec["read"], rec["words"]
        gold = []
        pos = 0
        for grd, gsurf in words:
            gold.append((pos, pos + len(grd), read[pos:pos + len(grd)], gsurf))
            pos += len(grd)
        span_map = {(i, j): c for i, j, c in lex.spans(read)}
        ok_all = True
        for s, e, grd, gsurf in gold:
            n_tok += 1
            if (grd, gsurf) in pairs:
                n_span += 1
            else:
                ok_all = False
            cands = span_map.get((s, e))
            if cands:
                n_cand += 1
                if any(c.surface == gsurf for c in cands):
                    n_surf += 1
                best = min(cands, key=lambda c: (c.cost, -c.count))
                if best.surface == gsurf:
                    n_mincost += 1
        if all((s, e) in span_map for s, e, _, _ in gold):
            n_sent_cov += 1
        if ok_all:
            _, path = viterbi(read, build_arcs(lex, read))
            if to_surfaces(path) == "".join(w[1] for w in words):
                n_vit += 1
        n_sent += 1

    tot_c = sum(k * v for k, v in cand_counts.items() if k < 99)
    hist = [cand_counts.get(i, 0) for i in range(1, topk + 1)]
    print(f"\n=== {lex_path} vs {os.path.basename(valid_path)} ({n} 文, k={topk}) ===")
    print(f"span_in_dict          : {n_span / max(n_tok, 1):.4f} ({n_span:,}/{n_tok:,})")
    print(f"surface_in_candidates : {n_surf / max(n_tok, 1):.4f} ({n_surf:,}/{n_tok:,})")
    print(f"sent full coverage    : {n_sent_cov / max(n_sent, 1):.4f}")
    print(f"mincost_acc (gold)    : {n_mincost / max(n_cand, 1):.4f} ({n_mincost:,}/{n_cand:,})")
    print(f"sent acc (viterbi)    : {n_vit / max(n_sent, 1):.4f} ({n_vit:,}/{n_sent:,})")
    print(f"readings={len(lex.entries):,} avg_cands={tot_c / max(len(lex.entries), 1):.2f} "
          f"hist(1..{topk})={hist}")


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-lexicon", default="data/lexicon.tsv",
                    help="(a) コーパス辞書（読み/表記/頻度の 3 列）")
    ap.add_argument("--out", default="data/lexicon_v3.tsv")
    ap.add_argument("--alt-out", default="data/alt_readings_v3.tsv")
    ap.add_argument("--sudachi-dict", default="full", choices=["full", "core"])
    ap.add_argument("--neologd-dir", default="data/raw/neologd/seed")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--no-sudachi", action="store_true")
    ap.add_argument("--no-neologd", action="store_true")
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--neologd-exclude", default=",".join(NEOLOGD_EXCLUDE_PATTERNS),
                    help="NEologd seed のうち除外するファイル名部分（カンマ区切り）")
    ap.add_argument("--variant-penalty", type=float, default=6.0,
                    help="コーパス既知表記の未観測読みに足すペナルティ（v2 は 4.5。"
                         "実測では 6.0 が最良、効果は僅か）")
    ap.add_argument("--default-cost", type=float, default=18.0,
                    help="コーパス未観測表記のコスト。目安は -log(1/総トークン数) 前後 "
                         "(旧 2,368 万トークンなら ≒17〜18)。14.5 では観測 1 回の語に"
                         "負けて候補選択が濁る")
    ap.add_argument("--neologd-offset", type=float, default=0.0)
    ap.add_argument("--title-offset", type=float, default=0.5,
                    help="タイトル由来は読みの確度が低いので少し高くする")
    ap.add_argument("--topk", type=int, default=8, help="読みあたり候補上限（k=8 前提）")
    ap.add_argument("--min-surface-len", type=int, default=2,
                    help="未観測表記の最小長（コーパス観測分には適用しない）")
    ap.add_argument("--max-surface-len", type=int, default=MAX_SURFACE_LEN_DEFAULT)
    ap.add_argument("--max-reading-len", type=int, default=24)
    ap.add_argument("--min-freq", type=int, default=1, help="コーパス側の最小頻度")
    ap.add_argument("--max-titles", type=int, default=400_000,
                    help="タイトル由来の上限（記事長の降順で採用）")
    ap.add_argument("--max-readings", type=int, default=500_000,
                    help="出力の読み数上限（端末ロード時間の支配要因）")
    ap.add_argument("--max-rows", type=int, default=1_200_000, help="出力の行数上限")
    ap.add_argument("--max-size-mb", type=float, default=30.0,
                    help="出力 TSV のサイズ上限（MB）")
    ap.add_argument("--rank-policy", default="balanced", choices=["balanced", "source", "length"],
                    help="未観測エントリの採用順: balanced=漢字+ソース混合 / source=ソース優先 / "
                         "length=表記長優先")
    ap.add_argument("--valid", default="", help="指定すると生成後に検証レポートを出す")
    ap.add_argument("--valid-limit", type=int, default=2000)
    args = ap.parse_args()
    args.neologd_exclude = [s for s in args.neologd_exclude.split(",") if s]

    corpus, total, surf_total = load_corpus(args.corpus_lexicon, args.min_freq)
    print(f"[info] corpus: pairs={len(corpus):,} surfaces={len(surf_total):,} "
          f"tokens={total:,}", flush=True)

    entries = build_entries(corpus, total, surf_total, args)
    by_reading, stats = select_entries(entries, args)
    write_outputs(by_reading, args.out, args.alt_out)
    if args.valid:
        report_valid(args.out, args.valid, args.topk, args.valid_limit)


if __name__ == "__main__":
    main()
