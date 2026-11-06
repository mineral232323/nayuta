"""日本語コーパスを追加取得する（Wikipedia 全量 / web テキスト / 話し言葉）。

学習データの元になる日本語テキストを集める。取得済みのファイルは飛ばすので、
何度実行しても安全（冪等）。取得元とライセンスは `data/corpus/MANIFEST.md` に
自動で書き出す。ライセンスが再配布可能でないものは取得しない。

使い方:
  # Wikipedia 20231101.ja の全 15 シャード（合計 3.94GB）を data/raw/ に置く
  python -m train.fetch_corpus --what wiki

  # mC4 (allenai/c4 config=ja) を streaming で読み、品質フィルタを通して
  # data/corpus/web/*.jsonl.gz に 250 万文つくる
  python -m train.fetch_corpus --what web

  # 話し言葉（国会会議録）を data/corpus/sns/*.jsonl.gz に 100 万文つくる
  python -m train.fetch_corpus --what sns

  # 全部まとめて + MANIFEST 更新
  python -m train.fetch_corpus --what all

  # 取得済みデータから MANIFEST だけ作り直す
  python -m train.fetch_corpus --what manifest

出力:
  data/raw/train-0000X-of-00015.parquet   Wikipedia（prep_data.py が読む）
  data/corpus/web/*.jsonl.gz              web テキスト（{"text": 1文} × 1行）
  data/corpus/sns/*.jsonl.gz              話し言葉（同上）
  data/corpus/_stats.json                 件数などの集計（MANIFEST の元ネタ）
  data/corpus/MANIFEST.md                 出典・ライセンス・文数の一覧
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Iterable, Iterator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.conv import is_japanese_char  # noqa: E402

# ---------------------------------------------------------------- 定数

WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.ja"
WIKI_SHARDS = 15
WIKI_LICENSE = "CC BY-SA 3.0 / GFDL（Wikipedia 日本語版）"
WIKI_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"

# シャードごとの実サイズ（バイト）。欠損・途中ダウンロードの検出に使う。
WIKI_SIZES = {
    0: 611504422, 1: 383148798, 2: 304603238, 3: 252027908, 4: 240133299,
    5: 240414553, 6: 233075429, 7: 217654475, 8: 224362017, 9: 234568965,
    10: 217186356, 11: 213095924, 12: 201165492, 13: 192710096, 14: 176347554,
}

WEB_REPO = "allenai/c4"
WEB_CONFIG = "ja"
WEB_LICENSE = "ODC-BY 1.0（C4 / mC4）"
WEB_URL = "https://huggingface.co/datasets/allenai/c4"

SNS_REPO = "Soma8622/kokkai_speech"
SNS_FILE = "all_speeches.parquet"
SNS_LICENSE = "MIT（リパッケージ）／原データは国会会議録検索システム（公共データ利用規約 1.0 ≒ CC BY 4.0）"
SNS_URL = "https://huggingface.co/datasets/Soma8622/kokkai_speech"

# 再配布可能と判断できるライセンス。ここに無いものは取得しない。
LICENSE_OK = {
    "cc-by-4.0", "cc-by-3.0", "cc-by-sa-4.0", "cc-by-sa-3.0", "cc0-1.0",
    "apache-2.0", "odc-by", "mit", "bsd-3-clause",
}

# ---------------------------------------------------------------- 品質フィルタ

MIN_LEN = 8
MAX_LEN = 70
JA_RATIO = 0.7
MAX_SYMBOL_RATIO = 0.25

# URL・HTML・実体参照・制御文字・ゼロ幅文字は文ごと捨てる
URL_RE = re.compile(
    r"(?:https?://|ftp://|www\.)\S*"
    r"|[\w-]+\.(?:com|jp|net|org|info|io|co|ne|or|ac|go|tv|biz|me|ly|gl)"
    r"(?:/\S*)?",
    re.IGNORECASE,
)
HTML_RE = re.compile(r"<[^<>\n]{1,60}>|&(?:[a-z]{2,10}|#\d{1,5}|#x[0-9a-f]{1,4});", re.IGNORECASE)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]")
MARKUP_RE = re.compile(r"(==+|\{\{|\}\}|\||｜|```|\t|^\s*[#*>]+\s*|\[|\]|»|‹|›)")
# メール・メンション・長いハッシュタグはノイズ
MENTION_RE = re.compile(r"[@＠][A-Za-z0-9_]{2,}|[#＃][^\s#＃]{12,}")
# 文末に並ぶ記号の連打（例: !!!、???、☆★☆★、w の連続）
SYMBOL_RUN_RE = re.compile(r"([^\w\s])\1{2,}|([wｗ]{4,})")

TRAIL = " 　\t。、,.!?！？・…‥"

# 生成テキスト・定型文くさい断片（web クロール由来のナビ/SEO 文言を含む）
BOILERPLATE_RE = re.compile(
    r"(ステップバイステップ|以下に(?:示|説明|まと)|ご注意ください|"
    r"このサイトは|全ての著作権|プライバシーポリシー|免責事項|"
    r"ログイン|会員登録|カートに入れる|送料無料|"
    r"続きを読む|この記事|人気記事|新着記事|記事一覧|サイト内|"
    r"トップページ|ページの先頭|前の記事|次の記事|当サイト|このページ|"
    r"お問い合わせ|引用元|転載を禁|無断転載|"
    r"ChatGPT|OpenAI|Gemini|Claude|AIが生成|自動生成)"
)

# web クロールに混ざる出会い系・アダルト広告のスパム文（一般語ではなく広告文言だけを落とす）
AD_SPAM_RE = re.compile(
    r"(出会い系|出会いサイト|セフレ|パパ活|逆援助|デリヘル|ソープランド|"
    r"風俗店|無修正動画|エロ動画|アダルトサイト|AV女優|オナニー|中出し|"
    r"無料登録|登録不要で|今すぐ無料)"
)

# 国会会議録の行頭に付く発言者ラベル（例: ○委員長（鈴木和美君）　本文）
KOKKAI_SPEAKER_RE = re.compile(r"^[○◯●◎]\s*\S{0,60}?[　\s]")
# 国会会議録の進行メモ・日付見出し（本文ではない）
KOKKAI_META_RE = re.compile(
    r"^(平成|昭和|令和|午前|午後|出席者|議事日程|本日の会議|委員の異動|"
    r"辞任|補欠選任|〔|――|会議の経過|目次)"
)
HIRA_MIN = 0.10  # ひらがな比率の下限（人名の羅列などを落とす）

# 記号ばかりの文を落とすための文字集合
SYMBOLS = set("!?！？♪☆★●○◆◇■□▲△▼▽※→←↑↓…‥・,，.．:：;；/／\\|｜~～^＾_＿`'\"”“‘’()（）[]［］{}｛｝<>＜＞+=＊*#＃@＠$＆%％&")


def clean_sentence(s: str) -> str:
    """1 文を正規化する（NFKC・前後トリム・内部の空白除去）。"""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().strip(TRAIL)
    # prep_data.py の clean() と同じく、文内の空白は全部落とす
    return re.sub(r"\s+", "", s)


def japanese_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if is_japanese_char(c)) / len(s)


def symbol_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if c in SYMBOLS) / len(s)


def has_repeated_ngram(s: str, k: int = 8) -> bool:
    """同じ 8 文字の並びが 2 回以上出てきたら「生成テキストの繰り返し」とみなす。"""
    if len(s) < k * 2:
        return False
    seen: set[str] = set()
    for i in range(len(s) - k + 1):
        g = s[i : i + k]
        if g in seen:
            return True
        seen.add(g)
    return False


def strip_speaker_prefix(line: str) -> str:
    """国会会議録の「○役職（氏名）　本文」から発言者ラベルを落とす。

    ラベルと本文が空白で区切られていない行は、本文を切り出せないので捨てる。
    """
    if not line or line[0] not in "○◯●◎":
        return line
    m = KOKKAI_SPEAKER_RE.match(line)
    if not m:
        return ""
    return line[m.end():].lstrip("　 \t")


def hiragana_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for c in s if 0x3041 <= ord(c) <= 0x309F) / len(s)


def ok_kokkai_sentence(s: str) -> bool:
    """議事録特有のノイズ（発言者名の羅列・進行メモ）を落とす。"""
    if "君" in s or "○" in s or "◯" in s or "〔" in s or "〕" in s:
        return False
    if KOKKAI_META_RE.match(s):
        return False
    if hiragana_ratio(s) < HIRA_MIN:
        return False
    return True


def ok_sentence(s: str, source: str = "web") -> bool:
    """1 文が学習データに使えるかを判定する。

    source="kokkai" のときだけ、議事録特有の進行メモ・発言者名の羅列を落とす
    追加条件を適用する（web では「君」などの一般語を消さないため）。
    """
    if not (MIN_LEN <= len(s) <= MAX_LEN):
        return False
    if source == "kokkai" and not ok_kokkai_sentence(s):
        return False
    if CTRL_RE.search(s):
        return False
    if URL_RE.search(s) or HTML_RE.search(s):
        return False
    if MARKUP_RE.search(s) or MENTION_RE.search(s):
        return False
    if BOILERPLATE_RE.search(s):
        return False
    if AD_SPAM_RE.search(s):
        return False
    if SYMBOL_RUN_RE.search(s):
        return False
    if symbol_ratio(s) > MAX_SYMBOL_RATIO:
        return False
    if japanese_ratio(s) < JA_RATIO:
        return False
    # 日本語文字が 1 つも無い文（数字・英字だけ）は落とす
    if not any(is_japanese_char(c) for c in s):
        return False
    # 同じ文字の 3 連続（「ああああ」「!!!!」など）は記号連打として落とす
    for i in range(len(s) - 2):
        if s[i] == s[i + 1] == s[i + 2] and not ("0" <= s[i] <= "9"):
            return False
    # 「。、」だけの文や句読点の詰まった文
    if sum(1 for c in s if c in "。、,.!?！？") > MAX_LEN // 3:
        return False
    if has_repeated_ngram(s):
        return False
    return True


def split_sentences(par: str) -> list[str]:
    """段落を文に分割する（prep_data.py と同じ規則）。"""
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(par)
    while i < n:
        ch = par[i]
        buf.append(ch)
        i += 1
        if ch in "。！？!?":
            while i < n and par[i] in "」』）)】〉》\"'":
                buf.append(par[i])
                i += 1
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return out


def iter_sentences(text: str, source: str = "web") -> Iterator[str]:
    """本文（段落入り）からフィルタ通過した文だけを順に返す。"""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if source == "kokkai":
            para = strip_speaker_prefix(para)
            if not para:
                continue
        for sent in split_sentences(para):
            s = clean_sentence(sent)
            if ok_sentence(s, source):
                yield s


def dedupe_key(s: str) -> str:
    """完全重複の判定キー（NFKC 済み・空白無しの文字列）。"""
    return unicodedata.normalize("NFKC", s)


# ---------------------------------------------------------------- 出力まわり


def _now_jst() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M JST")


def _human_size(n: int) -> str:
    """バイト数を 1000 進で読みやすくする（HF の表示や公称サイズと揃える）。"""
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1000 or unit == "TB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1000.0
    return f"{v:.1f} TB"


class CorpusWriter:
    """`{"text": 文}` を 1 行 1 文で gzip に書く。指定文数ごとにファイルを分ける。"""

    def __init__(self, out_dir: str, prefix: str, shard_sentences: int = 500_000) -> None:
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_sentences = shard_sentences
        self.shards: list[dict] = []
        self.total = 0
        self._fh = None
        self._path = ""
        self._n = 0
        os.makedirs(out_dir, exist_ok=True)

    def _open(self) -> None:
        idx = len(self.shards)
        self._path = os.path.join(self.out_dir, f"{self.prefix}-{idx:05d}.jsonl.gz")
        self._fh = gzip.open(self._path, "wt", encoding="utf-8", compresslevel=6)
        self._n = 0

    def write(self, text: str) -> None:
        if self._fh is None:
            self._open()
        self._fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self._n += 1
        self.total += 1
        if self._n >= self.shard_sentences:
            self.close_shard()

    def close_shard(self) -> None:
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        self.shards.append(
            {"path": self._path, "bytes": os.path.getsize(self._path), "sentences": self._n}
        )
        print(f"    -> {self._path} ({self._n} 文, {_human_size(os.path.getsize(self._path))})",
              flush=True)

    def close(self) -> None:
        self.close_shard()


def load_stats(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def save_stats(path: str, stats: dict) -> None:
    """`_stats.json` を更新する。並行実行時に他のキーを消さないようマージする。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merged = load_stats(path)
    merged.update(stats)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- Wikipedia


def _hf_download(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=local_dir
    )


def fetch_wikipedia(raw_dir: str, verify: bool = True) -> dict:
    """Wikipedia 日本語版 20231101 の全 15 シャードを data/raw/ 直下に置く。

    既存の `wiki_ja_00000.parquet` は shard 0 とバイト単位で同一なので、
    再ダウンロードせずリネームして使う。
    """
    t_start = time.time()
    os.makedirs(raw_dir, exist_ok=True)
    legacy = os.path.join(raw_dir, "wiki_ja_00000.parquet")
    first = os.path.join(raw_dir, "train-00000-of-00015.parquet")
    if os.path.exists(legacy) and not os.path.exists(first):
        os.replace(legacy, first)
        print(f"  既存の {os.path.basename(legacy)} を shard 0 にリネーム", flush=True)

    files: list[dict] = []
    for i in range(WIKI_SHARDS):
        name = f"train-{i:05d}-of-00015.parquet"
        dest = os.path.join(raw_dir, name)
        want = WIKI_SIZES[i]
        if os.path.exists(dest) and os.path.getsize(dest) == want:
            print(f"  [skip] {name} ({_human_size(want)})", flush=True)
        else:
            print(f"  [get ] {name} ({_human_size(want)})", flush=True)
            t0 = time.time()
            p = _hf_download(WIKI_REPO, f"{WIKI_CONFIG}/{name}", raw_dir)
            if os.path.abspath(p) != os.path.abspath(dest):
                os.replace(p, dest)
            got = os.path.getsize(dest)
            if got != want:
                raise SystemExit(f"{name}: サイズ不一致 {got} != {want}（再実行してください）")
            print(f"         ok {_human_size(got)} / {time.time() - t0:.1f}s", flush=True)

    # 直下に余計な parquet が無いことを確認する（prep_data は data/raw/*.parquet を全部読む）
    extra = [
        f for f in sorted(os.listdir(raw_dir))
        if f.endswith(".parquet") and not re.fullmatch(r"train-\d{5}-of-00015\.parquet", f)
    ]
    if extra:
        raise SystemExit(f"data/raw に想定外の parquet があります: {extra}")

    if verify:
        files = verify_wikipedia(raw_dir)

    # hf_hub が作る空ディレクトリを片付ける
    nested = os.path.join(raw_dir, WIKI_CONFIG)
    if os.path.isdir(nested) and not os.listdir(nested):
        os.rmdir(nested)

    total_rows = sum(f["rows"] for f in files) if files else 0
    print(f"  Wikipedia: {len(files)} シャード / {total_rows} 記事 / "
          f"{time.time() - t_start:.1f}s", flush=True)
    return {
        "source": "Wikipedia 日本語版（wikimedia/wikipedia）",
        "config": WIKI_CONFIG,
        "url": WIKI_URL,
        "license": WIKI_LICENSE,
        "fetched_at": _now_jst(),
        "files": files,
        "rows": total_rows,
    }


def verify_wikipedia(raw_dir: str) -> list[dict]:
    """各シャードの行数と title/text 列の存在を確認する。"""
    import pyarrow.parquet as pq

    out: list[dict] = []
    for i in range(WIKI_SHARDS):
        name = f"train-{i:05d}-of-00015.parquet"
        path = os.path.join(raw_dir, name)
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        if "text" not in cols or "title" not in cols:
            raise SystemExit(f"{name}: text/title 列がありません -> {cols}")
        out.append({
            "path": path,
            "bytes": os.path.getsize(path),
            "rows": pf.metadata.num_rows,
            "columns": cols,
        })
        print(f"    {name}: {pf.metadata.num_rows} 行, 列={cols}", flush=True)
    return out


# ---------------------------------------------------------------- web


def check_license(repo_id: str) -> str:
    """HF のデータセットカードからライセンスを読み、再配布可能か確認する。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo_id}") as r:
            meta = json.load(r)
    except Exception as e:  # ネットワーク断などは取得を止める
        raise SystemExit(f"{repo_id}: ライセンス確認に失敗しました ({e})")
    lic = (meta.get("cardData") or {}).get("license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    lic = str(lic or "").lower()
    if lic not in LICENSE_OK:
        raise SystemExit(
            f"{repo_id}: ライセンス {lic!r} は再配布可能と判断できないため取得しません"
            f"（許可: {sorted(LICENSE_OK)}）"
        )
    return lic


def iter_web_docs(limit_docs: int = 0) -> Iterator[str]:
    """allenai/c4 (config=ja) を streaming で読み、本文だけを返す。"""
    from datasets import load_dataset

    ds = load_dataset(WEB_REPO, WEB_CONFIG, split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit_docs and i >= limit_docs:
            return
        text = row.get("text") or ""
        if text:
            yield text


def fetch_web(out_dir: str, max_sentences: int, max_docs: int, shard_sentences: int,
              force: bool = False) -> dict:
    """mC4 (ja) から品質フィルタ済みの文を集める。"""
    stats_path = os.path.join(os.path.dirname(out_dir), "_stats.json")
    stats = load_stats(stats_path)
    if not force and stats.get("web", {}).get("shards"):
        w = stats["web"]
        print(f"  web: 取得済み ({w.get('sentences')} 文) のためスキップ "
              f"(--force で作り直し)", flush=True)
        return w

    print(f"  web: {WEB_REPO}/{WEB_CONFIG} を streaming で読みます "
          f"(目標 {max_sentences} 文, 入力上限 {max_docs or '無し'} doc)", flush=True)
    for f in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if f.endswith(".jsonl.gz"):
            os.remove(os.path.join(out_dir, f))

    writer = CorpusWriter(out_dir, "web", shard_sentences)
    seen: set[str] = set()
    docs = 0
    dropped = 0
    t0 = time.time()
    last = t0
    try:
        for text in iter_web_docs(max_docs):
            docs += 1
            sents = list(iter_sentences(text))
            # 定型文の羅列（同一文の繰り返し）は文書ごと捨てる
            if len(sents) >= 8 and len(set(sents)) / len(sents) < 0.7:
                dropped += len(sents)
                sents = []
            for s in sents:
                key = dedupe_key(s)
                h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
                if h in seen:
                    continue
                seen.add(h)
                writer.write(s)
                if writer.total >= max_sentences:
                    break
            now = time.time()
            if docs % 20_000 == 0 or now - last > 60:
                last = now
                rate = docs / max(now - t0, 1e-6)
                print(f"    docs={docs} sentences={writer.total} "
                      f"dup/定型={dropped} {rate:.0f} docs/s", flush=True)
            if writer.total >= max_sentences:
                break
    finally:
        writer.close()
    elapsed = time.time() - t0
    print(f"  web: {writer.total} 文 / {docs} docs / {elapsed:.0f}s", flush=True)
    out = {
        "source": "mC4 (allenai/c4, config=ja) streaming",
        "url": WEB_URL,
        "license": WEB_LICENSE,
        "fetched_at": _now_jst(),
        "docs_read": docs,
        "sentences": writer.total,
        "shards": writer.shards,
        "filter": {
            "min_len": MIN_LEN, "max_len": MAX_LEN, "ja_ratio": JA_RATIO,
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "dedupe": "NFKC 正規化後の完全一致 (blake2b 64bit)",
            "drop": "URL/HTML/制御文字/記号連打/定型文/同一 8-gram の繰り返し",
        },
    }
    stats["web"] = out
    save_stats(stats_path, stats)
    return out


# ---------------------------------------------------------------- 話し言葉


def _pick_text_column(names: Iterable[str]) -> str:
    cands = ["speech", "text", "content", "body", "utterance", "発言", "本文", "sentence"]
    lower = {n.lower(): n for n in names}
    for c in cands:
        if c in lower:
            return lower[c]
    raise SystemExit(f"本文らしき列が見つかりません: {list(names)}")


def fetch_sns(out_dir: str, max_sentences: int, shard_sentences: int,
              repo: str = SNS_REPO, filename: str = SNS_FILE,
              cache_dir: str = "data/cache/sns", force: bool = False) -> dict:
    """話し言葉コーパス（国会会議録）から品質フィルタ済みの文を集める。

    ライセンスを HF API で確認してから取得する。再配布不可と判断したら止まる。
    """
    import pyarrow.parquet as pq

    stats_path = os.path.join(os.path.dirname(out_dir), "_stats.json")
    stats = load_stats(stats_path)
    if not force and stats.get("sns", {}).get("shards"):
        s = stats["sns"]
        print(f"  sns: 取得済み ({s.get('sentences')} 文) のためスキップ "
              f"(--force で作り直し)", flush=True)
        return s

    lic = check_license(repo)
    print(f"  sns: {repo} (license={lic}) を取得します", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    path = _hf_download(repo, filename, cache_dir)

    pf = pq.ParquetFile(path)
    col = _pick_text_column(pf.schema_arrow.names)
    print(f"    columns={pf.schema_arrow.names} -> text='{col}'", flush=True)

    for f in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if f.endswith(".jsonl.gz"):
            os.remove(os.path.join(out_dir, f))

    writer = CorpusWriter(out_dir, "sns", shard_sentences)
    seen: set[str] = set()
    rows = 0
    t0 = time.time()
    try:
        for batch in pf.iter_batches(batch_size=2000, columns=[col]):
            for text in batch[col].to_pylist():
                rows += 1
                if not text:
                    continue
                for s in iter_sentences(text, "kokkai"):
                    h = hashlib.blake2b(dedupe_key(s).encode("utf-8"), digest_size=8).digest()
                    if h in seen:
                        continue
                    seen.add(h)
                    writer.write(s)
                    if writer.total >= max_sentences:
                        break
            if rows % 20_000 == 0:
                print(f"    rows={rows} sentences={writer.total}", flush=True)
            if writer.total >= max_sentences:
                break
    finally:
        writer.close()
    print(f"  sns: {writer.total} 文 / {rows} 行 / {time.time() - t0:.0f}s", flush=True)
    out = {
        "source": "国会会議録（Soma8622/kokkai_speech の all_speeches.parquet）",
        "url": SNS_URL,
        "license": SNS_LICENSE,
        "license_tag": lic,
        "fetched_at": _now_jst(),
        "rows_read": rows,
        "sentences": writer.total,
        "shards": writer.shards,
        "filter": {
            "min_len": MIN_LEN, "max_len": MAX_LEN, "ja_ratio": JA_RATIO,
            "max_symbol_ratio": MAX_SYMBOL_RATIO,
            "dedupe": "NFKC 正規化後の完全一致 (blake2b 64bit)",
        },
    }
    stats["sns"] = out
    save_stats(stats_path, stats)
    return out


# ---------------------------------------------------------------- MANIFEST


def _scan_shards(out_dir: str, prefix: str) -> list[dict]:
    """出力済みの jsonl.gz を実際に数えて、ファイル一覧と文数を返す。

    MANIFEST は「今ディスクにあるもの」を反映させたいので、_stats.json ではなく
    実ファイルを数える（手で足し引きしても表がずれない）。
    """
    out: list[dict] = []
    if not os.path.isdir(out_dir):
        return out
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith(prefix) and name.endswith(".jsonl.gz")):
            continue
        path = os.path.join(out_dir, name)
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for _ in f:
                n += 1
        out.append({"path": path, "bytes": os.path.getsize(path), "sentences": n})
    return out


def write_manifest(corpus_dir: str) -> str:
    """`_stats.json` から MANIFEST.md を作る。"""
    stats = load_stats(os.path.join(corpus_dir, "_stats.json"))
    lines: list[str] = [
        "# コーパス一覧（MANIFEST）",
        "",
        f"最終更新: {_now_jst()}",
        "",
        "本ディレクトリの日本語テキストは、かな漢字変換モデルの学習データ作成に使う。",
        "各ソースの出典・ライセンス・文数は以下のとおり。NOTICE へはこの表を転記する。",
        "",
        "## ソース一覧",
        "",
        "| ソース | 出典 URL | ライセンス | 取得日 | パス | サイズ | 記事/行 | 文数 |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    rows: list[str] = []

    wiki = stats.get("wiki")
    if wiki:
        files = wiki.get("files") or []
        total_bytes = sum(f.get("bytes", 0) for f in files)
        total_rows = sum(f.get("rows", 0) for f in files)
        rows.append(
            f"| Wikipedia 日本語版 {wiki.get('config', '')} | {wiki.get('url')} | "
            f"{wiki.get('license')} | {wiki.get('fetched_at')} | "
            f"`data/raw/train-0000X-of-00015.parquet`（15 シャード） | "
            f"{_human_size(total_bytes)} | {total_rows} | - |"
        )
    for key, label, sub, prefix, path_note in (
        ("web", "mC4 (ja) web テキスト", "web", "web-", "`data/corpus/web/*.jsonl.gz`"),
        ("sns", "国会会議録（話し言葉）", "sns", "sns-", "`data/corpus/sns/*.jsonl.gz`"),
    ):
        s = stats.get(key)
        if not s:
            continue
        shards = _scan_shards(os.path.join(corpus_dir, sub), prefix) or (s.get("shards") or [])
        n_sent = sum(x.get("sentences", 0) for x in shards)
        rows.append(
            f"| {label} | {s.get('url')} | {s.get('license')} | {s.get('fetched_at')} | "
            f"{path_note}（{len(shards)} ファイル） | "
            f"{_human_size(sum(x.get('bytes', 0) for x in shards))} | "
            f"{s.get('rows_read', s.get('docs_read', '-'))} | {n_sent} |"
        )
    lines.extend(rows or ["| （未取得） | - | - | - | - | - | - | - |"])
    lines.append("")

    # ファイル単位の内訳（NOTICE に転記するとき用）
    wiki = stats.get("wiki") or {}
    if wiki.get("files"):
        lines += [
            "### Wikipedia のファイル内訳",
            "",
            "| ファイル | サイズ (bytes) | 記事数 |",
            "| --- | ---: | ---: |",
        ]
        for f in wiki["files"]:
            lines.append(
                f"| `{f['path']}` | {f.get('bytes', 0)} | {f.get('rows', 0)} |"
            )
        lines.append("")
    for label, sub, prefix in (
        ("web", "web", "web-"),
        ("sns", "sns", "sns-"),
    ):
        shards = _scan_shards(os.path.join(corpus_dir, sub), prefix)
        if not shards:
            continue
        lines += [
            f"### {label} のファイル内訳",
            "",
            "| ファイル | サイズ (bytes) | 文数 |",
            "| --- | ---: | ---: |",
        ]
        for x in shards:
            lines.append(
                f"| `{x['path']}` | {x['bytes']} | {x['sentences']} |"
            )
        lines.append("")

    # 品質フィルタ
    lines += [
        "## 品質フィルタ（web / 話し言葉 共通）",
        "",
        "`train/fetch_corpus.py` の `ok_sentence()`。1 行 1 文の `{\"text\": ...}` 形式で保存。",
        "",
        "| 条件 | 値 |",
        "| --- | --- |",
        f"| 文字数 | {MIN_LEN}〜{MAX_LEN} 文字（NFKC 正規化・空白除去後） |",
        f"| 日本語文字比率 | {JA_RATIO} 以上（ひらがな・カタカナ・漢字・々〆） |",
        f"| 記号比率の上限 | {MAX_SYMBOL_RATIO} |",
        "| 除去: URL / HTML タグ / 実体参照 | `URL_RE` / `HTML_RE` |",
        "| 除去: 制御文字・ゼロ幅文字・BOM | `CTRL_RE` |",
        "| 除去: 記号・絵文字の 3 連以上 / `wwww` | `SYMBOL_RUN_RE`、同一文字 3 連続 |",
        "| 除去: 定型文・SEO/ナビ文言（続きを読む・ログイン・カート・ChatGPT 等） | `BOILERPLATE_RE` |",
        "| 除去: 出会い系・アダルト広告の文言 | `AD_SPAM_RE` |",
        "| 除去: 括弧・パイプ・`»` などのナビ表記 | `MARKUP_RE` |",
        "| 除去: 同一 8-gram の繰り返し（生成文らしさ） | `has_repeated_ngram()` |",
        "| 除去: 文書内で同一文が 30% 超を占める定型ページ | `fetch_web()` |",
        "| 完全重複除去 | NFKC 正規化後の blake2b 64bit 一致を全コーパス共通で排除 |",
        "",
        "### 話し言葉（国会会議録）だけの追加フィルタ",
        "",
        "| 条件 | 内容 |",
        "| --- | --- |",
        "| 発言者ラベルの除去 | 行頭 `○役職（氏名）　本文` のラベルを落として本文だけ残す（`strip_speaker_prefix()`） |",
        "| 進行メモ・日付見出しの除去 | `KOKKAI_META_RE`（平成/昭和/令和・出席者・議事日程・辞任 など） |",
        "| 人名の羅列の除去 | `君` を含む文、ひらがな比率 10% 未満の文（`HIRA_MIN`） |",
        "",
    ]

    # 未採用・検討したソース
    lines += [
        "## 採用しなかったソースと理由",
        "",
        "| ソース | ライセンス | 不採用の理由 |",
        "| --- | --- | --- |",
        "| CC-100 (ja) | 不明瞭 | `datasets` 5.x で script-based dataset の読み込みが不可。web は mC4 で代替 |",
        "| community-datasets/covid_tweets_japanese | CC BY-ND 4.0 | ND（改変禁止）で学習利用の解釈が不明確、COVID 関連の偏りも大 |",
        "| kubota/defamation-japanese-twitter | CC BY 4.0 | 誹謗中傷の収集データで、IME 学習用テキストとして不適切 |",
        "| IngCrowd/japanese-themed-dialogue | CC BY-NC 4.0 | NC（非商用）で再配布条件が合わない |",
        "| cyberagent/chatbot-arena-ja-calm2-*-experimental | CC BY 4.0 | gated（利用同意必須）で HF_TOKEN 無しでは取得不可 |",
        "| shi3z/Japanese_Wikipedia_Conversation | CC BY-SA 4.0 | LLM 生成の対話で「生成テキスト除去」方針に反する |",
        "| OmniAICreator/Japanese-Roleplay-Dialogues | Apache-2.0 | LLM 生成のロールプレイで文面が不適切なものが混在 |",
        "| globis-university/aozorabunko-chats | CC BY 4.0 | 青空文庫からの生成対話で、やはり生成テキスト |",
        "",
        "→ 日本語 SNS コーパスは、再配布可能なライセンスのものが HF 上に見つからなかった。",
        "   話し言葉は国会会議録（議事録 = 実際の発話）で代替した。",
        "",
        "## ローカルキャッシュ（学習には使わない）",
        "",
        "| パス | 内容 |",
        "| --- | --- |",
        "| `data/cache/sns/all_speeches.parquet` | 国会会議録の元 parquet（3.1GB）。`--what sns` の再実行時に再ダウンロードを省くだけ |",
        "",
        "## 再生成方法",
        "",
        "```sh",
        "# 全部（冪等。取得済みはスキップ）",
        "python -m train.fetch_corpus --what all",
        "# 個別",
        "python -m train.fetch_corpus --what wiki",
        "python -m train.fetch_corpus --what web --web-max-sentences 2500000",
        "python -m train.fetch_corpus --what sns",
        "# MANIFEST だけ作り直す",
        "python -m train.fetch_corpus --what manifest",
        "```",
        "",
    ]

    path = os.path.join(corpus_dir, "MANIFEST.md")
    os.makedirs(corpus_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  MANIFEST -> {path}", flush=True)
    return path


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="all",
                    choices=["all", "wiki", "web", "sns", "manifest"],
                    help="取得する対象")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--corpus-dir", default="data/corpus")
    ap.add_argument("--web-max-sentences", type=int, default=2_500_000)
    ap.add_argument("--web-max-docs", type=int, default=0,
                    help="入力 doc 数の上限（0 は無制限）")
    ap.add_argument("--sns-max-sentences", type=int, default=1_000_000)
    ap.add_argument("--shard-sentences", type=int, default=500_000)
    ap.add_argument("--force", action="store_true", help="取得済みでも作り直す")
    args = ap.parse_args()

    stats_path = os.path.join(args.corpus_dir, "_stats.json")
    stats = load_stats(stats_path)
    web_dir = os.path.join(args.corpus_dir, "web")
    sns_dir = os.path.join(args.corpus_dir, "sns")

    what = args.what
    if what in ("all", "wiki"):
        prev = (stats.get("wiki") or {}).get("files") or []
        wiki = fetch_wikipedia(args.raw_dir)
        # 中身が変わっていなければ「取得日」は前回の値を保つ（再実行で MANIFEST が揺れない）
        same = [(f["path"], f["bytes"], f["rows"]) for f in prev] == [
            (f["path"], f["bytes"], f["rows"]) for f in wiki["files"]
        ]
        if same and (stats.get("wiki") or {}).get("fetched_at"):
            wiki["fetched_at"] = stats["wiki"]["fetched_at"]
        stats["wiki"] = wiki
        save_stats(stats_path, stats)
    if what in ("all", "web"):
        stats["web"] = fetch_web(web_dir, args.web_max_sentences, args.web_max_docs,
                                 args.shard_sentences, args.force)
        save_stats(stats_path, stats)
    if what in ("all", "sns"):
        stats["sns"] = fetch_sns(sns_dir, args.sns_max_sentences, args.shard_sentences,
                                 force=args.force)
        save_stats(stats_path, stats)
    if what in ("all", "manifest"):
        write_manifest(args.corpus_dir)

    print("done", flush=True)


if __name__ == "__main__":
    main()
