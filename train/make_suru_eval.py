"""「する」接続の曖昧性を集めた評価セットを作る（P0）。

同じ読みに「サ変可能な名詞」と「そうでない名詞」が候補として並び、直後に
`する / し / さ / せ / す` が来る箇所を valid から抽出する。非サ変の表記を選ぶと
「記者する」「内臓する」のような文法的に壊れた出力になるので、現行モデルが
連接の不自然さを判定できているかを測るのに使う。

  python -m train.make_suru_eval --limit 300
  # => data/eval/suru_amb.jsonl

評価ハーネス（train.evaluate_v2 / train.cli eval）と同じ
{"ctx", "read", "words"} 形式で書き出す。

条件:
  * 直後の語が `する` の活用形（する / し / さ / せ / す / しろ / しよう）
  * その名詞（正解表記）の Sudachi POS が「サ変可能」
  * 同じ読みの候補（最大 k=8）に「サ変可能でない表記」が入っている
  * 誤候補のコストが正解コスト + --max-gap 以内（競合するものだけ残す）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.conv import Lexicon  # noqa: E402
from train.train import load_lexicon  # noqa: E402

SURU_FORMS = {"する", "し", "さ", "せ", "す", "しろ", "しよう", "しよ", "します"}

# 手作りの最小ケース（コーパスに gold が無い語でも判定できるかを見る）
HANDMADE = [
    {"ctx": "", "read": "きしゃする",
     "words": [["きしゃ", "帰社"], ["する", "する"]], "src": "handmade"},
    {"ctx": "", "read": "きしゃし",
     "words": [["きしゃ", "帰社"], ["し", "し"]], "src": "handmade"},
    {"ctx": "", "read": "ないぞうする",
     "words": [["ないぞう", "内蔵"], ["する", "する"]], "src": "handmade"},
]


def _pos_of(tk, split_mode, cache: Dict[str, Tuple[str, ...]], surface: str) -> Tuple[str, ...]:
    if surface in cache:
        return cache[surface]
    try:
        ms = tk.tokenize(surface, split_mode)
        pos = ms[0].part_of_speech() if ms else ()
    except Exception:
        pos = ()
    cache[surface] = pos
    return pos


def main() -> None:
    ap = argparse.ArgumentParser(description="する接続の曖昧性を集めた評価セットを作る")
    ap.add_argument("--input", default="data/valid_v3.jsonl")
    ap.add_argument("--out", default="data/eval/suru_amb.jsonl")
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-gap", type=float, default=12.0,
                    help="誤候補と正解のコスト差の上限（nat）")
    ap.add_argument("--handmade", action="store_true", default=True,
                    help="手作りの最小ケースを先頭に足す")
    ap.add_argument("--no-handmade", dest="handmade", action="store_false")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"入力が見つかりません: {args.input}")

    from sudachipy import dictionary, tokenizer

    lex: Lexicon = load_lexicon(args.lexicon, 8)
    tk = dictionary.Dictionary(dict="full").tokenizer()
    split_mode = tokenizer.Tokenizer.SplitMode.C
    pos_cache: Dict[str, Tuple[str, ...]] = {}

    rows: List[dict] = []
    if args.handmade:
        rows.extend(HANDMADE)

    gaps: List[float] = []
    n_scanned = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if len(rows) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            words = rec.get("words") or []
            if len(words) < 3:
                continue
            n_scanned += 1
            for i in range(len(words) - 1):
                rd, gold = words[i]
                nxt = words[i + 1][1]
                if nxt not in SURU_FORMS:
                    continue
                if "サ変可能" not in _pos_of(tk, split_mode, pos_cache, gold):
                    continue
                cands = lex.entries.get(rd, [])
                gold_c = next((c for c in cands if c.surface == gold), None)
                if gold_c is None:
                    continue
                bad = [
                    c for c in cands
                    if c.surface != gold
                    and "サ変可能" not in _pos_of(tk, split_mode, pos_cache, c.surface)
                ]
                if not bad:
                    continue
                worst = min(bad, key=lambda c: c.cost)
                gap = worst.cost - gold_c.cost
                if gap > args.max_gap:
                    continue
                out = {
                    "ctx": rec.get("ctx", ""),
                    "read": rec.get("read") or "".join(w[0] for w in words),
                    "words": words,
                    "src": "valid",
                }
                rows.append(out)
                gaps.append(gap)
                break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"wrote {args.out}: {len(rows)} 文"
        f"（scanned={n_scanned}, handmade={len(HANDMADE) if args.handmade else 0}）"
    )
    if gaps:
        gaps.sort()
        print(
            "cost gap (誤候補 - 正解): "
            f"min={gaps[0]:.1f} p50={gaps[len(gaps) // 2]:.1f} "
            f"max={gaps[-1]:.1f}（nat）"
        )


if __name__ == "__main__":
    main()
