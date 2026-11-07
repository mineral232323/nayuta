"""文頭（入力の最初の語）の prior 負けケースを集めた評価セットを作る。

条件:
  * 文の先頭の語（words[0]）の読みに候補が 2 つ以上ある
  * 正解表記がその候補に入っている
  * **最もコストの低い候補（頻度 prior）が正解と異なる**（= 文脈なしでは prior 負けする）

既存の評価セット（wiki2000 / sns2000）から抽出し、固定の評価セットとして使う。

  python -m train.make_head_eval
  # => data/eval/head_prior_wrong.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import load_lexicon  # noqa: E402

SOURCES = [
    ("wiki", "data/eval/wiki2000.jsonl"),
    ("sns", "data/eval/sns2000.jsonl"),
]

# 文頭の代表例（手作り）。gold の words は連結が正解文になるようにする。
HANDMADE = [
    {
        "ctx": "",
        "read": "せんとうのもじのしっぱいりつたかくない",
        "words": [["せんとう", "先頭"], ["の", "の"], ["もじ", "文字"],
                  ["の", "の"], ["しっぱいりつ", "失敗率"], ["たかく", "高く"],
                  ["ない", "ない"]],
        "src": "handmade",
    },
    {
        "ctx": "",
        "read": "せんとうのひとがおおい",
        "words": [["せんとう", "先頭"], ["の", "の"], ["ひと", "人"],
                  ["が", "が"], ["おおい", "多い"]],
        "src": "handmade",
    },
    {
        "ctx": "",
        "read": "れつのいちばんまえをあるく",
        "words": [["れつ", "列"], ["の", "の"], ["いちばん", "一番"],
                  ["まえ", "前"], ["を", "を"], ["あるく", "歩く"]],
        "src": "handmade",
    },
]


def head_info(lex, words):
    """先頭語の (reading, gold, prior, candidates) と prior-wrong 判定を返す。"""
    if not words:
        return None
    rd0, gold0 = words[0]
    cands = lex.entries.get(rd0, [])
    surfaces = [c.surface for c in cands]
    if len(surfaces) < 2 or gold0 not in surfaces:
        return None
    prior = cands[0].surface
    return {
        "reading": rd0,
        "gold": gold0,
        "prior": prior,
        "candidates": surfaces,
        "prior_wrong": prior != gold0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="文頭 prior-wrong 評価セットの生成")
    ap.add_argument("--lexicon", default="data/lexicon_v3_frozen.tsv")
    ap.add_argument("--out", default="data/eval/head_prior_wrong.jsonl")
    ap.add_argument("--limit-per-source", type=int, default=0, help="各ソースの上限（0=全件）")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon, 8)
    rows = []
    counts = {"handmade": 0}
    for rec in HANDMADE:
        h = head_info(lex, rec.get("words") or [])
        if h is None or not h["prior_wrong"]:
            continue
        r = dict(rec)
        r["head"] = h
        rows.append(r)
        counts["handmade"] += 1
    for name, path in SOURCES:
        if not os.path.exists(path):
            print(f"skip: {path} がありません", file=sys.stderr)
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if args.limit_per_source and n >= args.limit_per_source:
                    break
                rec = json.loads(line)
                words = rec.get("words") or []
                if not words:
                    continue
                h = head_info(lex, words)
                if h is None or not h["prior_wrong"]:
                    continue
                out = {
                    "ctx": rec.get("ctx", ""),
                    "read": rec.get("read") or "".join(w[0] for w in words),
                    "words": words,
                    "src": name,
                    "head": h,
                }
                rows.append(out)
                n += 1
        counts[name] = n

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} 文")
    print("sources:", counts)
    print("先頭語の例:", [(r["head"]["gold"], r["head"]["prior"]) for r in rows[:5]])


if __name__ == "__main__":
    main()
