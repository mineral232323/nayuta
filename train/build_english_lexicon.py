"""カタカナ語 → 英単語の辞書を作る（JMdict 由来）。

「あんどろいど → Android」のように、カタカナの外来語を英単語に変換するための
対応表を作る。元データは JMdict（CC BY-SA 4.0）。

  python -m train.build_english_lexicon

出力:
  data/english_lexicon.tsv   読み(ひらがな) \t 英単語
"""

from __future__ import annotations

import argparse
import json
import re
import sys

KATA = re.compile(r"^[ァ-ヶー・]+$")
GLOSS_OK = re.compile(r"^[A-Za-z][A-Za-z .'\-]*$")

# 固有名詞・商標っぽいタグが付いていたら先頭を大文字にする
PROPER_FIELDS = {"tradem", "company", "product", "serv", "work"}
PROPER_MISC = {"company", "product", "serv"}


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jmdict", default="data/raw/jmdict_eng.json")
    ap.add_argument("--out", default="data/english_lexicon.tsv")
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=24)
    args = ap.parse_args()

    d = json.load(open(args.jmdict, encoding="utf-8"))
    print(f"words: {len(d['words'])}", flush=True)

    best: dict[str, tuple[bool, str]] = {}
    for w in d["words"]:
        kana = w.get("kana", [])
        if not kana:
            continue
        # 主たる読みがカタカナのものだけ（ひらがな語のカタカナ表記ゆれを除外）
        if not KATA.match(kana[0]["text"]):
            continue
        k = kana[0]["text"]
        if not (args.min_len <= len(k) <= 30):
            continue

        gloss = None
        proper = False
        for sense in w.get("sense", []):
            fields = set(sense.get("field", []) or [])
            misc = set(sense.get("misc", []) or [])
            if fields & PROPER_FIELDS or misc & PROPER_MISC:
                proper = True
            if gloss:
                continue
            for gl in sense.get("gloss", []):
                t = gl.get("text", "")
                if gl.get("lang") == "eng" and GLOSS_OK.match(t) and 2 <= len(t) <= args.max_len:
                    gloss = t
                    break
        if not gloss:
            continue

        if proper:
            gloss = gloss[0].upper() + gloss[1:]

        reading = kata2hira(k)
        is_common = any(x.get("common") for x in kana)
        prev = best.get(reading)
        # よく使う語を優先して採用
        if prev is None or (is_common and not prev[0]):
            best[reading] = (is_common, gloss)

    with open(args.out, "w", encoding="utf-8") as f:
        for reading in sorted(best):
            f.write(f"{reading}\t{best[reading][1]}\n")

    print(f"entries: {len(best)} -> {args.out}")
    for r in ["あんどろいど", "こんぴゅーたー", "かめら", "あぷり", "ぐーぐる",
              "すまーとふぉん", "てれび", "いんたーねっと", "でーた", "ろぐいん"]:
        print(f"  {r} -> {best.get(r, ('', None))[1]}")


if __name__ == "__main__":
    main()
