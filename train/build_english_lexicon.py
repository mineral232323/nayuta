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

# 外来語らしくないもの（擬音語・擬態語など）を除くためのタグ
EXCLUDE_MISC = {"on-mim"}

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
    ap.add_argument("--abbr-ratio", type=float, default=1.8,
                    help="略語のとき、読みの長さに対する英語の長さの許容比")
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
        # 漢字表記があるものは日本語由来（魚名・植物名など）なので除外する。
        # 外来語はふつう漢字表記を持たない。
        if w.get("kanji"):
            continue
        k = kana[0]["text"]
        if not (args.min_len <= len(k) <= 30):
            continue

        senses = w.get("sense", [])
        # 名詞でないもの（擬音語・副詞など）は外来語ではないので除外する
        if not senses:
            continue
        first_misc = set(senses[0].get("misc", []) or [])
        first_pos = senses[0].get("partOfSpeech", []) or []
        if first_misc & EXCLUDE_MISC or "n" not in first_pos:
            continue

        # 略語（コラボ = collaboration など）は、読みに近い長さの英語を選ぶ。
        # 全綴りしか無い場合は英語化しない（コラボ → collaboration を防ぐ）。
        is_abbr = "abbr" in first_misc or "abbr" in (senses[0].get("misc") or [])
        all_glosses = [
            gl["text"]
            for sense in senses
            for gl in sense.get("gloss", [])
            if gl.get("lang") == "eng" and GLOSS_OK.match(gl.get("text", ""))
        ]

        gloss = None
        proper = False
        for sense in senses:
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

        if is_abbr:
            reading_len = max(1, len(k))
            best_gloss = None
            for g in all_glosses:
                r = len(g) / reading_len
                if 0.4 <= r <= args.abbr_ratio:
                    score = abs(r - 1.0)
                    if best_gloss is None or score < best_gloss[0]:
                        best_gloss = (score, g)
            if best_gloss is None:
                continue
            gloss = best_gloss[1]

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
