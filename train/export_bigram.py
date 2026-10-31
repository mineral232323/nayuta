"""bigram 連接コストを端末用のバイナリに書き出す。

  python -m train.export_bigram

形式（リトルエンディアン）:
  int32 magic 'NBGY' / int32 version / int32 V / int32 P
  int32   starts[V+1]   （prev id -> rights/costs の範囲）
  int32   rights[P]
  int16   costs[P]      （-log P(cur|prev) * 100）
  残り: 表記 V 行（UTF-8, \\n 区切り）
"""

from __future__ import annotations

import argparse
import os
import struct
from array import array

MAGIC = 0x5947424E  # "NBGY" little-endian


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigram", default="data/bigram.tsv")
    ap.add_argument("--words", default="data/bigram_words.tsv")
    ap.add_argument("--out", default="android/app/src/main/assets/dict/bigram.bin")
    args = ap.parse_args()

    words: list[str] = []
    with open(args.words, encoding="utf-8") as f:
        for line in f:
            i, w, _c = line.rstrip("\n").split("\t")
            assert int(i) == len(words)
            words.append(w)
    V = len(words)
    print(f"words={V}")

    pairs: dict[int, list[tuple[int, int]]] = {}
    with open(args.bigram, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            prev, cur, cost = int(p[0]), int(p[1]), int(p[2])
            pairs.setdefault(prev, []).append((cur, cost))

    starts = array("i", [0]) * (V + 1)
    rights = array("i")
    costs = array("h")
    for i in range(V):
        starts[i] = len(rights)
        lst = pairs.get(i)
        if lst:
            lst.sort(key=lambda x: x[0])
            for cur, c in lst:
                rights.append(cur)
                costs.append(max(-32768, min(32767, c)))
    starts[V] = len(rights)
    P = len(rights)
    print(f"pairs={P}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<iiii", MAGIC, 1, V, P))
        f.write(starts.tobytes())
        f.write(rights.tobytes())
        f.write(costs.tobytes())
        f.write(("\n".join(words) + "\n").encode("utf-8"))

    size = os.path.getsize(args.out)
    print(f"-> {args.out} ({size/1e6:.2f}MB)")


if __name__ == "__main__":
    main()
