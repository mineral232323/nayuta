"""端末同梱用の辞書アセットを書き出す。

入力: data/lexicon_v2.tsv （読み \t 表記 \t 頻度 \t コスト*100）
出力: android/app/src/main/assets/dict/lexicon.tsv.gz

形式（TSV, gzip）:
  #NayutaDict  v2
  読み(ひらがな) \t 表記 \t コスト(int, = -log P * 100)
読みは端末側で二分探索するため UTF-16 コード単位順に昇順ソートしておく。
"""

from __future__ import annotations

import argparse
import gzip
import os
import struct
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MAGIC = b"NDX2"
VERSION = 1
HEADER = 64


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def write_bin(rows: List[Tuple[str, str, int]], path: str) -> None:
    """Dictionary.kt が mmap で読むパック済みバイナリ（little endian）を書く。

    レイアウト（すべて 4 バイト境界に整列）:
      0  magic "NDX2" / 4 version / 8 readingCount / 12 entryCount / 16 maxReadingLen
      20 readingCharsOff / 24 readingCharsLen（UTF-16 コード単位）
      28 surfaceCharsOff / 32 surfaceCharsLen
      36 readingOffsetsOff（rc+1 個の u32）/ 40 entryStartOff（rc 個の u32）
      44 entryCountOff（rc 個の u16）/ 48 surfaceOffsetsOff（ec+1 個の u32）
      52 costsOff（ec 個の i16 = -logP*100）
    """
    readings: List[str] = []
    entry_start: List[int] = []
    entry_count: List[int] = []
    surfaces: List[str] = []
    costs: List[int] = []
    prev = None
    for rd, surf, cost in rows:
        if rd != prev:
            readings.append(rd)
            entry_start.append(len(surfaces))
            entry_count.append(0)
            prev = rd
        entry_count[-1] += 1
        surfaces.append(surf)
        costs.append(cost)
    rc = len(readings)
    ec = len(surfaces)
    if not rc or not ec:
        raise SystemExit("辞書が空です")
    if any(c > 32767 or c < -32768 for c in costs):
        raise SystemExit("コストが i16 の範囲を超えています")
    if max(entry_count) > 65535:
        raise SystemExit("候補数が u16 の範囲を超えています")

    r_bytes = b"".join(r.encode("utf-16-le") for r in readings)
    s_bytes = b"".join(s.encode("utf-16-le") for s in surfaces)
    r_len = len(r_bytes) // 2
    s_len = len(s_bytes) // 2
    r_offsets: List[int] = []
    acc = 0
    for r in readings:
        r_offsets.append(acc)
        acc += len(r.encode("utf-16-le")) // 2
    r_offsets.append(acc)
    s_offsets: List[int] = []
    acc = 0
    for s in surfaces:
        s_offsets.append(acc)
        acc += len(s.encode("utf-16-le")) // 2
    s_offsets.append(acc)
    max_reading_len = max(len(r.encode("utf-16-le")) // 2 for r in readings)

    off = HEADER
    r_char_off = off
    off = _align(off + len(r_bytes), 4)
    s_char_off = off
    off = _align(off + len(s_bytes), 4)
    r_off_off = off
    off += 4 * (rc + 1)
    s_off_off = off
    off += 4 * (ec + 1)
    e_start_off = off
    off += 4 * rc
    e_count_off = off
    off += 2 * rc
    cost_off = off
    off += 2 * ec
    total = off

    header = struct.pack(
        "<4s13I", MAGIC, VERSION, rc, ec, max_reading_len,
        r_char_off, r_len, s_char_off, s_len,
        r_off_off, e_start_off, e_count_off, s_off_off, cost_off,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (HEADER - len(header)))
        f.write(r_bytes)
        f.write(b"\x00" * (s_char_off - (HEADER + len(r_bytes))))
        f.write(s_bytes)
        f.write(b"\x00" * (r_off_off - (s_char_off + len(s_bytes))))
        f.write(struct.pack(f"<{len(r_offsets)}I", *r_offsets))
        f.write(struct.pack(f"<{len(s_offsets)}I", *s_offsets))
        f.write(struct.pack(f"<{rc}I", *entry_start))
        f.write(struct.pack(f"<{rc}H", *entry_count))
        f.write(struct.pack(f"<{ec}h", *costs))
    print(
        f"bin: readings={rc} entries={ec} "
        f"chars(r/s)={r_len}/{s_len} size={os.path.getsize(path)/1e6:.2f}MB ({path})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default="data/lexicon_v2.tsv")
    ap.add_argument("--out", default="", help="旧形式 TSV.gz の出力先（空文字で無効）")
    ap.add_argument("--topk", type=int, default=8, help="読みあたりの候補数上限")
    ap.add_argument("--bin-out", default="android/app/src/main/assets/dict/lexicon.ndict",
                    help="パック済みバイナリ辞書の出力先（空文字で無効）")
    args = ap.parse_args()

    ents: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    with open(args.lexicon, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rd, surf = parts[0], parts[1]
            if len(parts) >= 4:
                cost = int(parts[3])
            else:
                cost = 0
            ents[rd].append((cost, surf))

    rows: List[Tuple[str, str, int]] = []
    for rd, lst in ents.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        for cost, surf in lst[: args.topk]:
            rows.append((rd, surf, cost))

    rows.sort(key=lambda r: (r[0].encode("utf-16-be"), r[1].encode("utf-16-be")))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        raw = 0
        with gzip.open(args.out, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write("#NayutaDict\tv2\t%d\n" % len(rows))
            for rd, surf, cost in rows:
                line = f"{rd}\t{surf}\t{cost}\n"
                raw += len(line.encode("utf-8"))
                f.write(line)
        size = os.path.getsize(args.out)
        print(f"raw={raw/1e6:.1f}MB -> gz={size/1e6:.2f}MB  ({args.out})")

    print(f"entries={len(rows)} readings={len(ents)}")

    if args.bin_out:
        write_bin(rows, args.bin_out)


if __name__ == "__main__":
    main()
