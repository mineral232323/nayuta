"""辞書の規模（読み数・行数）と、ロード時間・端末サイズの関係を測る実験。

    python -m train.experiment_lexicon_scale --lexicon data/lexicon_v3_frozen.tsv
    python -m train.experiment_lexicon_scale --lexicon logs/lex_old_1m.tsv --gzip-test

出力:
  * 行数 / 読み数 / コーパス頻度付き行数
  * Python での辞書ロード時間（train.load_lexicon）
  * 端末ロード時間の推定（README 実測の「読み数にほぼ比例」モデル）
  * --gzip-test で端末辞書（3 列 TSV の gzip）のサイズ見積もり

端末ロードの推定式: 実測 14.5 万読み = 518ms / 50 万読み = 1.8s から
1 読みあたり約 3.6µs とみなす（実機の実測ではない）。
"""

from __future__ import annotations

import argparse
import gzip
import os
import resource
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.train import load_lexicon  # noqa: E402

DEVICE_SEC_PER_READING = 3.6e-6  # 500k 読み ≒ 1.8s の実測から


def scan(path: str) -> Tuple[int, int, int, int, int]:
    rows = attested = 0
    readings = set()
    attested_surfaces = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            rd, surf, cnt, _cost = line.rstrip("\n").split("\t")
            rows += 1
            readings.add(rd)
            if int(cnt) > 0:
                attested += 1
                attested_surfaces.add(surf)
    return rows, len(readings), attested, len(attested_surfaces), 0


def gzip_size(path: str, tmp: str = "/tmp/_lex_size.gz") -> int:
    n = 0
    with open(path, encoding="utf-8") as f, gzip.open(tmp, "wb", compresslevel=9) as g:
        for line in f:
            rd, surf, _cnt, cost = line.rstrip("\n").split("\t")
            n += g.write(f"{rd}\t{surf}\t{cost}\n".encode("utf-8"))
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def main() -> None:
    ap = argparse.ArgumentParser(description="辞書規模とロード時間の実験")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--gzip-test", action="store_true",
                    help="端末辞書（3列TSVのgzip）サイズも測る")
    args = ap.parse_args()

    size_mb = os.path.getsize(args.lexicon) / 1e6
    t0 = time.perf_counter()
    rows, readings, attested, attested_surf, _ = scan(args.lexicon)
    scan_s = time.perf_counter() - t0
    print(f"lexicon : {args.lexicon}")
    print(f"  file      : {size_mb:.2f} MB")
    print(f"  rows      : {rows:,}")
    print(f"  readings  : {readings:,}")
    print(f"  attested  : {attested:,} 行 / {attested_surf:,} 表記")
    print(f"  scan      : {scan_s:.1f}s")
    est = readings * DEVICE_SEC_PER_READING
    print(f"  端末ロード推定: {est:.2f}s（読み数×3.6µs）")

    t0 = time.perf_counter()
    lex = load_lexicon(args.lexicon, args.topk)
    load_s = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"  Python load: {load_s:.2f}s  maxrss={rss:.2f}GB  entries={len(lex.entries):,}")

    if args.gzip_test:
        t0 = time.perf_counter()
        gz = gzip_size(args.lexicon)
        print(f"  端末辞書 gzip(9): {gz / 1e6:.2f} MB（{time.perf_counter() - t0:.1f}s）")


if __name__ == "__main__":
    main()
