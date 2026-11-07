"""端末用 lexicon.tsv(.gz) から mmap 用 lexicon.ndict を作る（CI 用・標準ライブラリのみ）。

CI のリリースビルドは、前回リリースの APK から `assets/dict/lexicon.tsv.gz` を取り出す。
そのままでは旧形式（読み込みに 1.8s / 文字列で数十 MB）なので、ビルド時にこのスクリプトで
パック形式に変換し、旧 TSV は APK に入れない。

  python3 -m train.pack_dict_from_device \
      --in android/app/src/main/assets/dict/lexicon.tsv \
      --out android/app/src/main/assets/dict/lexicon.ndict

入力は端末辞書の 3 列（読み / 表記 / コスト*100）でも、学習側の 4 列
（読み / 表記 / 頻度 / コスト*100）でもよい。
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.export_dict import write_bin  # noqa: E402


def read_rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rd, surf = parts[0], parts[1]
            cost = int(parts[3]) if len(parts) >= 4 else int(parts[2])
            yield (rd, surf, cost)


def main() -> None:
    ap = argparse.ArgumentParser(description="端末用 TSV から ndict を作る")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(read_rows(args.inp))
    if not rows:
        raise SystemExit(f"空の辞書です: {args.inp}")
    # 端末 TSV は読み順に並んでいるが、念のため読みだけで安定ソート（候補順は保つ）
    rows.sort(key=lambda r: r[0].encode("utf-16-be"))
    write_bin(rows, args.out)


if __name__ == "__main__":
    main()
