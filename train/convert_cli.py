"""後方互換エイリアス。実体は `train/cli.py` に移した。

  python -m train.convert_cli --text きょうは
  python -m train.convert_cli convert きょうは

新しく使うならこちら:
  python -m train.cli convert きょうは
  python -m train.cli repl
  python -m train.cli eval --limit 200
"""

from __future__ import annotations

import sys

from train.cli import main

if __name__ == "__main__":
    sys.exit(main())
