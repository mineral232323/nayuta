"""PC で Nayuta の変換を試す CLI。

Android IME（`LatticeConverter`）と同じ 2 段構え
（単語ラティス → ビーム探索 → BERT 再ランク）を PC で回して、
入力ごとの変換結果・候補・スコア・所要時間をその場で確認する。

    # 1 回だけ変換（読みはひらがな。カタカナも可）
    python -m train.cli convert にほんごのへんかんせいど
    python -m train.cli convert --text きょうはいいてんき --context "おはよう、"

    # 入力を打ち込みながら試す（モデルのロードは 1 回だけ）
    python -m train.cli repl

    # 評価セットで文正解率を測る（誤りの一覧も出す）
    python -m train.cli eval --data data/eval/wiki2000.jsonl --limit 200

アセット（辞書 / bigram / ONNX）は `data/` と `android/app/src/main/assets/` から
自動で探す。無い場合は `bash scripts/android/fetch_assets.sh` で配布 APK から展開する。

プリセット:

    wiki    bigram 生成 + BERT（delta=1.0, beta=0.5）… README の Wikipedia 57.10%
    sns     bigram 生成 + BERT（delta=2.0, beta=2.0）… 話し言葉 67.60%
    device  unigram 生成 + bigram 連接 + BERT（beta=1.2）
            … Android の LatticeConverter と同じ条件（端末のみの機能は含まない）

`python -m train.convert_cli` は後方互換のエイリアス。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.conv import Arc, Lexicon, kata2hira, normalize  # noqa: E402
from train.evaluate_bigram import BigramLM, beam_search_bigram  # noqa: E402
from train.evaluate_rerank import edit_distance  # noqa: E402
from train.evaluate_v2 import (  # noqa: E402
    MAX_CANDIDATES,
    OnnxScorer,
    build_lattice,
    gold_info,
    path_surface,
    rank_paths,
    score_spans,
)
from train.train import load_lexicon  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 評価ハーネス（train.evaluate_v2）と同じ解釈にする。
# gen: 変換仮説の作り方。device は Android と同じ unigram ビーム。
PRESETS: Dict[str, Dict[str, Any]] = {
    "wiki": {"gen": "bigram", "delta": 1.0, "beta": 0.5},
    "sns": {"gen": "bigram", "delta": 2.0, "beta": 2.0},
    "device": {"gen": "unigram", "delta": 0.0, "beta": 1.2},
}

LEXICON_CANDIDATES = [
    "data/lexicon_v3_frozen.tsv",
    "data/lexicon_v3.tsv",
    "data/lexicon_v2.tsv",
    "data/lexicon.tsv",
]
BIGRAM_CANDIDATES = ["data/bigram_v3.tsv", "data/bigram.tsv"]
BIGRAM_WORDS_CANDIDATES = ["data/bigram_words_v3.tsv", "data/bigram_words.tsv"]
ONNX_CANDIDATES = [
    "android/app/src/main/assets/model/scorer.onnx",
    "models/scorer_int8.onnx",
]
VOCAB_CANDIDATES = ["android/app/src/main/assets/model/vocab.txt"]

ASSET_MISSING_HELP = """\
辞書などのアセットが見つかりません。先に用意してください:

    bash scripts/android/fetch_assets.sh   # 配布 APK から展開（最短）
    # または README の「学習済みアセットを生成」の手順で生成

探した場所: {paths}
"""


# ---------------------------------------------------------------------------
# アセットの解決
# ---------------------------------------------------------------------------


def _pick(explicit: str, candidates: Sequence[str], what: str) -> str:
    """明示指定があればそれを、無ければ既定候補から最初に存在するものを返す。"""
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"{what}が見つかりません: {explicit}")
        return explicit
    for p in candidates:
        if os.path.exists(p):
            return p
    raise SystemExit(
        ASSET_MISSING_HELP.format(paths="\n".join(f"  - {p}" for p in candidates))
    )


def _pick_optional(explicit: str, candidates: Sequence[str]) -> str:
    if explicit:
        if not os.path.exists(explicit):
            raise SystemExit(f"ファイルが見つかりません: {explicit}")
        return explicit
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


# ---------------------------------------------------------------------------
# 変換エンジン
# ---------------------------------------------------------------------------


class Engine:
    """辞書・bigram・BERT をロードして、端末と同じ手順で変換する。"""

    def __init__(
        self,
        *,
        lexicon: str,
        bigram: str,
        bigram_words: str,
        onnx: str,
        vocab: str,
        gen: str,
        delta: float,
        beta: float,
        beam: int,
        want: int,
        threads: int,
        use_bert: bool = True,
    ):
        t0 = time.perf_counter()
        # ロード中の進捗（BigramLM など）は stderr に出して stdout を汚さない
        with contextlib.redirect_stdout(sys.stderr):
            self.lex: Lexicon = load_lexicon(lexicon, MAX_CANDIDATES)
            self.lm: Optional[BigramLM] = (
                BigramLM(bigram, bigram_words) if bigram else None
            )
            self.scorer: Optional[OnnxScorer] = None
            if use_bert:
                try:  # telemetry などの警告を黙らせる（結果には影響しない）
                    import onnxruntime as ort

                    ort.set_default_logger_severity(3)
                except Exception:
                    pass
                try:
                    self.scorer = OnnxScorer(onnx, vocab, threads=threads)
                except Exception as e:  # 端末と同じく、壊れていたら BERT 無しで続行
                    print(
                        f"注意: BERT モデルを読み込めませんでした（{e}）。"
                        "再ランクなしで動きます",
                        file=sys.stderr,
                    )
                    self.scorer = None
        self.load_ms = (time.perf_counter() - t0) * 1000.0

        self.lexicon_path = lexicon
        self.bigram_path = bigram
        self.bigram_words_path = bigram_words
        self.onnx_path = onnx if self.scorer is not None else ""
        self.vocab_path = vocab
        self.use_bert = self.scorer is not None
        # bigram が無ければ生成も unigram に落とす
        self.gen = gen if (gen == "unigram" or self.lm is not None) else "unigram"
        self.delta = delta
        self.beta = beta if self.use_bert else 0.0
        self.beam = beam
        self.want = want
        self.threads = threads
        self.last_beam_pred = ""
        self.last_timing = {
            "lattice_ms": 0.0,
            "beam_ms": 0.0,
            "bert_ms": 0.0,
            "rerank_ms": 0.0,
            "total_ms": 0.0,
        }

    # -- 変換 ------------------------------------------------------------

    def convert(
        self, text: str, ctx: str = "", top: int = 5
    ) -> List[Tuple[float, str, List[Tuple[str, float, float]]]]:
        """読みを変換し、[(コスト, 表記, [(スパン表記, 連接コスト, logP)])] を返す。"""
        read = kata2hira(normalize(text))
        if not read:
            return []
        t0 = time.perf_counter()
        n, arcs, _spans = build_lattice(self.lex, read)
        t1 = time.perf_counter()
        if self.gen == "bigram" and self.lm is not None:
            paths = beam_search_bigram(n, arcs, self.lm, self.beam, self.want, self.delta)
        else:
            from train.evaluate_rerank import beam_search as beam_search_unigram

            paths = beam_search_unigram(n, arcs, self.beam, self.want)
        t2 = time.perf_counter()

        logp = self._logp(read, ctx, paths)
        t3 = time.perf_counter()

        ranked = rank_paths(paths, self.lm, logp, self.beta)
        # BERT を使わない順位 = 探索 1 位（端末で BERT が無いときの出力と同じ）
        self.last_beam_pred = path_surface(paths[0]) if paths else ""
        t4 = time.perf_counter()

        out: List[Tuple[float, str, List[Tuple[str, float, float]]]] = []
        seen: set = set()
        for p in ranked:
            s = path_surface(p)
            if s in seen:
                continue
            seen.add(s)
            out.append(self._detail(p, logp))
            if len(out) >= top:
                break
        self.last_timing = {
            "lattice_ms": (t1 - t0) * 1000.0,
            "beam_ms": (t2 - t1) * 1000.0,
            "bert_ms": (t3 - t2) * 1000.0,
            "rerank_ms": (t4 - t3) * 1000.0,
            "total_ms": (t4 - t0) * 1000.0,
        }
        return out

    def _logp(
        self, read: str, ctx: str, paths: Sequence[Sequence[Arc]]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        if self.scorer is None:
            # BERT 無しでも rank_paths を通せるよう、全スパンを 0 として埋める
            out: Dict[Tuple[int, int], Dict[str, float]] = {}
            for p in paths:
                for a in p:
                    out.setdefault((a.start, a.end), {})[a.cand.surface] = 0.0
            return out
        return score_spans(self.scorer, ctx, read, paths)

    def _detail(
        self,
        path: Sequence[Arc],
        logp: Dict[Tuple[int, int], Dict[str, float]],
    ) -> Tuple[float, str, List[Tuple[str, float, float]]]:
        total = 0.0
        rows: List[Tuple[str, float, float]] = []
        prev: Optional[str] = None
        for a in path:
            conn = self.lm.conn(prev, a.cand.surface, 0.0) if self.lm else None
            base = a.cost if conn is None else conn
            lp = logp.get((a.start, a.end), {}).get(a.cand.surface, -6.0)
            total += base - self.beta * lp
            rows.append((a.cand.surface, base, lp))
            prev = a.cand.surface
        return total, path_surface(path), rows

    # -- 表示 ------------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"辞書    : {self.lexicon_path} ({len(self.lex.entries)} 読み)",
            f"bigram  : {self.bigram_path or '(なし)'}"
            + (f" / {self.bigram_words_path}" if self.bigram_path else ""),
            f"BERT    : {self.onnx_path or '(なし)'}",
            f"生成    : {self.gen}"
            + (f" (delta={self.delta})" if self.gen == "bigram" else "")
            + f" / beam={self.beam} want={self.want}",
            f"BERT重み: beta={self.beta}",
            f"ロード  : {self.load_ms:.0f}ms",
        ]
        return "\n".join(lines)


def _print_results(
    engine: Engine,
    text: str,
    ctx: str,
    results: Sequence[Tuple[float, str, List[Tuple[str, float, float]]]],
    verbose: bool,
) -> None:
    print(f"入力  : {text}")
    if ctx:
        print(f"文脈  : {ctx!r}")
    if not results:
        print("  （変換仮説なし）")
        return
    for i, (cost, surface, rows) in enumerate(results, 1):
        print(f"  {i:>2}位  {surface}  (cost={cost:.2f})")
        if verbose:
            for surf, base, lp in rows:
                print(f"          {surf:<12} 連接={base:7.2f}  logP={lp:8.4f}")
    t = engine.last_timing
    print(
        f"  時間  : ラティス {t['lattice_ms']:.1f} / 探索 {t['beam_ms']:.1f} / "
        f"BERT {t['bert_ms']:.1f} / 再ランク {t['rerank_ms']:.1f} / "
        f"計 {t['total_ms']:.1f} ms"
    )


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def cmd_convert(args: argparse.Namespace, engine: Engine) -> int:
    text = args.text_opt or args.text
    if not text:
        raise SystemExit("読みを指定してください: python -m train.cli convert にほんご")
    results = engine.convert(text, args.context, top=args.top)
    _print_results(engine, text, args.context, results, args.verbose)
    return 0


# ---------------------------------------------------------------------------
# repl
# ---------------------------------------------------------------------------


REPL_HELP = """\
変換したい読み（ひらがな / カタカナ）を入力して Enter。
コマンドは : か / で始める（日本語 IME の全角の ： や ／ でも可）。

  :ctx <text>    文脈（直前の確定済みテキスト）を設定。:ctx だけならクリア
  :preset <name> 変換条件のプリセット（wiki / sns / device）
  :beta [x]      BERT の重み
  :delta [x]     bigram 生成の連接コスト重み
  :gen [mode]    unigram|bigram
  :beam [n]      探索のビーム幅
  :want [n]      探索で残す仮説数（top の上限）
  :top [n]       表示する候補数
  :verbose [on|off]
  :status        現在の設定を表示
  :help / :quit
値を省略したコマンドは現在値を表示する。:top=3 の形でもよい。
設定のあとに読みを続けると、そのまま変換する（例: :top=3 にほんご）。
"""


def _split_command(rest: str) -> Tuple[str, str, str]:
    """`:top 3` / `:top=3` / `/top` を (コマンド, 値, 続きの読み) に分ける。

    設定のあとに変換したい読みを続けて書ける（例: `:top=3 にほんご`）。
    `:ctx` だけは後ろ全部を文脈として扱う。
    """
    rest = rest.strip()
    if not rest:
        return "", "", ""
    first, _, after = rest.partition(" ")
    after = after.strip()
    if "=" in first:
        key, _, val = first.partition("=")
        return key.strip().lower(), val.strip(), after
    key = first.lower()
    if key == "ctx":  # 文脈はスペース込みで後ろ全部
        return key, after, ""
    if not after:
        return key, "", ""
    val, _, trailing = after.partition(" ")
    return key, val.strip(), trailing.strip()


def cmd_repl(args: argparse.Namespace, engine: Engine) -> int:
    ctx = args.context
    top = args.top
    verbose = args.verbose
    preset = args.preset

    def settings() -> str:
        return (
            f"preset={preset} gen={engine.gen} delta={engine.delta} "
            f"beta={engine.beta} beam={engine.beam} want={engine.want} "
            f"top={top} verbose={verbose}"
        )

    print(engine.describe())
    print(REPL_HELP)
    while True:
        try:
            raw = input("かな> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        # 日本語 IME では : が ：、数字が全角になりがちなので NFKC で寄せる
        line = unicodedata.normalize("NFKC", raw).strip()
        if not line:
            continue
        if line[0] in ":/":
            cmd, val, trailing = _split_command(line[1:])
            if cmd in ("q", "quit", "exit"):
                return 0
            if cmd in ("h", "help", "?"):
                print(REPL_HELP)
                continue
            changed = True
            try:
                if cmd == "ctx":
                    ctx = val
                    print(f"文脈 = {ctx!r}")
                    changed = False
                elif cmd == "preset":
                    name = val.lower()
                    if not name:
                        print(f"preset={preset}")
                        changed = False
                    elif name not in PRESETS:
                        print(
                            "プリセットは "
                            f"{', '.join(sorted(PRESETS))} から選んでください"
                        )
                        changed = False
                    else:
                        preset = name
                        engine.gen = PRESETS[name]["gen"]
                        engine.delta = PRESETS[name]["delta"]
                        engine.beta = PRESETS[name]["beta"]
                elif cmd == "beta":
                    if val:
                        engine.beta = float(val)
                    else:
                        print(f"beta={engine.beta}")
                        changed = False
                elif cmd == "delta":
                    if val:
                        engine.delta = float(val)
                    else:
                        print(f"delta={engine.delta}")
                        changed = False
                elif cmd == "beam":
                    if val:
                        engine.beam = int(val)
                    else:
                        print(f"beam={engine.beam}")
                        changed = False
                elif cmd == "want":
                    if val:
                        engine.want = int(val)
                    else:
                        print(f"want={engine.want}")
                        changed = False
                elif cmd == "top":
                    if val:
                        top = int(val)
                    else:
                        print(f"top={top}")
                        changed = False
                elif cmd == "gen":
                    mode = val.lower()
                    if not mode:
                        print(f"gen={engine.gen}")
                        changed = False
                    elif mode not in ("unigram", "bigram"):
                        print("unigram か bigram を指定してください")
                        changed = False
                    else:
                        engine.gen = mode
                elif cmd == "verbose":
                    v = val.lower()
                    if v in ("on", "1", "true"):
                        verbose = True
                    elif v in ("off", "0", "false"):
                        verbose = False
                    elif v:  # 値として解釈できない → 読みとして変換する
                        trailing = val
                    else:
                        verbose = not verbose
                elif cmd == "status":
                    print(settings())
                    changed = False
                else:
                    print(
                        f"不明なコマンド: {cmd}"
                        "（:help 参照。全角の ： でも可）"
                    )
                    changed = False
                    trailing = ""
            except ValueError:
                print(f"値が不正です: {val!r}")
                changed = False
            if changed:
                print(settings())
            # 設定のあとに読みが続いていたら、そのまま変換する
            if trailing:
                results = engine.convert(trailing, ctx, top=top)
                _print_results(engine, trailing, ctx, results, verbose)
                print()
            continue

        results = engine.convert(line, ctx, top=top)
        _print_results(engine, line, ctx, results, verbose)
        print()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace, engine: Engine) -> int:
    if not os.path.exists(args.data):
        raise SystemExit(f"評価データが見つかりません: {args.data}")
    rows = _load_rows(args.data, args.limit)
    if not rows:
        raise SystemExit(f"評価データが空です: {args.data}")

    n_done = n_skip = n_bert = n_beam = 0
    head_n = head_ok = 0
    n_fixed = n_broken = 0
    edit_sum = gold_len = 0
    lat: List[float] = []
    errors: List[Dict[str, str]] = []
    fixed_examples: List[Dict[str, str]] = []
    broken_examples: List[Dict[str, str]] = []

    for rec in rows:
        read, gold, _bounds = gold_info(rec)
        if not read:
            n_skip += 1
            continue
        ctx = rec.get("ctx", "")
        results = engine.convert(read, ctx, top=1)
        if not results:
            n_skip += 1
            continue
        n_done += 1
        lat.append(engine.last_timing["total_ms"])
        pred = results[0][1]
        beam_pred = engine.last_beam_pred
        ok = pred == gold
        beam_ok = beam_pred == gold
        n_bert += int(ok)
        n_beam += int(beam_ok)
        if ok and not beam_ok:
            n_fixed += 1
            if len(fixed_examples) < 3:
                fixed_examples.append(
                    {"read": read, "gold": gold, "pred": pred, "beam": beam_pred}
                )
        elif beam_ok and not ok:
            n_broken += 1
            if len(broken_examples) < 3:
                broken_examples.append(
                    {"read": read, "gold": gold, "pred": pred, "beam": beam_pred}
                )
        words = rec.get("words") or []
        if words:
            head_gold = words[0][1]
            head_n += 1
            head_ok += int(pred.startswith(head_gold))
        ed = edit_distance(pred, gold)
        edit_sum += ed
        gold_len += len(gold)
        if not ok:
            item = {"read": read, "gold": gold, "pred": pred, "beam": beam_pred}
            if len(errors) < args.errors:
                errors.append(item)

    n = max(n_done, 1)
    acc = n_bert / n
    acc_beam = n_beam / n
    cer = edit_sum / max(gold_len, 1)
    lat_sorted = sorted(lat)

    def pct(q: float) -> float:
        if not lat_sorted:
            return 0.0
        i = min(len(lat_sorted) - 1, int(q * len(lat_sorted)))
        return lat_sorted[i]

    print()
    print(f"data    : {args.data}  limit={args.limit}  domain={args.domain or '-'}")
    cond = f"gen={engine.gen}"
    if engine.gen == "bigram":
        cond += f" delta={engine.delta}"
    cond += f" beta={engine.beta} beam={engine.beam} want={engine.want}"
    print(f"条件    : {cond}")
    assets = [os.path.basename(engine.lexicon_path)]
    if engine.bigram_path:
        assets.append(os.path.basename(engine.bigram_path))
    assets.append(os.path.basename(engine.onnx_path) or "(BERTなし)")
    print(f"アセット: {' + '.join(assets)}")
    print(f"評価    : {n_done} 文（skip {n_skip}）")
    print()
    print(f"文正解率                : {acc:.4f}  ({n_bert}/{n_done})")
    print(f"文正解率 (BERT なし)    : {acc_beam:.4f}  ({n_beam}/{n_done})  探索 1 位")
    print(f"CER (micro)             : {cer:.4f}")
    if head_n:
        print(f"文頭語正解率            : {head_ok / head_n:.4f}  ({head_ok}/{head_n})")
    print(f"BERT が直した / 壊した  : {n_fixed} / {n_broken}")
    print(
        f"レイテンシ              : p50={pct(0.50):.1f}ms "
        f"p95={pct(0.95):.1f}ms mean={sum(lat) / n:.1f}ms"
    )

    if fixed_examples:
        print("\nBERT が直した例")
        for e in fixed_examples:
            print(f"  読み: {e['read']}")
            print(f"  正解: {e['gold']}")
            print(f"  探索1位: {e['beam']}")

    if broken_examples:
        print("\nBERT が壊した例")
        for e in broken_examples:
            print(f"  読み: {e['read']}")
            print(f"  正解: {e['gold']}")
            print(f"  BERT後: {e['pred']}")

    if errors:
        print(f"\n誤り一覧（最初の {len(errors)} 件）")
        for i, e in enumerate(errors, 1):
            print(f"[{i}] 読み: {e['read']}")
            print(f"    正解: {e['gold']}")
            print(f"    予測: {e['pred']}")
            if e["beam"] != e["pred"]:
                print(f"    BERTなし: {e['beam']}")
    print()

    if args.json_out:
        payload = {
            "data": args.data,
            "limit": args.limit,
            "domain": args.domain,
            "config": {
                "gen": engine.gen,
                "delta": engine.delta,
                "beta": engine.beta,
                "beam": engine.beam,
                "want": engine.want,
            },
            "assets": {
                "lexicon": engine.lexicon_path,
                "bigram": engine.bigram_path,
                "bigram_words": engine.bigram_words_path,
                "onnx": engine.onnx_path,
                "vocab": engine.vocab_path,
            },
            "counts": {
                "evaluated": n_done,
                "skipped": n_skip,
                "bert_fixed": n_fixed,
                "bert_broken": n_broken,
            },
            "metrics": {
                "sent_acc": acc,
                "sent_acc_beam_first": acc_beam,
                "cer_micro": cer,
                "head_acc": (head_ok / head_n) if head_n else None,
            },
            "latency_ms": {
                "p50": pct(0.50),
                "p95": pct(0.95),
                "mean": sum(lat) / n,
            },
            "errors": errors,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {args.json_out}")
    return 0


def _load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        default="wiki",
        choices=sorted(PRESETS),
        help="変換条件のプリセット（既定: wiki）",
    )
    parser.add_argument("--gen", choices=["unigram", "bigram"], default=None,
                        help="仮説の作り方。既定はプリセット依存")
    parser.add_argument("--delta", type=float, default=None,
                        help="bigram 生成の連接コスト重み")
    parser.add_argument("--beta", type=float, default=None,
                        help="BERT スコアの重み")
    parser.add_argument("--beam", type=int, default=32, help="探索のビーム幅")
    parser.add_argument("--want", type=int, default=48, help="保持する仮説数")
    parser.add_argument("--lexicon", default="", help="辞書 tsv（省略時は自動）")
    parser.add_argument(
        "--bigram", default="", help="bigram tsv（省略時は自動。無効化は --no-bigram）"
    )
    parser.add_argument("--bigram-words", default="", help="bigram 語彙 tsv")
    parser.add_argument("--onnx", default="", help="BERT の ONNX（省略時は自動）")
    parser.add_argument("--vocab", default="", help="文字語彙 txt")
    parser.add_argument("--no-bert", action="store_true", help="BERT を使わない")
    parser.add_argument("--no-bigram", action="store_true", help="bigram を使わない")
    parser.add_argument("--threads", type=int, default=4,
                        help="ONNX Runtime のスレッド数")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m train.cli",
        description="PC で Nayuta の変換を試す CLI（変換 / 対話 / 評価）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="1 回だけ変換する")
    _common(c)
    c.add_argument("text", nargs="?", default="", help="読み（ひらがな / カタカナ）")
    c.add_argument("--text", dest="text_opt", default="", help="読み（--text でも可）")
    c.add_argument("--context", default="", help="文脈（直前の確定済みテキスト）")
    c.add_argument("--top", type=int, default=5, help="表示する候補数")
    c.add_argument("-v", "--verbose", action="store_true", help="文節ごとの内訳を出す")

    r = sub.add_parser("repl", help="対話的に変換を試す")
    _common(r)
    r.add_argument("--context", default="", help="初期の文脈")
    r.add_argument("--top", type=int, default=5, help="表示する候補数")
    r.add_argument("-v", "--verbose", action="store_true")

    e = sub.add_parser("eval", help="評価セットで文正解率を測る")
    _common(e)
    e.add_argument("--data", default="data/eval/wiki2000.jsonl",
                   help="評価用 jsonl（read / words / ctx）")
    e.add_argument("--limit", type=int, default=200, help="0 で全件")
    e.add_argument("--errors", type=int, default=10, help="誤りを何件表示するか")
    e.add_argument("--domain", default="", help="記録用のドメイン名")
    e.add_argument("--json-out", default="", help="結果 JSON の出力先")
    return p


def _resolve(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, str]]:
    preset = PRESETS[args.preset]
    cfg = {
        "gen": args.gen or preset["gen"],
        "delta": args.delta if args.delta is not None else preset["delta"],
        "beta": args.beta if args.beta is not None else preset["beta"],
        "beam": args.beam,
        "want": args.want,
    }
    paths = {
        "lexicon": _pick(args.lexicon, LEXICON_CANDIDATES, "辞書"),
        "bigram": "" if args.no_bigram else _pick_optional(args.bigram, BIGRAM_CANDIDATES),
        "bigram_words": "",
        "onnx": "",
        "vocab": "",
    }
    if paths["bigram"]:
        paths["bigram_words"] = _pick_optional(
            args.bigram_words, BIGRAM_WORDS_CANDIDATES
        )
        if not paths["bigram_words"]:
            raise SystemExit(
                "bigram の語彙ファイルが見つかりません: "
                + ", ".join(BIGRAM_WORDS_CANDIDATES)
            )
    if not args.no_bert:
        paths["onnx"] = _pick_optional(args.onnx, ONNX_CANDIDATES)
        if paths["onnx"]:
            paths["vocab"] = _pick_optional(args.vocab, VOCAB_CANDIDATES)
            if not paths["vocab"]:
                raise SystemExit(
                    "vocab.txt が見つかりません: " + ", ".join(VOCAB_CANDIDATES)
                )
        else:
            print(
                "注意: BERT モデルが見つからないため、再ランクなしで動きます"
                "（bash scripts/android/fetch_assets.sh で展開できます）",
                file=sys.stderr,
            )
    return cfg, paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 後方互換: `python -m train.convert_cli --text きょうは` を convert として扱う
    if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
        argv = ["convert"] + argv
    args = build_parser().parse_args(argv)
    cfg, paths = _resolve(args)
    engine = Engine(
        lexicon=paths["lexicon"],
        bigram=paths["bigram"],
        bigram_words=paths["bigram_words"],
        onnx=paths["onnx"],
        vocab=paths["vocab"],
        gen=cfg["gen"],
        delta=cfg["delta"],
        beta=cfg["beta"],
        beam=cfg["beam"],
        want=cfg["want"],
        threads=args.threads,
        use_bert=not args.no_bert and bool(paths["onnx"]),
    )
    if args.cmd == "convert":
        return cmd_convert(args, engine)
    if args.cmd == "repl":
        return cmd_repl(args, engine)
    return cmd_eval(args, engine)


if __name__ == "__main__":
    raise SystemExit(main())
