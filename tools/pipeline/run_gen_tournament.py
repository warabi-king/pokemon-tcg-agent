"""複数の世代(gen_XXX)をまるごと指定して総当たり戦をさせる。

`tools/run_matches_round_robin.py` は個々のagent(main.py)を1体ずつ `--agent` で
指定する必要があるが、自己対戦パイプラインの世代は1世代=16クラスタ(cl00〜cl15)の
self.pth+opp.pth+deck.csvの集まりなので、そのままでは使えない（main.pyが無い）。

このツールは指定した世代×クラスタを`package_agent.py`で自動的に実行可能な
main.py一式へ梱包し、`tools/run_matches_round_robin.py`へ橋渡しする。
バックエンド（legacy/batched/worker-batched/cuda-streams/cuda-ensemble/gpu-tree）は
そのまま透過的に指定できる（既定はbatched, device=auto）。

対戦後は agent単位の総合成績に加えて、世代単位に集約した勝率表も出す。

使用例:
    # gen_000 / gen_005 / gen_010 を全16クラスタで総当たり（既定: batched backend）
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen 5 --gen 10

    # gen_XXX の命名から外れたディレクトリ（手動コピー等）もそのまま指定できる
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen gen_005-copy --gen 10

    # クラスタを絞って手早く確認
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen 5 --gen 10 \\
        --cluster cl00 --cluster cl05 --games 10

    # legacyバックエンド・CPUで少数試合だけ試す
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen 5 \\
        --backend legacy --games 4

`--gen`には以下のいずれかを指定できる:
    - 数値（例: `5`）  … `<root>/gen_005` を指す
    - `gen_XXX`以外のディレクトリ名（例: `gen_005-copy`） … `<root>/<name>` を指す
    - 絶対/相対パス（例: `/tmp/backup/gen_005`） … そのパスを直接指す

単体実行: `python tools/pipeline/run_gen_tournament.py -h`
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import config
from deck_utils import load_agents
from package_agent import package_agent


def resolve_gen_dir(spec: str, root: Path) -> Path:
    """--gen に渡された文字列を実ディレクトリへ解決する。

    数値なら `<root>/gen_{spec:03d}`、それ以外はディレクトリ名またはパスとして扱う
    （相対パスはまず`<root>`基準で探し、無ければカレントディレクトリ基準にする）。
    """
    if spec.isdigit():
        return root / f"gen_{int(spec):03d}"
    p = Path(spec)
    if p.is_absolute():
        return p
    candidate = root / spec
    if candidate.exists():
        return candidate
    return p


def gen_label(gen_dir: Path) -> str:
    """ディレクトリ名から、agent名のプレフィックスに使える識別子を作る（英数字のみ）。"""
    return re.sub(r"[^0-9A-Za-z]", "", gen_dir.name) or gen_dir.name


def package_generations(
    gen_specs: list[str],
    clusters: list[str],
    root: Path,
    workdir: Path,
    search_count: int,
) -> tuple[list[tuple[str, Path, Path]], dict[str, str]]:
    """指定した世代×クラスタを梱包し、(name, main.py, deck.csv) のリストと
    agent名->世代ラベルの対応表を返す。

    欠けている（self.pth/opp.pth/deck.csvが揃っていない）組み合わせは
    警告を出してスキップする。
    """
    workdir.mkdir(parents=True, exist_ok=True)

    resolved: list[tuple[Path, str]] = []
    seen_labels: dict[str, Path] = {}
    for spec in gen_specs:
        gen_dir = resolve_gen_dir(spec, root)
        if not gen_dir.exists():
            print(f"[skip] --gen {spec}: {gen_dir} が存在しません。", file=sys.stderr)
            continue
        label = gen_label(gen_dir)
        if label in seen_labels and seen_labels[label] != gen_dir:
            raise SystemExit(
                f"--gen {spec} の識別子 '{label}' が {seen_labels[label]} と衝突します。"
                "ディレクトリ名を変えてください。"
            )
        seen_labels[label] = gen_dir
        resolved.append((gen_dir, label))

    specs: list[tuple[str, Path, Path]] = []
    name_to_gen: dict[str, str] = {}
    for gen_dir, label in resolved:
        for cluster in clusters:
            agdir = gen_dir / "agents" / cluster
            self_pth, opp_pth, deck_csv = agdir / "self.pth", agdir / "opp.pth", agdir / "deck.csv"
            if not (self_pth.exists() and opp_pth.exists() and deck_csv.exists()):
                print(f"[skip] {agdir}: self.pth/opp.pth/deck.csvが揃っていません。", file=sys.stderr)
                continue
            name = f"{label}_{cluster}"
            out_src = workdir / name
            package_agent(self_pth, opp_pth, deck_csv, out_src, search_count=search_count)
            specs.append((name, out_src / "main.py", out_src / "deck.csv"))
            name_to_gen[name] = label
    return specs, name_to_gen


def build_pairings(names: list[str], no_self: bool) -> list[tuple[str, str]]:
    if no_self:
        return list(itertools.combinations(names, 2))
    return list(itertools.combinations_with_replacement(names, 2))


def summarize_by_generation(
    result_json: dict,
    pairings: list[tuple[str, str]],
    name_to_gen: dict[str, str],
) -> None:
    """agent単位の head_to_head を gen単位に集約して表示する。"""
    h2h_by_key = result_json["head_to_head"]

    gen_overall: dict[str, dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    # (genA, genB) genA<genB のキーで集約（gen同士のクロス集計。自世代内の対戦は含めない）。
    gen_h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"a_wins": 0, "b_wins": 0, "draws": 0, "games": 0}
    )

    for name0, name1 in pairings:
        h2h = h2h_by_key[f"{name0}_vs_{name1}"]
        wins0, wins1, draws, total = (
            h2h["name0_wins"], h2h["name1_wins"], h2h["draws"], h2h["total"]
        )
        g0, g1 = name_to_gen[name0], name_to_gen[name1]

        if h2h["is_self_match"]:
            gen_overall[g0]["wins"] += wins0
            gen_overall[g0]["losses"] += wins1
            gen_overall[g0]["draws"] += draws
            gen_overall[g0]["games"] += total
            continue

        gen_overall[g0]["wins"] += wins0
        gen_overall[g0]["losses"] += wins1
        gen_overall[g0]["draws"] += draws
        gen_overall[g0]["games"] += total

        gen_overall[g1]["wins"] += wins1
        gen_overall[g1]["losses"] += wins0
        gen_overall[g1]["draws"] += draws
        gen_overall[g1]["games"] += total

        if g0 == g1:
            continue
        key = tuple(sorted((g0, g1)))
        rec = gen_h2h[key]
        if key[0] == g0:
            rec["a_wins"] += wins0
            rec["b_wins"] += wins1
        else:
            rec["a_wins"] += wins1
            rec["b_wins"] += wins0
        rec["draws"] += draws
        rec["games"] += total

    print("\n=== 世代単位の総合成績（勝率順） ===")
    ranking = sorted(
        gen_overall.items(),
        key=lambda kv: (kv[1]["wins"] / kv[1]["games"] if kv[1]["games"] else 0.0),
        reverse=True,
    )
    header = f"{'順位':<4}{'世代':<12}{'試合数':>6}{'勝':>6}{'負':>6}{'分':>5}{'勝率':>8}"
    print(header)
    for rank, (gen, rec) in enumerate(ranking, start=1):
        win_rate = (rec["wins"] / rec["games"] * 100) if rec["games"] else 0.0
        print(
            f"{rank:<4}{gen:<12}{rec['games']:>6}{rec['wins']:>6}"
            f"{rec['losses']:>6}{rec['draws']:>5}{win_rate:>7.1f}%"
        )

    if gen_h2h:
        print("\n=== 世代同士の対戦成績（全クラスタ合算） ===")
        for (a, b), rec in sorted(gen_h2h.items()):
            games = rec["games"]
            a_rate = (rec["a_wins"] / games * 100) if games else 0.0
            b_rate = (rec["b_wins"] / games * 100) if games else 0.0
            print(
                f"  {a} vs {b}: {a} {rec['a_wins']}勝({a_rate:.1f}%) / "
                f"{b} {rec['b_wins']}勝({b_rate:.1f}%) / 引き分け {rec['draws']} "
                f"(全{games}試合)"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", type=str, action="append", required=True, dest="gens",
                         help="対戦させる世代。複数回指定できる（例: --gen 0 --gen 5 --gen 10）。"
                              "数値以外（例: gen_005-copy や任意のパス）も指定可能。")
    parser.add_argument("--cluster", type=str, action="append", dest="clusters", default=None,
                         help="対象クラスタ名を絞り込む（省略時はclusters.jsonの全クラスタ）。複数回指定可。")
    parser.add_argument("--root", type=Path, default=config.ROOT, help="pipeline成果物のルート（既定: PIPE_ROOT）")
    parser.add_argument("--workdir", type=Path, default=None,
                         help="梱包済みagentの出力先（既定: results/tournament_cache/）")
    parser.add_argument("--games", type=int, default=5, help="対戦カードあたりの試合数（既定5）")
    parser.add_argument("--search-count", type=int, default=config.LEAGUE_SEARCH_COUNT,
                         help="MCTS探索回数。梱包main.py(legacy用)とbatched系backend両方に使う"
                              f"（既定: {config.LEAGUE_SEARCH_COUNT} = PIPE_LEAGUE_SEARCH_COUNT）")
    parser.add_argument(
        "--backend",
        choices=("legacy", "batched", "worker-batched", "cuda-streams", "cuda-ensemble", "gpu-tree"),
        default="batched",
        help="tools/run_matches_round_robin.pyへそのまま渡すbackend（既定: batched）",
    )
    parser.add_argument("--device", default="auto", help="batched系backendのdevice（既定: auto）")
    parser.add_argument("--lanes", type=int, default=0, help="batched系backendの同時試合数（既定0=全試合）")
    parser.add_argument("--batch-size", type=int, default=None, help="batched系backendのNN batch size")
    parser.add_argument("--workers", type=int, default=0, help="legacy/worker-batchedの並列数")
    parser.add_argument("--no-self", action="store_true", help="自己対戦（同一世代・同一クラスタ）を除外する")
    parser.add_argument("--no-alternate", action="store_true", help="先手/後手を固定する")
    parser.add_argument("--quiet", action="store_true", help="試合ごとの結果表示を省略する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root
    workdir = args.workdir or (config.REPO_ROOT / "results" / "tournament_cache")

    clusters_json = root / "clusters.json"
    all_clusters = [a["name"] for a in load_agents(clusters_json)]
    clusters = args.clusters or all_clusters
    unknown = sorted(set(clusters) - set(all_clusters))
    if unknown:
        raise SystemExit(f"clusters.jsonに無いクラスタが指定されました: {unknown}")

    print(f"=== 梱包: 世代指定={args.gens} クラスタ数={len(clusters)} ===")
    specs, name_to_gen = package_generations(args.gens, clusters, root, workdir, args.search_count)
    if len(specs) < 2:
        raise SystemExit("梱包できたagentが2体未満です（世代/クラスタの指定を確認してください）。")
    for name, main_py, deck_csv in specs:
        print(f"  {name}: {main_py} (deck: {deck_csv})")

    names = [name for name, _, _ in specs]
    pairings = build_pairings(names, args.no_self)

    round_robin_script = config.REPO_ROOT / "tools" / "run_matches_round_robin.py"
    argv = [sys.executable, str(round_robin_script)]
    for name, main_py, deck_csv in specs:
        argv += ["--agent", f"{name}={main_py}:{deck_csv}"]
    argv += [
        "--games", str(args.games),
        "--backend", args.backend,
        "--device", args.device,
        "--lanes", str(args.lanes),
        "--search-count", str(args.search_count),
        "--workers", str(args.workers),
        "--save-json",
    ]
    if args.batch_size is not None:
        argv += ["--batch-size", str(args.batch_size)]
    if args.no_self:
        argv.append("--no-self")
    if args.no_alternate:
        argv.append("--no-alternate")
    if args.quiet:
        argv.append("--quiet")

    started = time.time()
    subprocess.run(argv, cwd=config.REPO_ROOT, check=True)

    results_root = config.REPO_ROOT / "results"
    candidates = [
        p for p in results_root.glob("tournament_*.json") if p.stat().st_mtime >= started - 1
    ]
    if not candidates:
        print("[warn] 詳細JSONが見つからず、世代単位の集約表示をスキップします。", file=sys.stderr)
        return
    result_path = max(candidates, key=lambda p: p.stat().st_mtime)
    result_json = json.loads(result_path.read_text(encoding="utf-8"))
    summarize_by_generation(result_json, pairings, name_to_gen)
    print(f"\n詳細JSON: {result_path}")


if __name__ == "__main__":
    main()
