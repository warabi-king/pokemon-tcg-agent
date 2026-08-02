"""複数の世代(gen_XXX)をまるごと指定して総当たり戦をさせる。

`tools/run_matches_round_robin.py` は個々のagent(main.py)を1体ずつ `--agent` で
指定する必要があるが、自己対戦パイプラインの世代は1世代=16クラスタ(cl00〜cl15)の
self.pth+opp.pth+deck.csvの集まりなので、そのままでは使えない（main.pyが無い）。

このツールは指定した世代×クラスタを`package_agent.py`で自動的に実行可能な
main.py一式へ梱包し、`tools/run_matches_round_robin.py`のバックエンド実装
（legacy/batched/worker-batched/cuda-streams/cuda-ensemble/gpu-tree）を
関数として直接呼び出す（CLI越しではなく import して使う）。

CLIをそのまま呼ばずimportする理由: `--cross-gen-only`
（同一世代同士の対戦を除外し、異なる世代同士の組み合わせだけを対戦させる）を
実現するには対戦カード一覧（pairings）を外から差し込む必要があるが、
`run_matches_round_robin.py`のCLIは常に指定agent全体の総当たり
（combinations_with_replacement）を内部で組み立てるため、それを渡す口が無い。
一方バックエンド実装（run_tournament_parallel/run_batched_tournament/...）は
いずれも`pairings`を引数に取るので、ここではそれらを直接呼んで好きな
対戦カード一覧を渡す。

対戦後は agent単位の総合成績に加えて、世代単位に集約した勝率表も出す。

使用例:
    # gen_000 / gen_005 / gen_010 を全16クラスタで総当たり（既定: batched backend）
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen 5 --gen 10

    # 世代をまたぐ組み合わせだけ対戦させる（同一世代同士は対戦しない）。
    # gen_000とgen_010なら 16×16=256 対戦カードになる。
    python tools/pipeline/run_gen_tournament.py --gen 0 --gen 10 --cross-gen-only

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
import sys
import time

import config
from deck_utils import load_agents
from package_agent import package_agent

import run_matches_round_robin as rr  # noqa: E402


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


def build_pairings(
    names: list[str], name_to_gen: dict[str, str], no_self: bool, cross_gen_only: bool
) -> list[tuple[str, str]]:
    pairings = (
        list(itertools.combinations(names, 2))
        if no_self
        else list(itertools.combinations_with_replacement(names, 2))
    )
    if cross_gen_only:
        pairings = [
            (n0, n1) for n0, n1 in pairings if name_to_gen[n0] != name_to_gen[n1]
        ]
    return pairings


def run_tournament(
    specs: list[rr.AgentSpec],
    pairings: list[tuple[str, str]],
    args: argparse.Namespace,
) -> dict[tuple[str, str], "rr.HeadToHead"]:
    """`run_matches_round_robin.py`のbackend実装をpairings指定で直接呼ぶ。"""
    total_games = len(pairings) * args.games
    lanes = total_games if args.lanes == 0 else min(args.lanes, total_games)
    workers = 0
    if args.backend == "worker-batched" and args.workers == 0:
        workers = lanes
    elif args.backend in ("legacy", "worker-batched"):
        workers = rr.resolve_worker_count(args.workers, total_games)

    if args.backend == "worker-batched":
        from batched_tournament import run_worker_batched_tournament

        out = run_worker_batched_tournament(
            specs=specs, pairings=pairings, num_games=args.games,
            alternate_sides=not args.no_alternate, device_name=args.device,
            batch_size=args.batch_size, lanes=lanes, search_count=args.search_count,
            seed=args.seed, cpu_workers=workers,
        )
        h2h_map, _ = rr.aggregate_tournament_results(
            pairings, out.results, num_games=args.games, verbose=not args.quiet
        )
    elif args.backend == "gpu-tree":
        from gpu_tree_tournament import run_gpu_tree_tournament

        out = run_gpu_tree_tournament(
            specs=specs, pairings=pairings, num_games=args.games,
            alternate_sides=not args.no_alternate, device_name=args.device,
            batch_size=args.batch_size, lanes=lanes, search_count=args.search_count,
            seed=args.seed,
        )
        h2h_map, _ = rr.aggregate_tournament_results(
            pairings, out.results, num_games=args.games, verbose=not args.quiet
        )
    elif args.backend in ("batched", "cuda-streams", "cuda-ensemble"):
        from batched_tournament import run_batched_tournament

        out = run_batched_tournament(
            specs=specs, pairings=pairings, num_games=args.games,
            alternate_sides=not args.no_alternate, device_name=args.device,
            batch_size=args.batch_size, lanes=lanes, search_count=args.search_count,
            seed=args.seed,
            parallel_cuda_models=args.backend == "cuda-streams",
            cuda_ensemble_models=args.backend == "cuda-ensemble",
        )
        h2h_map, _ = rr.aggregate_tournament_results(
            pairings, out.results, num_games=args.games, verbose=not args.quiet
        )
    elif workers == 1:
        loaded = {spec.name: rr.load_agent(spec, f"gen_tournament_agent_{i}") for i, spec in enumerate(specs)}
        h2h_map = {}
        for name0, name1 in pairings:
            h2h, _ = rr.run_pairing(
                loaded[name0], loaded[name1], num_games=args.games,
                alternate_sides=not args.no_alternate, debug=False, verbose=not args.quiet,
            )
            h2h_map[(name0, name1)] = h2h
    else:
        h2h_map, _ = rr.run_tournament_parallel(
            specs=specs, pairings=pairings, num_games=args.games, workers=workers,
            alternate_sides=not args.no_alternate, debug=False, verbose=not args.quiet,
        )
    return h2h_map


def print_overall(names: list[str], pairings: list[tuple[str, str]], h2h_map) -> dict[str, "rr.OverallRecord"]:
    overall = {name: rr.OverallRecord(name=name) for name in names}
    for name0, name1 in pairings:
        h2h = h2h_map[(name0, name1)]
        if h2h.is_self_match:
            overall[name0].wins += h2h.name0_wins
            overall[name0].losses += h2h.name1_wins
            overall[name0].draws += h2h.draws
            overall[name0].unresolved += h2h.unresolved
            continue
        overall[name0].wins += h2h.name0_wins
        overall[name0].losses += h2h.name1_wins
        overall[name0].draws += h2h.draws
        overall[name0].unresolved += h2h.unresolved
        overall[name1].wins += h2h.name1_wins
        overall[name1].losses += h2h.name0_wins
        overall[name1].draws += h2h.draws
        overall[name1].unresolved += h2h.unresolved

    print("\n=== 対戦カード別 勝ち数（行 vs 列） ===")
    rr.print_head_to_head_table(names, h2h_map)
    print("  ※対角成分は自己対戦。対戦していない組み合わせは n/a")

    print("\n=== agent単位 総合成績（勝率順） ===")
    ranking = sorted(overall.values(), key=lambda r: r.win_rate, reverse=True)
    header = f"{'順位':<4}{'エージェント':<20}{'試合数':>6}{'勝':>5}{'負':>5}{'分':>5}{'勝率':>8}"
    print(header)
    for rank, record in enumerate(ranking, start=1):
        print(
            f"{rank:<4}{record.name:<20}{record.games:>6}{record.wins:>5}"
            f"{record.losses:>5}{record.draws:>5}{record.win_rate:>7.1f}%"
        )
    return overall


def summarize_by_generation(
    pairings: list[tuple[str, str]],
    h2h_map,
    name_to_gen: dict[str, str],
) -> None:
    """agent単位の head_to_head を gen単位に集約して表示する。"""
    gen_overall: dict[str, dict[str, int]] = defaultdict(
        lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
    )
    # (genA, genB) genA<genB のキーで集約（gen同士のクロス集計。自世代内の対戦は含めない）。
    gen_h2h: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"a_wins": 0, "b_wins": 0, "draws": 0, "games": 0}
    )

    for name0, name1 in pairings:
        h2h = h2h_map[(name0, name1)]
        wins0, wins1, draws, total = h2h.name0_wins, h2h.name1_wins, h2h.draws, h2h.total
        g0, g1 = name_to_gen[name0], name_to_gen[name1]

        if h2h.is_self_match:
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


def save_json(
    out_path: Path,
    specs: list[tuple[str, Path, Path]],
    pairings: list[tuple[str, str]],
    h2h_map,
    overall: dict[str, "rr.OverallRecord"],
    args: argparse.Namespace,
    elapsed: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "agents": [
                    {"name": name, "agent_path": str(main_py), "deck_path": str(deck_csv)}
                    for name, main_py, deck_csv in specs
                ],
                "games_per_matchup": args.games,
                "cross_gen_only": args.cross_gen_only,
                "include_self_matches": not args.no_self,
                "backend": args.backend,
                "overall": {
                    name: {
                        "games": rec.games, "wins": rec.wins, "losses": rec.losses,
                        "draws": rec.draws, "unresolved": rec.unresolved,
                        "win_rate": rec.win_rate,
                    }
                    for name, rec in overall.items()
                },
                "head_to_head": {
                    f"{n0}_vs_{n1}": {
                        "is_self_match": h2h_map[(n0, n1)].is_self_match,
                        "name0_wins": h2h_map[(n0, n1)].name0_wins,
                        "name1_wins": h2h_map[(n0, n1)].name1_wins,
                        "draws": h2h_map[(n0, n1)].draws,
                        "unresolved": h2h_map[(n0, n1)].unresolved,
                        "total": h2h_map[(n0, n1)].total,
                    }
                    for n0, n1 in pairings
                },
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n詳細JSON: {out_path}")


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
        help="対戦の実行backend（既定: batched）",
    )
    parser.add_argument("--device", default="auto", help="batched系backendのdevice（既定: auto）")
    parser.add_argument("--lanes", type=int, default=0, help="batched系backendの同時試合数（既定0=全試合）")
    parser.add_argument("--batch-size", type=int, default=128, help="batched系backendのNN batch size")
    parser.add_argument("--workers", type=int, default=0, help="legacy/worker-batchedの並列数")
    parser.add_argument("--seed", type=int, default=0, help="batched系backendの乱数seed")
    parser.add_argument("--no-self", action="store_true", help="自己対戦（同一世代・同一クラスタ）を除外する")
    parser.add_argument(
        "--cross-gen-only", action="store_true",
        help="同一世代同士の対戦（クラスタ違い含む）を除外し、異なる世代の組み合わせだけ対戦させる。"
             "例: --gen 0 --gen 10 --cross-gen-only なら 16x16=256 対戦カードになる。",
    )
    parser.add_argument("--no-alternate", action="store_true", help="先手/後手を固定する")
    parser.add_argument("--quiet", action="store_true", help="試合ごとの結果表示を省略する")
    parser.add_argument("--no-save-json", action="store_true", help="詳細JSONをresults/へ保存しない")
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
    packaged, name_to_gen = package_generations(args.gens, clusters, root, workdir, args.search_count)
    if len(packaged) < 2:
        raise SystemExit("梱包できたagentが2体未満です（世代/クラスタの指定を確認してください）。")
    for name, main_py, deck_csv in packaged:
        print(f"  {name}: {main_py} (deck: {deck_csv})")

    names = [name for name, _, _ in packaged]
    pairings = build_pairings(names, name_to_gen, args.no_self, args.cross_gen_only)
    if not pairings:
        raise SystemExit(
            "対戦カードが0件です。--cross-gen-onlyを指定した場合は--genを2つ以上（異なる世代）指定してください。"
        )

    specs = [rr.AgentSpec(name=n, agent_path=p.resolve(), deck_path=d.resolve()) for n, p, d in packaged]

    n_gens = len(set(name_to_gen.values()))
    print(
        f"\n=== 総当たり戦開始 (世代数={n_gens}, 対戦カード数={len(pairings)}, "
        f"カードあたり{args.games}試合, cross_gen_only={args.cross_gen_only}, "
        f"backend={args.backend}, device={args.device}) ===\n"
    )

    started = time.time()
    h2h_map = run_tournament(specs, pairings, args)
    elapsed = time.time() - started

    overall = print_overall(names, pairings, h2h_map)
    print(f"\n所要時間: {elapsed:.1f}秒")

    summarize_by_generation(pairings, h2h_map, name_to_gen)

    if not args.no_save_json:
        out_path = config.REPO_ROOT / "results" / f"gen_tournament_{int(time.time())}.json"
        save_json(out_path, packaged, pairings, h2h_map, overall, args, elapsed)


if __name__ == "__main__":
    main()
