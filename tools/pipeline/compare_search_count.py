"""同じ世代・同じクラスタのモデルを、MCTS探索回数だけ変えて対戦させる。

1世代(gen_XXX)は self.pth/opp.pth/deck.csv の組を持つが、search_countは
`package_agent.py`が梱包時にmain.pyへ焼き込む値なので、「重みは同じまま
探索回数だけ変えて比較したい」場合は同じself/opp/deckを異なるsearch_countで
2回梱包し直す必要がある。このツールはそれを自動化する。

`tools/run_matches_round_robin.py` のload_agent/run_pairingをそのまま使うため、
対戦結果の判定・先手/後手の入れ替えなど既存ツールと同じ挙動になる。

使用例:
    # gen_000 の cl00 で 探索10 vs 探索1000 を5試合
    python tools/pipeline/compare_search_count.py --gen 0 --cluster cl00 \\
        --search-count-a 10 --search-count-b 1000 --games 5

    # 先手/後手を固定して1試合だけデバッグ出力付きで確認
    python tools/pipeline/compare_search_count.py --gen 0 --cluster cl00 \\
        --games 1 --no-alternate --debug

単体実行: python tools/pipeline/compare_search_count.py -h
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import config
from package_agent import package_agent
from run_gen_tournament import resolve_gen_dir

import run_matches_round_robin as rr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", type=str, default="0",
                         help="世代指定。数値なら<root>/gen_XXX、それ以外はディレクトリ名/パス（既定: 0 = gen_000）")
    parser.add_argument("--cluster", type=str, required=True, help="対象クラスタ名（例: cl00）")
    parser.add_argument("--root", type=Path, default=config.ROOT, help="pipeline成果物のルート（既定: PIPE_ROOT）")
    parser.add_argument("--workdir", type=Path, default=None,
                         help="梱包済みagentの出力先（既定: results/search_count_cache/）")
    parser.add_argument("--search-count-a", type=int, default=10, help="比較する探索回数A（既定10）")
    parser.add_argument("--search-count-b", type=int, default=1000, help="比較する探索回数B（既定1000）")
    parser.add_argument("--games", type=int, default=5, help="対戦させる試合数（既定5）")
    parser.add_argument("--no-alternate", action="store_true", help="先手/後手を固定する（既定は1試合ごとに入れ替え）")
    parser.add_argument("--debug", action="store_true", help="kaggle_environmentsのデバッグ出力を表示する")
    parser.add_argument("--quiet", action="store_true", help="試合ごとの結果表示を省略する")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.search_count_a == args.search_count_b:
        raise SystemExit("--search-count-a と --search-count-b は異なる値にしてください。")

    gen_dir = resolve_gen_dir(args.gen, args.root)
    agdir = gen_dir / "agents" / args.cluster
    self_pth, opp_pth, deck_csv = agdir / "self.pth", agdir / "opp.pth", agdir / "deck.csv"
    for p, what in ((self_pth, "self.pth"), (opp_pth, "opp.pth"), (deck_csv, "deck.csv")):
        if not p.exists():
            raise SystemExit(f"{what} が見つかりません: {p} (--gen/--cluster を確認してください)")

    workdir = args.workdir or (config.REPO_ROOT / "results" / "search_count_cache")
    workdir.mkdir(parents=True, exist_ok=True)

    name_a = f"search{args.search_count_a}"
    name_b = f"search{args.search_count_b}"

    print(f"=== 梱包: gen={gen_dir.name} cluster={args.cluster} ===")
    out_a = package_agent(self_pth, opp_pth, deck_csv, workdir / name_a, search_count=args.search_count_a)
    out_b = package_agent(self_pth, opp_pth, deck_csv, workdir / name_b, search_count=args.search_count_b)
    print(f"  {name_a}: {out_a / 'main.py'}")
    print(f"  {name_b}: {out_b / 'main.py'}")

    spec_a = rr.AgentSpec(name=name_a, agent_path=(out_a / "main.py").resolve(), deck_path=deck_csv.resolve())
    spec_b = rr.AgentSpec(name=name_b, agent_path=(out_b / "main.py").resolve(), deck_path=deck_csv.resolve())
    agent_a = rr.load_agent(spec_a, "compare_search_count_a")
    agent_b = rr.load_agent(spec_b, "compare_search_count_b")

    print(f"\n=== 対戦開始 ({name_a} vs {name_b}, {args.games}試合) ===\n")
    started = time.time()
    h2h, _ = rr.run_pairing(
        agent_a, agent_b, num_games=args.games,
        alternate_sides=not args.no_alternate, debug=args.debug, verbose=not args.quiet,
    )
    elapsed = time.time() - started

    total = h2h.total
    print(f"\n=== 結果 ({total}試合, 所要時間{elapsed:.1f}秒) ===")
    if total > 0:
        print(f"{name_a}: {h2h.name0_wins}勝 ({h2h.name0_wins / total * 100:.1f}%)")
        print(f"{name_b}: {h2h.name1_wins}勝 ({h2h.name1_wins / total * 100:.1f}%)")
        print(f"引き分け: {h2h.draws} ({h2h.draws / total * 100:.1f}%)")
    if h2h.unresolved:
        print(f"不明(エラー): {h2h.unresolved}")


if __name__ == "__main__":
    main()
