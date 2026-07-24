"""match_agentsのデッキ・初期重みから学習可能なagentを生成する。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "agents" / "match_agents",
        help="deck.csvとmodel.pthを持つ番号別ディレクトリ",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "agents" / "rl_mcts_r_robin1" / "src",
        help="cg・rl_mcts・main.pyの生成元",
    )
    parser.add_argument("--prefix", default="rl_mcts_match_", help="生成agent名のprefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(path for path in args.source.iterdir() if path.is_dir())
    if len(sources) != 8:
        raise SystemExit(f"match agentは8個必要です: {args.source} ({len(sources)}個)")

    for source in sources:
        deck_path = source / "deck.csv"
        model_path = source / "model.pth"
        deck = [line for line in deck_path.read_text().splitlines() if line.strip()]
        if len(deck) != 60:
            raise SystemExit(f"deck.csvは60枚必要です: {deck_path} ({len(deck)}枚)")
        if not model_path.exists():
            raise SystemExit(f"model.pthがありません: {model_path}")

        target = ROOT / "agents" / f"{args.prefix}{source.name}"
        if target.exists():
            raise SystemExit(f"生成先が既に存在します: {target}")
        target_src = target / "src"
        target_src.mkdir(parents=True)
        shutil.copy2(args.template / "main.py", target_src / "main.py")
        shutil.copytree(
            args.template / "cg",
            target_src / "cg",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        shutil.copytree(
            args.template / "rl_mcts",
            target_src / "rl_mcts",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        shutil.copy2(deck_path, target_src / "deck.csv")
        shutil.copy2(model_path, target_src / "model.pth")
        (target / "README.md").write_text(
            f"# {target.name}\n\n"
            f"`agents/match_agents/{source.name}` のデッキとモデルを初期値にした"
            "中央ラウンドロビン学習agentです。\n",
            encoding="utf-8",
        )
        print(f"created {target.name} from match_agents/{source.name}")


if __name__ == "__main__":
    main()
