"""同じagent実装(デフォルトrl_mcts)で、2つの異なるmodel.pthを対戦させ勝率を比較する。

使い方:
    python tools/compare_models.py \
        --model-a agents/rl_mcts/src/model.pth \
        --model-b agents/rl_mcts/train/checkpoints/imitation_2000ep.pth \
        --name-a baseline --name-b imitation \
        --games 20 --search-count 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agents" / "rl_mcts"
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl_mcts.agent import RlMctsAgent  # noqa: E402
from rl_mcts.deck import read_deck_csv  # noqa: E402


def play_one_game(agent_a: RlMctsAgent, agent_b: RlMctsAgent, deck: list[int], a_is_first: bool) -> int:
    """1試合実行する。戻り値は 0=agent_a勝ち, 1=agent_b勝ち, 2=引き分け(先手/後手に依らず固定の意味)。"""
    agents = [agent_a, agent_b] if a_is_first else [agent_b, agent_a]
    obs, start_data = battle_start(deck, deck)
    if start_data.errorPlayer >= 0:
        raise ValueError(f"deck error: errorType={start_data.errorType}")

    try:
        while obs["current"]["result"] < 0:
            player_index = obs["current"]["yourIndex"]
            selected = agents[player_index].select_action(obs)
            obs = battle_select(selected)
    finally:
        battle_finish()

    raw_result = obs["current"]["result"]
    if raw_result == 2:
        return 2
    is_a_winner = (raw_result == 0) == a_is_first
    return 0 if is_a_winner else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--search-count", type=int, default=10)
    parser.add_argument("--deck", type=Path, default=None, help="省略時はagents/rl_mcts/src/deck.csvを使う")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.deck:
        deck = [int(line.strip()) for line in args.deck.read_text().splitlines() if line.strip()]
    else:
        deck = read_deck_csv()

    agent_a = RlMctsAgent(model_path=args.model_a, search_count=args.search_count)
    agent_b = RlMctsAgent(model_path=args.model_b, search_count=args.search_count)

    wins_a = wins_b = draws = 0
    for i in range(args.games):
        a_is_first = i % 2 == 0
        result = play_one_game(agent_a, agent_b, deck, a_is_first)
        if result == 0:
            wins_a += 1
        elif result == 1:
            wins_b += 1
        else:
            draws += 1

        first_label = args.name_a if a_is_first else args.name_b
        outcome_label = "draw" if result == 2 else (args.name_a if result == 0 else args.name_b)
        print(
            f"game {i + 1}/{args.games}: first={first_label} -> {outcome_label} wins "
            f"[{args.name_a} {wins_a} / {args.name_b} {wins_b} / draw {draws}]",
            flush=True,
        )

    total = wins_a + wins_b + draws
    print()
    print(f"=== 結果 ({total}試合) ===")
    print(f"{args.name_a}: {wins_a}勝 ({wins_a / total * 100:.1f}%)")
    print(f"{args.name_b}: {wins_b}勝 ({wins_b / total * 100:.1f}%)")
    print(f"引き分け: {draws} ({draws / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
