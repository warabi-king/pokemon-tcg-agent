"""手札モデルだけを無効にしたA/Bテスト用エントリポイント。"""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from cg.api import Observation, to_observation_class  # noqa: E402
from rl_mcts.agent import RlMctsAgent  # noqa: E402
from rl_mcts.deck import read_deck_csv  # noqa: E402


_AGENT = RlMctsAgent(
    model_path=SRC_ROOT / "model.pth",
    opponent_model_path=SRC_ROOT / "opponent_model.pth",
    hand_model_path=SRC_ROOT / "rl_mcts" / "hand_model.pth",
    deck_database_path=SRC_ROOT / "rl_mcts" / "deck_candidates_by_wins.jsonl",
    use_hand_model=False,
    belief_seed=20260807,
)


def agent(obs_dict: dict) -> list[int]:
    """DBデッキ推定は使い、手札保持スコアだけを0にした比較agent。"""

    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return _AGENT.select_action(obs_dict)
