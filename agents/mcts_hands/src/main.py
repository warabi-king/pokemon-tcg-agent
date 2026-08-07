from __future__ import annotations

from pathlib import Path

from cg.api import Observation, to_observation_class
from rl_mcts.agent import RlMctsAgent
from rl_mcts.deck import read_deck_csv

_SRC_ROOT = Path(__file__).resolve().parent
_AGENT = RlMctsAgent(
    model_path=_SRC_ROOT / "model.pth",
    opponent_model_path=_SRC_ROOT / "opponent_model.pth",
    hand_model_path=_SRC_ROOT / "rl_mcts" / "hand_model.pth",
    deck_database_path=_SRC_ROOT / "rl_mcts" / "deck_candidates_by_wins.jsonl",
    use_hand_model=True,
    belief_seed=20260807,
)


def agent(obs_dict: dict) -> list[int]:
    """Kaggle/cabtから呼び出されるエージェント本体。"""
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()

    return _AGENT.select_action(obs_dict)
