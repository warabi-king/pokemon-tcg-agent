"""Kaggle/cabtから呼ばれるbelief-aware PUCT agentのentrypoint。"""

from __future__ import annotations

from cg.api import Observation, to_observation_class
from rl_mcts.agent import RlMctsAgent
from rl_mcts.deck import read_deck_csv

_AGENT = RlMctsAgent(search_count=24, determinizations=1)


def agent(obs_dict: dict) -> list[int]:
    """Kaggle/cabtから呼び出されるエージェント本体。"""
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        _AGENT.reset_match()
        return read_deck_csv()

    return _AGENT.select_action(obs_dict)
