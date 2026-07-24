"""公式リプレイ(エピソードJSON)から模倣学習サンプルを取り出す共通ロジック。

各エピソードJSONの steps[i][player]["observation"]["select"] が提示された選択肢で、
それに対する実際の回答は steps[i+1][player]["action"] に入っている
(kaggle_environmentsの一般的な規約: action[i]はobservation[i-1]への回答)。

policyは実際に選ばれた手を正解クラスとした分類(交差エントロピーで学習)、
valueはMCTS探索を使わず、そのエピソードの実際の勝敗(rewards)をそのまま使う。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agents" / "rl_mcts"
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cg.api import to_observation_class  # noqa: E402
from rl_mcts.features import get_decoder_input, get_encoder_input  # noqa: E402
from rl_mcts.mcts import enumerate_actions  # noqa: E402

# (enc_index, enc_value, enc_offset, dec_index, dec_value, dec_offset, chosen_index, value)
SampleTuple = tuple[list[int], list[float], list[int], list[int], list[float], list[int], int, float]

DeckFilter = Callable[[list[int]], bool]


def extract_samples_from_episode(data: bytes, deck_filter: DeckFilter | None = None) -> list[SampleTuple]:
    """1エピソード分のJSONから学習サンプルを取り出す。

    deck_filterを指定すると、そのプレイヤーのデッキがdeck_filter(deck)でTrueを返す
    場合だけ採用する(デッキグループでの絞り込みに使う)。
    """
    j = json.loads(data)
    rewards = j.get("rewards")
    if not rewards or len(rewards) != 2 or any(r is None for r in rewards):
        return []

    steps = j["steps"]
    if len(steps) < 3:
        return []

    decks = [steps[1][0]["action"], steps[1][1]["action"]]
    if len(decks[0]) != 60 or len(decks[1]) != 60:
        return []

    samples: list[SampleTuple] = []
    for player in range(2):
        your_deck = decks[player]
        if deck_filter is not None and not deck_filter(your_deck):
            continue

        value = float(rewards[player])

        for i in range(1, len(steps) - 1):
            sel = steps[i][player]["observation"].get("select")
            if sel is None:
                continue

            actual_action = steps[i + 1][player]["action"]
            if actual_action is None:
                continue

            obs = to_observation_class(steps[i][player]["observation"])
            actions = enumerate_actions(len(obs.select.option), obs.select.maxCount)
            if not actions:
                continue

            target = tuple(sorted(actual_action))
            chosen_index = next(
                (idx for idx, candidate in enumerate(actions) if tuple(candidate) == target),
                None,
            )
            if chosen_index is None:
                continue

            sv_enc = get_encoder_input(obs, your_deck)
            sv_dec = get_decoder_input(obs, actions)
            samples.append(
                (
                    sv_enc.index,
                    sv_enc.value,
                    sv_enc.offset,
                    sv_dec.index,
                    sv_dec.value,
                    sv_dec.offset,
                    chosen_index,
                    value,
                )
            )

    return samples
