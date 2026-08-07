"""公式リプレイ(エピソードJSON)から模倣学習サンプルを取り出す共通ロジック。

各エピソードJSONの steps[i][player]["observation"]["select"] が提示された選択肢で、
それに対する実際の回答は steps[i+1][player]["action"] に入っている
(kaggle_environmentsの一般的な規約: action[i]はobservation[i-1]への回答)。

policyは実際に選ばれた手を正解クラスとした分類(交差エントロピーで学習)、
valueはMCTS探索を使わず、そのエピソードの実際の勝敗(rewards)を教師にする。

value_decay(既定1.0=無効)を1未満にすると、終局に近い局面ほど勝敗(±1)そのものを、
終局から遠い(序盤の)局面ほど0に近い値を教師にする(指数減衰)。全局面へ一律で
最終結果を貼ると、五分の序盤局面までvalueヘッドが±1へ過学習し飽和しやすいため
(実測: 経路依存の弱い模倣学習の重みで root_value が同一デッキのミラー戦3ターン目でも
+0.9台に張り付く現象を確認)、経過ターン(終局からの距離)に応じて教師を緩和する。
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

DeckFilter = Callable[[list[int], list[int]], bool]


def extract_samples_from_episode(
    data: bytes,
    deck_filter: DeckFilter | None = None,
    value_decay: float = 1.0,
) -> list[SampleTuple]:
    """1エピソード分のJSONから学習サンプルを取り出す。

    deck_filterを指定すると、そのプレイヤーの(your_deck, opponent_deck)で
    deck_filter(your_deck, opponent_deck)がTrueを返す場合だけ採用する
    (デッキグループでの絞り込みに使う。自分のデッキ基準・相手のデッキ基準の
    どちらでも絞り込めるように両方を渡す)。

    value_decayはモジュールdocstring参照。1.0なら従来通り全局面に
    最終結果をそのまま付与する(既定、後方互換)。
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
        opponent_deck = decks[1 - player]
        if deck_filter is not None and not deck_filter(your_deck, opponent_deck):
            continue

        final_value = float(rewards[player])

        # value(教師)を確定させる前に、まずこのプレイヤーの決断局面だけを集める
        # (末尾の(chosen_index)まで。valueは終局からの距離が分かってから付与する)。
        player_samples: list[tuple] = []
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
            player_samples.append(
                (
                    sv_enc.index,
                    sv_enc.value,
                    sv_enc.offset,
                    sv_dec.index,
                    sv_dec.value,
                    sv_dec.offset,
                    chosen_index,
                )
            )

        # 終局に一番近い局面(末尾)にfinal_valueをそのまま付与し、そこから遡るほど
        # value_decay倍ずつ0へ近づける。value_decay=1.0なら全局面が従来通りfinal_value。
        n = len(player_samples)
        for offset, sample in enumerate(player_samples):
            distance_from_end = n - 1 - offset
            decayed_value = final_value * (value_decay**distance_from_end)
            samples.append((*sample, decayed_value))

    return samples
