"""公式リプレイ(エピソードJSON)から模倣学習サンプルを取り出す共通ロジック。

各エピソードJSONの steps[i][player]["observation"]["select"] が提示された選択肢で、
それに対する実際の回答は steps[i+1][player]["action"] に入っている
(kaggle_environmentsの一般的な規約: action[i]はobservation[i-1]への回答)。

policyは実際に選ばれた手を正解クラスとした分類、valueはMCTS探索を使わず、
そのエピソードの実際の勝敗(rewards, 勝+1/負-1/分0)から計算した割引リターンを教師にする。

割引リターン G_t = r_終局 * x^d (d = その局面から終局までの、そのプレイヤー自身の手数)。
割引率 x は「対局ごと」に決め、その対局の最初の手の割引が first_move_discount(既定0.3)
になるよう x = first_move_discount^(1/(N-1)) とする(N = そのプレイヤーの決断回数)。
これにより、対局の長さに依らず「終局直前の手 = full ±1、最初の手 ≈ 0.3」に揃う。
first_move_discount=1.0 なら全局面が full ±1(=割引なし)。

この G_t は value ヘッドの回帰教師であると同時に、train_imitation.py の AWR
(advantage-weighted imitation)における advantage A_t = G_t - V(s_t) のベースにもなる。
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


def discounted_return_series(
    final_value: float, n: int, first_move_discount: float = 0.3
) -> list[float]:
    """1プレイヤーのn個の決断局面(index 0=最初の手 〜 n-1=終局直前の手)について、
    割引リターン G_t = final_value * x^d の列を返す(d = 終局までの残り手数 = n-1-index)。

    割引率 x はこの対局のNから x = first_move_discount^(1/(n-1)) と決める。これにより
    最後の手(d=0)は full な final_value、最初の手(d=n-1)は final_value*first_move_discount
    になり、対局の長さに依らず「最初の手の割引 ≈ first_move_discount」に揃う。

    n<=1(決断が1回以下)は割引を定義できないので final_value をそのまま返す。
    first_move_discount=1.0 なら x=1 で全局面 full(=割引なし・後方互換)。
    """
    if n <= 0:
        return []
    if n == 1:
        return [final_value]
    x = first_move_discount ** (1.0 / (n - 1))
    return [final_value * (x ** (n - 1 - index)) for index in range(n)]


def extract_samples_from_episode(
    data: bytes,
    deck_filter: DeckFilter | None = None,
    first_move_discount: float = 1.0,
) -> list[SampleTuple]:
    """1エピソード分のJSONから学習サンプルを取り出す。

    deck_filterを指定すると、そのプレイヤーの(your_deck, opponent_deck)で
    deck_filter(your_deck, opponent_deck)がTrueを返す場合だけ採用する
    (デッキグループでの絞り込みに使う。自分のデッキ基準・相手のデッキ基準の
    どちらでも絞り込めるように両方を渡す)。

    first_move_discountはモジュールdocstring/ discounted_return_series 参照。
    1.0なら従来通り全局面に最終結果をそのまま付与する(既定、後方互換)。
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

        # 対局ごとの割引率で G_t を計算(末尾=full±1、最初の手≈first_move_discount)。
        returns = discounted_return_series(final_value, len(player_samples), first_move_discount)
        for sample, g_t in zip(player_samples, returns):
            samples.append((*sample, g_t))

    return samples
