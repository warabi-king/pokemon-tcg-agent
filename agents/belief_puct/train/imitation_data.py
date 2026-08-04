"""公式リプレイJSONから模倣学習サンプルを取り出す共通ロジック。

通常のKaggleエピソードJSONに加え、``minimize_episodes.py`` が出力する
学習専用の最小JSONも同じ入口で読み込める。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from cg.api import to_observation_class  # noqa: E402
from rl_mcts.features import get_decoder_input, get_encoder_input  # noqa: E402
from rl_mcts.mcts import enumerate_actions  # noqa: E402

# (enc_index, enc_value, enc_offset, dec_index, dec_value, dec_offset, chosen_index, value)
SampleTuple = tuple[list[int], list[float], list[int], list[int], list[float], list[int], int, float]

DeckFilter = Callable[[list[int], list[int]], bool]

MINIMAL_EPISODE_FORMAT = "pokemon-tcg-agent/imitation-episode-v1"

_OPTION_FEATURE_FIELDS = (
    "type",
    "number",
    "area",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
    "cardId",
    "specialConditionType",
)


def _valid_header(rewards: object, decks: object) -> bool:
    return bool(
        isinstance(rewards, list)
        and len(rewards) == 2
        and all(reward is not None for reward in rewards)
        and isinstance(decks, list)
        and len(decks) == 2
        and all(isinstance(deck, list) and len(deck) == 60 for deck in decks)
    )


def _card_id(card: dict | int | None) -> int | None:
    if card is None or isinstance(card, int):
        return card
    return card["id"]


def _minimal_pokemon(pokemon: dict | None) -> dict | None:
    if pokemon is None:
        return None
    return {
        "id": pokemon["id"],
        "hp": pokemon["hp"],
        "energyCards": [_card_id(card) for card in pokemon["energyCards"]],
        "tools": [_card_id(card) for card in pokemon["tools"]],
    }


def _minimal_player(player: dict) -> dict:
    return {
        "active": [_minimal_pokemon(pokemon) for pokemon in player["active"]],
        "bench": [_minimal_pokemon(pokemon) for pokemon in player["bench"]],
        "deckCount": player["deckCount"],
        "discard": [_card_id(card) for card in player["discard"]],
        "prize": [_card_id(card) for card in player["prize"]],
        "handCount": player["handCount"],
        "hand": None
        if player["hand"] is None
        else [_card_id(card) for card in player["hand"]],
        "poisoned": player["poisoned"],
        "burned": player["burned"],
        "asleep": player["asleep"],
        "paralyzed": player["paralyzed"],
        "confused": player["confused"],
    }


def _minimal_observation(observation: dict) -> dict:
    current = observation["current"]
    select = observation["select"]
    return {
        "turn": current["turn"],
        "firstPlayer": current["firstPlayer"],
        "stadium": [_card_id(card) for card in current["stadium"]],
        "looking": None
        if current["looking"] is None
        else [_card_id(card) for card in current["looking"]],
        "players": [_minimal_player(player) for player in current["players"]],
        "select": {
            "context": select["context"],
            "maxCount": select["maxCount"],
            "option": [
                {
                    key: option[key]
                    for key in _OPTION_FEATURE_FIELDS
                    if key in option and option[key] is not None
                }
                for option in select["option"]
            ],
            "deck": None
            if select["deck"] is None
            else [_card_id(card) for card in select["deck"]],
        },
    }


def _full_episode_parts(episode: dict) -> tuple[list, list[list[int]], list[dict]] | None:
    rewards = episode.get("rewards")
    steps = episode.get("steps")
    if not isinstance(steps, list) or len(steps) < 3:
        return None

    try:
        decks = [steps[1][0]["action"], steps[1][1]["action"]]
    except (IndexError, KeyError, TypeError):
        return None
    if not _valid_header(rewards, decks):
        return None

    decisions: list[dict] = []
    for player in range(2):
        for index in range(1, len(steps) - 1):
            observation = steps[index][player].get("observation")
            select = observation.get("select") if isinstance(observation, dict) else None
            if select is None:
                continue

            actual_action = steps[index + 1][player].get("action")
            if actual_action is None:
                continue

            actions = enumerate_actions(len(select["option"]), select["maxCount"])
            target = tuple(sorted(actual_action))
            chosen_index = next(
                (
                    candidate_index
                    for candidate_index, candidate in enumerate(actions)
                    if tuple(candidate) == target
                ),
                None,
            )
            if chosen_index is None:
                continue

            decisions.append(
                {
                    "player": player,
                    "observation": observation,
                    "chosenIndex": chosen_index,
                }
            )

    return rewards, decks, decisions


def minimize_episode(data: bytes) -> dict | None:
    """元エピソードを、現在の模倣学習が実際に読む情報だけのJSONへ変換する。

    actionそのものは合法手列挙後の正解クラス ``chosenIndex`` に変換する。
    ログ、表示用情報、時間、status、途中rewardなどは出力しない。
    """
    episode = json.loads(data)
    if not isinstance(episode, dict):
        return None
    if episode.get("format") == MINIMAL_EPISODE_FORMAT:
        if not _valid_header(episode.get("rewards"), episode.get("decks")):
            return None
        return episode if isinstance(episode.get("decisions"), list) else None

    parts = _full_episode_parts(episode)
    if parts is None:
        return None
    rewards, decks, decisions = parts
    return {
        "format": MINIMAL_EPISODE_FORMAT,
        "rewards": rewards,
        "decks": decks,
        "decisions": [
            {
                "player": decision["player"],
                "observation": _minimal_observation(decision["observation"]),
                "chosenIndex": decision["chosenIndex"],
            }
            for decision in decisions
        ],
    }


def _inflate_card(card_id: int | None, player: int) -> dict | None:
    if card_id is None:
        return None
    return {"id": card_id, "serial": 0, "playerIndex": player}


def _inflate_pokemon(pokemon: dict | None, player: int) -> dict | None:
    if pokemon is None:
        return None
    return {
        "id": pokemon["id"],
        "serial": 0,
        "hp": pokemon["hp"],
        "maxHp": 0,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [_inflate_card(card_id, player) for card_id in pokemon["energyCards"]],
        "tools": [_inflate_card(card_id, player) for card_id in pokemon["tools"]],
        "preEvolution": [],
    }


def _inflate_player(player_state: dict, player: int) -> dict:
    hand = player_state["hand"]
    return {
        "active": [_inflate_pokemon(pokemon, player) for pokemon in player_state["active"]],
        "bench": [_inflate_pokemon(pokemon, player) for pokemon in player_state["bench"]],
        "benchMax": 0,
        "deckCount": player_state["deckCount"],
        "discard": [_inflate_card(card_id, player) for card_id in player_state["discard"]],
        "prize": [_inflate_card(card_id, player) for card_id in player_state["prize"]],
        "handCount": player_state["handCount"],
        "hand": None if hand is None else [_inflate_card(card_id, player) for card_id in hand],
        "poisoned": player_state["poisoned"],
        "burned": player_state["burned"],
        "asleep": player_state["asleep"],
        "paralyzed": player_state["paralyzed"],
        "confused": player_state["confused"],
    }


def _inflate_observation(observation: dict, player: int) -> dict:
    select = observation["select"]
    return {
        "select": {
            "type": 0,
            "context": select["context"],
            "minCount": 0,
            "maxCount": select["maxCount"],
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": select["option"],
            "deck": None
            if select["deck"] is None
            else [_inflate_card(card_id, player) for card_id in select["deck"]],
            "contextCard": None,
            "effect": None,
        },
        "logs": [],
        "current": {
            "turn": observation["turn"],
            "turnActionCount": 0,
            "yourIndex": player,
            "firstPlayer": observation["firstPlayer"],
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [_inflate_card(card_id, player) for card_id in observation["stadium"]],
            "looking": None
            if observation["looking"] is None
            else [_inflate_card(card_id, player) for card_id in observation["looking"]],
            "players": [
                _inflate_player(player_state, player_index)
                for player_index, player_state in enumerate(observation["players"])
            ],
        },
        "search_begin_input": None,
    }


def _make_sample(
    observation: dict,
    deck: list[int],
    chosen_index: int,
    value: float,
) -> SampleTuple | None:
    obs = to_observation_class(observation)
    actions = enumerate_actions(len(obs.select.option), obs.select.maxCount)
    if not 0 <= chosen_index < len(actions):
        return None

    sv_enc = get_encoder_input(obs, deck)
    sv_dec = get_decoder_input(obs, actions)
    return (
        sv_enc.index,
        sv_enc.value,
        sv_enc.offset,
        sv_dec.index,
        sv_dec.value,
        sv_dec.offset,
        chosen_index,
        value,
    )


def extract_player_samples_from_episode(
    data: bytes,
) -> tuple[list[list[int]], list[list[SampleTuple]]] | None:
    """1試合を一度だけエンコードし、デッキとplayer別サンプルを返す。"""
    episode = json.loads(data)
    if not isinstance(episode, dict):
        return None
    minimal = episode.get("format") == MINIMAL_EPISODE_FORMAT

    if minimal:
        rewards = episode.get("rewards")
        decks = episode.get("decks")
        decisions = episode.get("decisions")
        if not _valid_header(rewards, decks) or not isinstance(decisions, list):
            return None
    else:
        parts = _full_episode_parts(episode)
        if parts is None:
            return None
        rewards, decks, decisions = parts

    player_samples: list[list[SampleTuple]] = [[], []]
    for player in range(2):
        your_deck = decks[player]
        value = float(rewards[player])
        for decision in decisions:
            if decision.get("player") != player:
                continue
            observation = decision["observation"]
            if minimal:
                observation = _inflate_observation(observation, player)
            sample = _make_sample(observation, your_deck, decision["chosenIndex"], value)
            if sample is not None:
                player_samples[player].append(sample)

    return decks, player_samples


def extract_samples_from_episode(data: bytes, deck_filter: DeckFilter | None = None) -> list[SampleTuple]:
    """通常JSONまたは最小JSONから1エピソード分の学習サンプルを取り出す。"""
    extracted = extract_player_samples_from_episode(data)
    if extracted is None:
        return []
    decks, player_samples = extracted

    samples: list[SampleTuple] = []
    for player in range(2):
        if deck_filter is None or deck_filter(decks[player], decks[1 - player]):
            samples.extend(player_samples[player])
    return samples
