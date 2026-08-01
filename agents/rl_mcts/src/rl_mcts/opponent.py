from __future__ import annotations

import random
from collections import Counter
from functools import lru_cache

from cg.api import Card, CardType, Log, Pokemon, State, all_card_data
from rl_mcts.deck_belief_db import predict_full_deck

OPPONENT_DECKS: dict[str, list[int]] = {
    "00": [
        292, 292, 292, 292, 293, 293, 293, 293, 303, 303, 906, 906, 277, 257, 258,
        1113, 1113, 1113, 1113, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1097, 1097, 1118, 1118,
        1123, 1123, 1123, 1088, 1221, 1221, 1221, 1227, 1227, 1227, 1227, 1182, 1182, 1253, 1253,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    ],
    "01": [
        677, 677, 677, 677, 678, 678, 678, 678, 117, 117, 117, 886, 886, 1142, 1142,
        1142, 1142, 1121, 1121, 1121, 1121, 1145, 1145, 1145, 1145, 1118, 1118, 1118, 1097, 1097,
        1123, 1123, 1123, 1088, 1227, 1227, 1227, 1227, 1182, 1182, 1211, 1211, 1122, 1122, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 20, 20, 20, 20,
    ],
    "02": [
        661, 661, 661, 661, 662, 662, 662, 662, 795, 795, 259, 259, 1121, 1121, 1121,
        1121, 1145, 1145, 1145, 1145, 1123, 1123, 1123, 1097, 1097, 1118, 1118, 1122, 1122, 1148,
        1148, 1163, 1163, 1092, 1227, 1227, 1227, 1227, 1182, 1182, 1232, 1232, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    ],
    "03": [
        96, 96, 96, 96, 149, 149, 149, 149, 347, 347, 150, 150, 150, 1079, 1079,
        1079, 1079, 1086, 1086, 1086, 1086, 1094, 1094, 1094, 1094, 1121, 1121, 1121, 1127, 1127,
        1118, 1118, 1097, 1097, 1123, 1123, 1088, 1227, 1227, 1227, 1227, 1182, 1182, 1261, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ],
    "04": [
        268, 268, 268, 268, 269, 269, 269, 269, 265, 265, 266, 266, 270, 270, 271,
        271, 1121, 1121, 1121, 1121, 1122, 1122, 1122, 1118, 1118, 1118, 1097, 1097, 1123, 1123,
        1123, 1233, 1233, 1233, 1227, 1227, 1227, 1227, 1182, 1182, 1254, 1254, 1088, 4, 4,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    ],
    "05": [
        169, 169, 169, 169, 190, 190, 190, 190, 547, 547, 547, 695, 695, 1121, 1121,
        1121, 1121, 1118, 1118, 1118, 1097, 1097, 1097, 1123, 1123, 1123, 1140, 1140, 1116, 1116,
        1122, 1122, 1128, 1227, 1227, 1227, 1227, 1182, 1182, 1244, 1244, 1145, 1145, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    ],
    "06": [
        745, 745, 745, 745, 746, 746, 746, 747, 747, 747, 766, 766, 272, 272, 1079,
        1079, 1079, 1079, 1121, 1121, 1121, 1121, 1086, 1086, 1086, 1086, 1145, 1145, 1145, 1118,
        1118, 1123, 1123, 1097, 1097, 1088, 1227, 1227, 1227, 1227, 1182, 1182, 1263, 1263, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 19, 19, 19, 19,
    ],
    "07": [
        721, 721, 722, 722, 722, 722, 723, 723, 723, 723, 1092, 1121, 1121, 1145, 1145,
        1163, 1163, 1219, 1219, 1219, 1219, 1227, 1227, 1227, 1227, 1262, 1262, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    ],
}


@lru_cache(maxsize=1)
def _card_lookup() -> dict[int, object]:
    return {card.cardId: card for card in all_card_data()}


def _card_type(card_id: int) -> int | None:
    card = _card_lookup().get(card_id)
    if card is None:
        return None
    return int(card.cardType)


def _is_energy_card(card_id: int) -> bool:
    return _card_type(card_id) in (int(CardType.BASIC_ENERGY), int(CardType.SPECIAL_ENERGY))


def _is_basic_pokemon(card_id: int) -> bool:
    card = _card_lookup().get(card_id)
    return bool(card is not None and int(card.cardType) == int(CardType.POKEMON) and card.basic)


def _add_energy(counter: Counter[int], card: Card | None) -> None:
    if card is not None and _is_energy_card(card.id):
        counter[card.id] += 1


def _add_pokemon_energies(counter: Counter[int], pokemon: Pokemon | None) -> None:
    if pokemon is None:
        return
    for card in pokemon.energyCards:
        _add_energy(counter, card)


def revealed_opponent_energy_ids(state: State, opponent_index: int, logs: list[Log]) -> Counter[int]:
    energies: Counter[int] = Counter()
    opponent = state.players[opponent_index]

    for pokemon in opponent.active:
        _add_pokemon_energies(energies, pokemon)
    for pokemon in opponent.bench:
        _add_pokemon_energies(energies, pokemon)
    for card in opponent.discard:
        _add_energy(energies, card)
    for card in opponent.prize:
        _add_energy(energies, card)

    for log in logs:
        if log.playerIndex == opponent_index and log.cardId is not None and _is_energy_card(log.cardId):
            energies[log.cardId] += 1

    return energies


def _add_pokemon_all_cards(counter: Counter[int], pokemon: Pokemon | None) -> None:
    """ポケモン1体から観測できる全カードID（本体・進化元・付随エネ/道具）を数える。"""
    if pokemon is None:
        return
    counter[pokemon.id] += 1
    for field_name in ("preEvolution", "energyCards", "tools"):
        for card in getattr(pokemon, field_name, None) or []:
            if card is not None:
                counter[card.id] += 1


def revealed_opponent_card_ids(state: State, opponent_index: int, logs: list[Log]) -> Counter[int]:
    """相手について観測できた全カードID（エネルギーに限らない）を集計する。"""
    cards: Counter[int] = Counter()
    opponent = state.players[opponent_index]

    for pokemon in opponent.active:
        _add_pokemon_all_cards(cards, pokemon)
    for pokemon in opponent.bench:
        _add_pokemon_all_cards(cards, pokemon)
    for card in opponent.discard:
        if card is not None:
            cards[card.id] += 1

    for log in logs:
        if log.playerIndex == opponent_index and log.cardId is not None:
            cards[log.cardId] += 1

    return cards


def infer_opponent_deck(state: State, opponent_index: int, logs: list[Log]) -> list[int]:
    # ② 候補DB(633デッキ)から観測カードに最も合致する実在デッキを推定する。
    # DB無し・観測ゼロ・失敗時は None が返るので、従来の固定デッキ辞書へフォールバック。
    observed = revealed_opponent_card_ids(state, opponent_index, logs)
    predicted = predict_full_deck(observed.elements()) if observed else None
    if predicted is not None:
        return predicted

    # --- フォールバック: エネルギー構成で固定8デッキの最寄りを選ぶ（旧①方式）---
    revealed_energies = revealed_opponent_energy_ids(state, opponent_index, logs)
    if not revealed_energies:
        return OPPONENT_DECKS["00"]

    def score(deck: list[int]) -> tuple[int, int]:
        deck_counts = Counter(card_id for card_id in deck if _is_energy_card(card_id))
        matched = sum(min(count, deck_counts.get(card_id, 0)) for card_id, count in revealed_energies.items())
        extra = sum(count for card_id, count in revealed_energies.items() if card_id not in deck_counts)
        return matched, -extra

    return max(OPPONENT_DECKS.values(), key=score)


def sample_from_deck(deck: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    return random.sample(deck, min(len(deck), count))


def predict_facedown_active(deck: list[int]) -> list[int]:
    basics = [card_id for card_id in deck if _is_basic_pokemon(card_id)]
    if not basics:
        return []
    return [random.choice(basics)]
