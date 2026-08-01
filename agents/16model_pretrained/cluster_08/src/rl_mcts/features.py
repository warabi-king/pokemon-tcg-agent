"""局面と候補手をニューラルネットワーク入力へ変換する処理。"""

from __future__ import annotations

from functools import lru_cache

from cg.api import (
    AreaType,
    Card,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SelectContext,
)
from rl_mcts.model import DECODER_ATTACK_OFFSET, DECODER_MAIN_FEATURE, attack_count, card_count


class SparseVector:
    """EmbeddingBagに渡す疎な特徴量を保持するコンテナ。"""

    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_many(self, indices: list[int], value: float | int | bool) -> None:
        """同じ重みの特徴をPython method呼び出し1回でまとめて追加する。"""
        value = float(value)
        if value == 0.0 or not indices:
            return
        self.index.extend([self.pos + index for index in indices])
        self.value.extend([value] * len(indices))

    def add_cached(
        self,
        absolute_indices: tuple[int, ...],
        values: tuple[float, ...],
    ) -> None:
        """事前計算済みの絶対index/valueをlist.extendだけで追加する。"""
        self.index.extend(absolute_indices)
        self.value.extend(values)

    def add_pos(self, pos: int) -> None:
        self.pos += pos

    def add_single(self, value: float | int | bool) -> None:
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self) -> None:
        self.offset.append(len(self.index))


def decoder_card_offset() -> int:
    """decoder側でカード特徴が始まるoffsetを返す。"""
    return DECODER_ATTACK_OFFSET + attack_count()


def add_card(sv: SparseVector, card: Card | Pokemon | None) -> None:
    """encoder特徴にカードIDを追加する。"""
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(card_count())


def add_cards(sv: SparseVector, cards: list[Card] | None, value: float) -> None:
    """encoder特徴にカード集合を追加する。"""
    if cards is not None:
        sv.add_many([card.id for card in cards], value)
    sv.add_pos(card_count())


@lru_cache(maxsize=64)
def _cached_deck_features(
    position: int,
    deck: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """固定deckの60回のaddとfloat変換を評価ごとに繰り返さない。"""
    return (
        tuple(position + card_id for card_id in deck),
        (0.25,) * len(deck),
    )


def add_pokemon(sv: SparseVector, poke: Pokemon | None) -> None:
    """encoder特徴にポケモン1体の状態を追加する。"""
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count())
        return

    sv.add_single(0)
    sv.add_single(poke.hp / 400)
    add_card(sv, poke)
    add_cards(sv, poke.tools, 1.0)
    add_cards(sv, poke.energyCards, 0.5)


def add_player(sv: SparseVector, ps: PlayerState) -> None:
    """encoder特徴にプレイヤー状態を追加する。"""
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    """現在局面をencoder入力へ変換する。"""
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)

    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)

    sv.word_start()
    sv.add_cached(*_cached_deck_features(sv.pos, tuple(your_deck)))
    sv.add_pos(card_count())

    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def get_card(
    obs: Observation,
    area: AreaType,
    index: int,
    player_index: int,
) -> Pokemon | Card | None:
    """指定された領域からカードを取得する。"""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None) -> None:
    """主要行動に関係するカード特徴をdecoder入力へ追加する。"""
    if card is not None:
        sv.add(decoder_card_offset() + feature_index * card_count() + card.id, 1)


def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int) -> None:
    """context付きカードID特徴をdecoder入力へ追加する。"""
    sv.add(
        decoder_card_offset() + (DECODER_MAIN_FEATURE + int(context)) * card_count() + card_id,
        1,
    )


def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None) -> None:
    """カードが存在する場合だけdecoder入力へ追加する。"""
    if card is not None:
        decoder_card_id(sv, context, card.id)


def _add_decoder_option(
    sv: SparseVector,
    obs: Observation,
    option_index: int,
    your_index: int,
) -> None:
    """1つのoptionをdecoder特徴へ変換する。"""
    ps = obs.current.players[your_index]
    context = obs.select.context
    option = obs.select.option[option_index]
    match option.type:
        case OptionType.END:
            sv.add(1, 1)
        case OptionType.YES:
            sv.add(2, 1)
        case OptionType.NO:
            sv.add(3, 1)
        case OptionType.SPECIAL_CONDITION:
            sv.add(4 + option.specialConditionType, 1)
        case OptionType.NUMBER:
            sv.add(9 + min(option.number, 4), 1)
        case OptionType.ATTACK:
            sv.add(DECODER_ATTACK_OFFSET + option.attackId, 1)
        case OptionType.PLAY:
            decoder_main(sv, 0, ps.hand[option.index])
        case OptionType.ATTACH:
            decoder_main(sv, 1, get_card(obs, option.area, option.index, your_index))
            decoder_main(
                sv,
                2,
                get_card(obs, option.inPlayArea, option.inPlayIndex, your_index),
            )
        case OptionType.EVOLVE:
            decoder_main(sv, 3, get_card(obs, option.area, option.index, your_index))
            decoder_main(
                sv,
                4,
                get_card(obs, option.inPlayArea, option.inPlayIndex, your_index),
            )
        case OptionType.ABILITY:
            decoder_main(sv, 5, get_card(obs, option.area, option.index, your_index))
        case OptionType.DISCARD:
            decoder_main(sv, 6, get_card(obs, option.area, option.index, your_index))
        case OptionType.RETREAT:
            decoder_main(sv, 7, ps.active[0] if ps.active else None)
        case OptionType.CARD:
            decoder_card(
                sv,
                context,
                get_card(obs, option.area, option.index, option.playerIndex),
            )
        case OptionType.TOOL_CARD:
            card = get_card(obs, option.area, option.index, option.playerIndex)
            if card is not None:
                decoder_card(sv, context, card.tools[option.toolIndex])
        case OptionType.ENERGY_CARD | OptionType.ENERGY:
            card = get_card(obs, option.area, option.index, option.playerIndex)
            if card is not None:
                decoder_card(sv, context, card.energyCards[option.energyIndex])
        case OptionType.SKILL:
            decoder_card_id(sv, context, option.cardId)


def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    """候補手一覧をdecoder入力へ変換する。"""
    sv = SparseVector()
    your_index = obs.current.yourIndex
    option_reference_count = sum(len(action) for action in actions)
    cache_options = option_reference_count > len(obs.select.option)
    option_features: list[tuple[list[int], list[float]]] | None = None
    if cache_options:
        option_features = []
        for option_index in range(len(obs.select.option)):
            option_sv = SparseVector()
            _add_decoder_option(option_sv, obs, option_index, your_index)
            option_features.append((option_sv.index, option_sv.value))

    for action in actions:
        sv.word_start()

        if len(action) == 0:
            sv.add(0, 1)
            continue

        for option_index in action:
            if option_features is None:
                _add_decoder_option(sv, obs, option_index, your_index)
            else:
                indices, values = option_features[option_index]
                sv.index.extend(indices)
                sv.value.extend(values)

    return sv
