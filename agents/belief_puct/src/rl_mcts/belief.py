"""観測済みカードから60枚デッキと非公開zoneを復元する。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Any, Iterable, MutableMapping, Sequence

DECK_SIZE = 60
DEFAULT_DATABASE = Path(__file__).with_name("deck_candidates_by_wins.jsonl")


class DeckBeliefError(ValueError):
    """観測zoneと60枚デッキの整合性を作れない。"""


@dataclass(frozen=True)
class HiddenZoneSample:
    """libcg SearchBeginへ渡す、1つの整合した非公開情報sample。"""

    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]
    full_decks: tuple[tuple[int, ...], tuple[int, ...]]


def load_candidates(database_path: Path) -> list[dict[str, Any]]:
    """同梱JSONLから60枚の候補デッキだけを読み込む。

    提出物の外部パスには依存しない。壊れた行は候補DB全体の破損を隠さないため、
    JSON例外を呼び出し側へ伝える。
    """

    records: list[dict[str, Any]] = []
    with database_path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record.get("deck"), list) and len(record["deck"]) == DECK_SIZE:
                records.append(record)
    if not records:
        raise DeckBeliefError(f"有効なデッキ候補がありません: {database_path}")
    return records


def _card_id(card: Any) -> int | None:
    if not isinstance(card, dict):
        return None
    value = card.get("id")
    return int(value) if value is not None else None


def _card_serial(card: Any) -> int | None:
    if not isinstance(card, dict):
        return None
    value = card.get("serial")
    return int(value) if value is not None else None


def _add_card(
    card: Any,
    serial_cards: MutableMapping[int, int],
    anonymous_cards: list[int],
) -> None:
    card_id = _card_id(card)
    if card_id is None:
        return
    serial = _card_serial(card)
    if serial is None:
        anonymous_cards.append(card_id)
    else:
        serial_cards[serial] = card_id


def _add_pokemon(
    pokemon: Any,
    serial_cards: MutableMapping[int, int],
    anonymous_cards: list[int],
) -> None:
    if not isinstance(pokemon, dict):
        return
    _add_card(pokemon, serial_cards, anonymous_cards)
    for field_name in ("preEvolution", "energyCards", "tools"):
        for card in pokemon.get(field_name) or []:
            _add_card(card, serial_cards, anonymous_cards)


def public_cards_by_player(
    state: dict[str, Any],
    select: dict[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """現在、非公開の山札・サイド・手札以外にあるカードをowner別に返す。"""
    serial_cards: tuple[dict[int, int], dict[int, int]] = ({}, {})
    anonymous_cards: tuple[list[int], list[int]] = ([], [])
    known_hidden_serials: tuple[set[int], set[int]] = (set(), set())
    players = state.get("players") or []
    for player_index, player in enumerate(players[:2]):
        for zone_name in ("hand", "prize"):
            for card in player.get(zone_name) or []:
                if (serial := _card_serial(card)) is not None:
                    known_hidden_serials[player_index].add(serial)
        for pokemon in player.get("active") or []:
            _add_pokemon(
                pokemon,
                serial_cards[player_index],
                anonymous_cards[player_index],
            )
        for pokemon in player.get("bench") or []:
            _add_pokemon(
                pokemon,
                serial_cards[player_index],
                anonymous_cards[player_index],
            )
        for card in player.get("discard") or []:
            _add_card(
                card,
                serial_cards[player_index],
                anonymous_cards[player_index],
            )

    for stadium in state.get("stadium") or []:
        if not isinstance(stadium, dict):
            continue
        owner = stadium.get("playerIndex")
        if owner in (0, 1):
            _add_card(stadium, serial_cards[int(owner)], anonymous_cards[int(owner)])

    # 「上からN枚を見る」等の解決中はlookingのカードがdeckCountから一時的に
    # 外れる。Card.playerIndexを使って元のownerへ数える。
    for card in state.get("looking") or []:
        if not isinstance(card, dict):
            continue
        owner = card.get("playerIndex")
        if owner in (0, 1):
            _add_card(card, serial_cards[int(owner)], anonymous_cards[int(owner)])

    # 効果解決中のカードは手札等から既に外れているが、まだトラッシュや場へ
    # 移っていない。select.deckは山札そのものなのでここには含めない。
    for field_name in ("contextCard", "effect"):
        card = (select or {}).get(field_name)
        if not isinstance(card, dict):
            continue
        owner = card.get("playerIndex")
        if owner in (0, 1):
            owner = int(owner)
            serial = _card_serial(card)
            # 選択中のcontextCardがまだ手札に残っている場合がある。
            # 同じ物理カードを2つのzoneとして数えない。
            if serial is not None and serial in known_hidden_serials[owner]:
                continue
            _add_card(card, serial_cards[owner], anonymous_cards[owner])

    return tuple(  # type: ignore[return-value]
        list(serial_cards[index].values()) + anonymous_cards[index]
        for index in range(2)
    )


def _known_cards(cards: Any) -> list[int]:
    if not isinstance(cards, list):
        return []
    return [card_id for card in cards if (card_id := _card_id(card)) is not None]


def _reconcile_transient_public_cards(
    public_cards: list[int],
    state: dict[str, Any],
    select: dict[str, Any] | None,
    player_index: int,
) -> list[int]:
    """zone数に対して余分な、効果解決中カードの重複だけを除く。"""
    player = (state.get("players") or [])[player_index]
    hand_count = int(
        player.get("handCount", len(player.get("hand") or []))
    )
    expected_public_count = (
        DECK_SIZE
        - int(player.get("deckCount", 0))
        - len(player.get("prize") or [])
        - hand_count
    )
    excess = len(public_cards) - expected_public_count
    if excess <= 0:
        return public_cards

    # contextCard/effectやlookingは、エンジンの効果解決段階によって元zoneにも
    # 一時的に残ることがある。安定zone（場・トラッシュ等）には触れず、これらを
    # 優先して余分な分だけ取り除く。
    transient_ids: list[int] = []
    for field_name in ("contextCard", "effect"):
        card = (select or {}).get(field_name)
        if isinstance(card, dict) and card.get("playerIndex") == player_index:
            if (card_id := _card_id(card)) is not None:
                transient_ids.append(card_id)
    for card in state.get("looking") or []:
        if isinstance(card, dict) and card.get("playerIndex") == player_index:
            if (card_id := _card_id(card)) is not None:
                transient_ids.append(card_id)

    reconciled = list(public_cards)
    for card_id in transient_ids:
        if excess <= 0:
            break
        try:
            reconciled.remove(card_id)
        except ValueError:
            continue
        excess -= 1
    if excess:
        raise DeckBeliefError(
            "公開zoneの枚数が60枚構成と一致せず、解決中カードだけでは"
            f"補正できません: player={player_index}, "
            f"public={len(public_cards)}, expected={expected_public_count}"
        )
    return reconciled


def update_seen_opponent_cards(
    observation: dict[str, Any],
    seen_by_viewer: tuple[dict[int, int], dict[int, int]],
) -> None:
    """viewerが実際に確認できた相手カードをserial単位で累積する。"""
    state = observation.get("current")
    if not isinstance(state, dict):
        return
    viewer = int(state.get("yourIndex", -1))
    if viewer not in (0, 1):
        return
    opponent = 1 - viewer
    target = seen_by_viewer[viewer]

    select = observation.get("select") or {}
    public_cards = public_cards_by_player(state, select)[opponent]
    # public_cards_by_playerは重複を除いた現在値だが、履歴保持にはserialが必要なため
    # 現在zoneをもう一度辿ってtargetへ登録する。serialのない軽量fixtureだけは
    # 負の仮serialで同一観測内の枚数を保持する。
    player = (state.get("players") or [])[opponent]
    anonymous_serial = -1

    def remember(card: Any) -> None:
        nonlocal anonymous_serial
        card_id = _card_id(card)
        if card_id is None:
            return
        serial = _card_serial(card)
        if serial is None:
            while anonymous_serial in target:
                anonymous_serial -= 1
            target[anonymous_serial] = card_id
            anonymous_serial -= 1
        else:
            target[serial] = card_id

    def remember_pokemon(pokemon: Any) -> None:
        if not isinstance(pokemon, dict):
            return
        remember(pokemon)
        for field_name in ("preEvolution", "energyCards", "tools"):
            for card in pokemon.get(field_name) or []:
                remember(card)

    for pokemon in player.get("active") or []:
        remember_pokemon(pokemon)
    for pokemon in player.get("bench") or []:
        remember_pokemon(pokemon)
    for card in player.get("discard") or []:
        remember(card)
    for card in player.get("prize") or []:
        remember(card)
    for stadium in state.get("stadium") or []:
        if isinstance(stadium, dict) and stadium.get("playerIndex") == opponent:
            remember(stadium)
    for card in state.get("looking") or []:
        if isinstance(card, dict) and card.get("playerIndex") == opponent:
            remember(card)
    for field_name in ("contextCard", "effect"):
        card = select.get(field_name)
        if isinstance(card, dict) and card.get("playerIndex") == opponent:
            remember(card)

    # logsは現在zoneから隠しzoneへ戻ったカードもDB照合の証拠として残す。
    for log in observation.get("logs") or []:
        if not isinstance(log, dict) or log.get("playerIndex") != opponent:
            continue
        for card_field, serial_field in (
            ("cardId", "serial"),
            ("cardIdBefore", "serialBefore"),
            ("cardIdAfter", "serialAfter"),
            ("cardIdActive", "serialActive"),
            ("cardIdBench", "serialBench"),
        ):
            card_id = log.get(card_field)
            serial = log.get(serial_field)
            if card_id is not None and serial is not None:
                target[int(serial)] = int(card_id)

    # 型検査と、fixtureで現在公開カードが失われないことを保証する。
    if public_cards and not target:
        raise DeckBeliefError("公開カードを観測履歴へ登録できませんでした。")


@lru_cache(maxsize=1)
def _candidate_records() -> tuple[dict[str, Any], ...]:
    return tuple(load_candidates(DEFAULT_DATABASE))


def _candidate_rank(record: dict[str, Any], index: int) -> tuple[int, int]:
    return (int(record.get("wins", 0)), -index)


def _reconcile_candidate(deck: Sequence[int], observed: Counter[int]) -> list[int]:
    """DB候補を観測カードを必ず含む60枚へ補正する。"""
    counts = Counter(int(card_id) for card_id in deck)
    if sum(counts.values()) != DECK_SIZE:
        raise DeckBeliefError("DB候補が60枚ではありません。")
    if sum(observed.values()) > DECK_SIZE:
        raise DeckBeliefError("観測カードが60枚を超えています。")

    missing = observed - counts
    remove_count = sum(missing.values())
    if remove_count:
        for card_id in reversed(deck):
            card_id = int(card_id)
            if remove_count <= 0:
                break
            if counts[card_id] > observed[card_id]:
                counts[card_id] -= 1
                remove_count -= 1
        if remove_count:
            raise DeckBeliefError("観測カードを含む60枚へDB候補を補正できません。")
        counts.update(missing)
    result = list(counts.elements())
    if len(result) != DECK_SIZE or observed - Counter(result):
        raise DeckBeliefError("補正後のDB候補が観測カードと一致しません。")
    return result


@lru_cache(maxsize=4096)
def _predict_full_deck_cached(observed_key: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    observed = Counter(dict(observed_key))
    records = _candidate_records()

    exact: list[tuple[int, dict[str, Any]]] = []
    best_record: dict[str, Any] | None = None
    best_rank: tuple[int, int, int] | None = None
    for index, record in enumerate(records):
        candidate = Counter(int(card_id) for card_id in record["deck"])
        matched = sum(
            min(count, candidate.get(card_id, 0))
            for card_id, count in observed.items()
        )
        rank = (matched, *_candidate_rank(record, index))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_record = record
        if not (observed - candidate):
            exact.append((index, record))

    if exact:
        _, selected = max(exact, key=lambda item: _candidate_rank(item[1], item[0]))
    else:
        if best_record is None:
            raise DeckBeliefError("デッキ候補DBが空です。")
        selected = best_record
    return tuple(_reconcile_candidate(selected["deck"], observed))


def predict_full_deck(observed_cards: Iterable[int]) -> tuple[int, ...]:
    observed = Counter(int(card_id) for card_id in observed_cards)
    return _predict_full_deck_cached(tuple(sorted(observed.items())))


def _subtract_cards(pool: Counter[int], cards: Iterable[int], label: str) -> None:
    for card_id in cards:
        card_id = int(card_id)
        if pool[card_id] <= 0:
            raise DeckBeliefError(f"{label}のカード{card_id}が60枚デッキにありません。")
        pool[card_id] -= 1
        if pool[card_id] == 0:
            del pool[card_id]


def _fill_card_slots(slots: Any, sampled: Iterable[int]) -> list[int]:
    iterator = iter(sampled)
    result: list[int] = []
    for card in slots or []:
        card_id = _card_id(card)
        result.append(next(iterator) if card_id is None else card_id)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise DeckBeliefError("非公開サイドの割当枚数が一致しません。")


def _take(pool: list[int], count: int, label: str) -> list[int]:
    if count < 0 or len(pool) < count:
        raise DeckBeliefError(f"{label}へ{count}枚を割り当てられません。")
    result = pool[:count]
    del pool[:count]
    return result


def sample_hidden_zones(
    observation: dict[str, Any],
    your_index: int,
    your_full_deck: Sequence[int],
    seen_opponent_cards: Iterable[int],
    rng: random.Random | Any = random,
) -> HiddenZoneSample:
    """SearchBeginに必要な非公開zoneだけをsampleする。"""
    state = observation.get("current")
    if not isinstance(state, dict):
        raise DeckBeliefError("current stateがありません。")
    players = state.get("players") or []
    if len(players) != 2 or your_index not in (0, 1):
        raise DeckBeliefError("player stateが不正です。")
    opponent_index = 1 - your_index

    own_full = tuple(int(card_id) for card_id in your_full_deck)
    if len(own_full) != DECK_SIZE:
        raise DeckBeliefError("自分のデッキが60枚ではありません。")
    opponent_full = predict_full_deck(seen_opponent_cards)
    full_decks_list: list[tuple[int, ...]] = [(), ()]
    full_decks_list[your_index] = own_full
    full_decks_list[opponent_index] = opponent_full
    full_decks = (full_decks_list[0], full_decks_list[1])

    select = observation.get("select")
    raw_public = public_cards_by_player(state, select)
    public = (
        _reconcile_transient_public_cards(
            raw_public[0], state, select, 0
        ),
        _reconcile_transient_public_cards(
            raw_public[1], state, select, 1
        ),
    )

    # 自分の手札はすべて見えており、BattleのObservationをそのまま使う。
    # ここで生成するのはSearchBegin引数になる山札と裏向きサイドだけ。
    own_player = players[your_index]
    own_prize_slots = own_player.get("prize") or []
    own_known_prize = _known_cards(own_prize_slots)
    own_remaining = Counter(own_full)
    _subtract_cards(own_remaining, public[your_index], "自分の公開zone")
    _subtract_cards(own_remaining, own_known_prize, "自分の表向きサイド")
    _subtract_cards(
        own_remaining,
        _known_cards(own_player.get("hand")),
        "自分の手札",
    )
    own_pool = list(own_remaining.elements())
    rng.shuffle(own_pool)
    own_deck_count = int(own_player.get("deckCount", 0))
    own_unknown_prize_count = len(own_prize_slots) - len(own_known_prize)
    if len(own_pool) < own_deck_count + own_unknown_prize_count:
        raise DeckBeliefError(
            "自分の非公開zoneへカードを割り当てられません: "
            f"pool={len(own_pool)}, deck={own_deck_count}, "
            f"unknown_prize={own_unknown_prize_count}, "
            f"public={len(public[your_index])}, "
            f"known_hand={len(_known_cards(own_player.get('hand')))}, "
            f"known_prize={len(own_known_prize)}"
        )
    your_deck = _take(own_pool, own_deck_count, "自分の山札")
    your_prize = _fill_card_slots(
        own_prize_slots,
        _take(own_pool, own_unknown_prize_count, "自分のサイド"),
    )

    # 相手の手札はまだ推定モデルがないため、山札・サイドと同じ残りpoolからsampleする。
    opponent_player = players[opponent_index]
    opponent_prize_slots = opponent_player.get("prize") or []
    opponent_known_prize = _known_cards(opponent_prize_slots)
    opponent_known_hand = _known_cards(opponent_player.get("hand"))
    opponent_remaining = Counter(opponent_full)
    _subtract_cards(opponent_remaining, public[opponent_index], "相手の公開zone")
    _subtract_cards(
        opponent_remaining,
        opponent_known_prize,
        "相手の表向きサイド",
    )
    _subtract_cards(opponent_remaining, opponent_known_hand, "相手の既知手札")
    opponent_pool = list(opponent_remaining.elements())
    rng.shuffle(opponent_pool)
    opponent_deck = _take(
        opponent_pool,
        int(opponent_player.get("deckCount", 0)),
        "相手の山札",
    )
    opponent_unknown_prize_count = len(opponent_prize_slots) - len(
        opponent_known_prize
    )
    opponent_prize = _fill_card_slots(
        opponent_prize_slots,
        _take(opponent_pool, opponent_unknown_prize_count, "相手のサイド"),
    )
    opponent_unknown_hand_count = max(
        0,
        int(opponent_player.get("handCount", 0)) - len(opponent_known_hand),
    )
    opponent_hand = opponent_known_hand + _take(
        opponent_pool,
        opponent_unknown_hand_count,
        "相手の手札",
    )

    return HiddenZoneSample(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        # セットアップ中はMCTSを呼ばないため、裏向きActiveの仮定は不要。
        opponent_active=[],
        full_decks=full_decks,
    )
