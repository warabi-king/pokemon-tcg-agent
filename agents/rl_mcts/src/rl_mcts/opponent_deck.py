"""公開済みカードから相手の60枚デッキを推定するMLPと状態追跡。"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DECK_SIZE = 60
SCALAR_FEATURES = 2  # 公開枚数、ターン
DEFAULT_TURN_SCALE = 20.0


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


@dataclass(frozen=True)
class PublicDeckObservation:
    """ある観測時点までに公開された相手カード。"""

    cards: tuple[int, ...]
    current_public_cards: tuple[int, ...]
    turn: int


@dataclass(frozen=True)
class OpponentHiddenCards:
    """cabtのsearch_beginへ渡す相手の非公開領域の推定値。"""

    deck: list[int]
    prize: list[int]
    hand: list[int]
    active: list[int]
    full_deck: list[int]
    observed_cards: list[int]


class PublicCardTracker:
    """相手が一度でも公開したカードをserial単位で追跡する。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_by_serial: dict[int, int] = {}
        self._last_turn = 0

    @property
    def cards(self) -> tuple[int, ...]:
        return tuple(sorted(self._seen_by_serial.values()))

    def _remember_card(
        self,
        card: Any,
        opponent_index: int,
        current: dict[int, int],
        assumed_owner: int | None = None,
    ) -> None:
        if card is None:
            return
        owner = _field(card, "playerIndex", assumed_owner)
        if owner is not None and int(owner) != opponent_index:
            return

        card_id = _field(card, "id", _field(card, "cardId"))
        serial = _field(card, "serial")
        if card_id is not None and serial is not None:
            card_id = int(card_id)
            serial = int(serial)
            if card_id > 0:
                self._seen_by_serial[serial] = card_id
                current[serial] = card_id

        for name in ("preEvolution", "tools", "energyCards"):
            for child in _items(_field(card, name)):
                self._remember_card(child, opponent_index, current, assumed_owner=owner)

    def update(self, observation: Any) -> PublicDeckObservation:
        """cabtのObservationまたは生dictを取り込み、累積公開カードを返す。"""
        state = _field(observation, "current")
        if state is None:
            return PublicDeckObservation(self.cards, (), self._last_turn)

        turn = int(_field(state, "turn", 0) or 0)
        self._last_turn = turn
        your_index = int(_field(state, "yourIndex", 0))
        opponent_index = 1 - your_index
        players = _items(_field(state, "players"))
        if len(players) < 2:
            return PublicDeckObservation(self.cards, (), turn)

        current: dict[int, int] = {}
        opponent = players[opponent_index]
        for zone_name in ("active", "bench", "discard", "prize"):
            for card in _items(_field(opponent, zone_name)):
                self._remember_card(card, opponent_index, current, assumed_owner=opponent_index)

        for card in _items(_field(state, "stadium")):
            self._remember_card(card, opponent_index, current)
        for card in _items(_field(state, "looking")):
            self._remember_card(card, opponent_index, current)

        select = _field(observation, "select")
        if select is not None:
            for name in ("contextCard", "effect"):
                self._remember_card(_field(select, name), opponent_index, {})
            for card in _items(_field(select, "deck")):
                self._remember_card(card, opponent_index, {})

        # 対戦者向けobservationでは非公開イベントのcardIdが除去されるため、
        # cardIdが実際に残っている相手ログだけを公開情報として扱える。
        for log in _items(_field(observation, "logs")):
            if _field(log, "playerIndex") == opponent_index:
                self._remember_card(log, opponent_index, {}, assumed_owner=opponent_index)

        return PublicDeckObservation(
            cards=self.cards,
            current_public_cards=tuple(sorted(current.values())),
            turn=turn,
        )


class OpponentDeckMLP(torch.nn.Module):
    """公開カードから採用確率と採用時枚数を推定する2ヘッドMLP。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        input_size = vocab_size + SCALAR_FEATURES
        for _ in range(layers):
            modules.append(torch.nn.Linear(input_size, hidden_size))
            modules.append(torch.nn.ReLU())
            if dropout > 0:
                modules.append(torch.nn.Dropout(dropout))
            input_size = hidden_size
        self.body = torch.nn.Sequential(*modules)
        output_size = output_size or vocab_size
        self.presence_head = torch.nn.Linear(input_size, output_size)
        self.count_head = torch.nn.Linear(input_size, output_size)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return self.presence_head(hidden), self.count_head(hidden)


def make_features(
    cards: Sequence[int],
    turn: int,
    vocab_size: int,
    count_scales: torch.Tensor,
    turn_scale: float = DEFAULT_TURN_SCALE,
) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.float32)
    for card_id in cards:
        if 0 <= int(card_id) < vocab_size:
            counts[int(card_id)] += 1.0
    return torch.cat(
        (
            counts / count_scales.cpu().clamp_min(1.0),
            torch.tensor(
                [min(len(cards), DECK_SIZE) / DECK_SIZE, max(0, turn) / turn_scale],
                dtype=torch.float32,
            ),
        )
    )


def complete_deck(
    observed_cards: Sequence[int],
    predicted_counts: torch.Tensor,
    known_card_ids: Sequence[int],
    basic_energy_ids: set[int],
    ace_spec_ids: set[int],
    card_names: Mapping[int, str],
    presence_scores: torch.Tensor | None = None,
    min_unique_cards: int | None = None,
    max_unique_cards: int | None = None,
) -> list[int]:
    """採用確率上位のカードだけを使い、予測枚数に近い60枚へ丸める。"""
    deck = [int(card_id) for card_id in observed_cards[:DECK_SIZE]]
    counts = Counter(deck)
    name_counts = Counter(card_names.get(card_id, str(card_id)) for card_id in deck)
    ace_count = sum(counts[card_id] for card_id in ace_spec_ids)
    scores = predicted_counts.detach().cpu().tolist()
    presence = (presence_scores if presence_scores is not None else predicted_counts).detach().cpu().tolist()

    candidates = [int(card_id) for card_id in known_card_ids if 0 < int(card_id) < len(scores)]
    ranked = sorted(
        candidates,
        key=lambda card_id: (float(presence[card_id]), float(scores[card_id]), -card_id),
        reverse=True,
    )
    if max_unique_cards is not None:
        observed_ids = set(deck)
        predicted_unique = sum(float(presence[card_id]) >= 0.5 for card_id in candidates)
        minimum = int(min_unique_cards or 1)
        unique_limit = max(minimum, predicted_unique, len(observed_ids), 1)
        unique_limit = min(unique_limit, max(int(max_unique_cards), len(observed_ids)))
        selected = set(observed_ids)
        for card_id in ranked:
            if len(selected) >= unique_limit:
                break
            selected.add(card_id)

        # 60枚まで確実に埋められるよう、基本エネルギーを最低1種類残す。
        energy = next((card_id for card_id in ranked if card_id in basic_energy_ids), None)
        if energy is not None and energy not in selected:
            removable = next(
                (card_id for card_id in reversed(ranked) if card_id in selected and card_id not in observed_ids),
                None,
            )
            if removable is not None and len(selected) >= unique_limit:
                selected.remove(removable)
            selected.add(energy)
        candidates = [card_id for card_id in ranked if card_id in selected]

    while len(deck) < DECK_SIZE:
        best_card: int | None = None
        best_score = float("-inf")
        for card_id in candidates:
            name = card_names.get(card_id, str(card_id))
            if card_id in ace_spec_ids and ace_count >= 1:
                continue
            if card_id not in basic_energy_ids and name_counts[name] >= 4:
                continue
            residual = float(scores[card_id]) - counts[card_id]
            if residual > best_score:
                best_card = card_id
                best_score = residual

        if best_card is None:
            best_card = min(basic_energy_ids) if basic_energy_ids else (candidates[0] if candidates else 1)
        deck.append(best_card)
        counts[best_card] += 1
        name_counts[card_names.get(best_card, str(best_card))] += 1
        if best_card in ace_spec_ids:
            ace_count += 1
    return deck


def _subtract_cards(deck: Sequence[int], cards: Sequence[int]) -> list[int]:
    remaining = Counter(int(card_id) for card_id in deck)
    for card_id in cards:
        if remaining[int(card_id)] > 0:
            remaining[int(card_id)] -= 1
    return [card_id for card_id, count in remaining.items() for _ in range(count)]


def _take_matching(cards: list[int], wanted: set[int]) -> int | None:
    for index, card_id in enumerate(cards):
        if card_id in wanted:
            return cards.pop(index)
    return None


class OpponentDeckPredictor:
    """checkpointの遅延読込、公開情報追跡、非公開領域への割当を行う。"""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.tracker = PublicCardTracker()
        self.model: OpponentDeckMLP | None = None
        self.config: dict[str, Any] | None = None
        self.count_scales: torch.Tensor | None = None

    @property
    def available(self) -> bool:
        return self.checkpoint_path.exists()

    def reset(self) -> None:
        self.tracker.reset()

    def _load(self) -> None:
        if self.model is not None:
            return
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        config = dict(checkpoint["config"])
        model = OpponentDeckMLP(
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
            output_size=int(config.get("output_size", len(config["known_card_ids"]))),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        self.model = model
        self.config = config
        self.count_scales = torch.tensor(config["count_scales"], dtype=torch.float32)

    def predict(self, observation: Any, rng: random.Random | None = None) -> OpponentHiddenCards:
        self._load()
        assert self.model is not None and self.config is not None and self.count_scales is not None
        public = self.tracker.update(observation)
        config = self.config
        vocab_size = int(config["vocab_size"])
        features = make_features(
            public.cards,
            public.turn,
            vocab_size,
            self.count_scales,
            float(config.get("turn_scale", DEFAULT_TURN_SCALE)),
        )
        known_card_ids = [int(card_id) for card_id in config["known_card_ids"]]
        candidate_scales = self.count_scales[known_card_ids]
        with torch.inference_mode():
            presence_logits, normalized_counts = self.model(features.unsqueeze(0))
            compact_presence = torch.sigmoid(presence_logits[0])
            compact_counts = torch.nn.functional.softplus(normalized_counts[0]) * candidate_scales

        predicted = torch.zeros(vocab_size, dtype=torch.float32)
        presence = torch.zeros(vocab_size, dtype=torch.float32)
        predicted[known_card_ids] = compact_counts
        presence[known_card_ids] = compact_presence

        basic_energy_ids = {int(card_id) for card_id in config.get("basic_energy_ids", [])}
        ace_spec_ids = {int(card_id) for card_id in config.get("ace_spec_ids", [])}
        basic_pokemon_ids = {int(card_id) for card_id in config.get("basic_pokemon_ids", [])}
        card_names = {int(card_id): str(name) for card_id, name in config.get("card_names", {}).items()}
        full_deck = complete_deck(
            public.cards,
            predicted,
            known_card_ids,
            basic_energy_ids,
            ace_spec_ids,
            card_names,
            presence_scores=presence,
            min_unique_cards=int(config.get("min_unique_cards", 10)),
            max_unique_cards=int(config.get("max_unique_cards", 30)),
        )

        state = _field(observation, "current")
        your_index = int(_field(state, "yourIndex", 0))
        opponent = _items(_field(state, "players"))[1 - your_index]
        deck_count = int(_field(opponent, "deckCount", 0))
        prize_count = len(_items(_field(opponent, "prize")))
        hand_count = int(_field(opponent, "handCount", 0))
        hidden_active_count = sum(card is None for card in _items(_field(opponent, "active")))

        pool = _subtract_cards(full_deck, public.current_public_cards)
        (rng or random).shuffle(pool)
        active: list[int] = []
        for _ in range(hidden_active_count):
            card_id = _take_matching(pool, basic_pokemon_ids)
            active.append(card_id if card_id is not None else (min(basic_pokemon_ids) if basic_pokemon_ids else 1072))

        # セットアップ時のSDK制約に備え、山札にもたねポケモンを最低1枚残す。
        if deck_count > 0 and basic_pokemon_ids and not any(card_id in basic_pokemon_ids for card_id in pool[:deck_count]):
            basic_index = next((i for i, card_id in enumerate(pool[deck_count:]) if card_id in basic_pokemon_ids), None)
            if basic_index is not None and pool:
                swap_index = deck_count + basic_index
                pool[0], pool[swap_index] = pool[swap_index], pool[0]

        fallback = min(basic_pokemon_ids) if basic_pokemon_ids else 1072
        required = deck_count + prize_count + hand_count
        if len(pool) < required:
            pool.extend([fallback] * (required - len(pool)))
        deck = pool[:deck_count]
        prize = pool[deck_count : deck_count + prize_count]
        hand = pool[deck_count + prize_count : required]
        return OpponentHiddenCards(
            deck=deck,
            prize=prize,
            hand=hand,
            active=active,
            full_deck=full_deck,
            observed_cards=list(public.cards),
        )
