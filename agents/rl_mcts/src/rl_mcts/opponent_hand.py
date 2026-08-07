"""Opponent hidden-card belief and learned hand-retention scoring."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from cg.api import AreaType, Card, LogType, Observation, PlayerState, Pokemon, all_card_data
from rl_mcts.deck_reconstructor import DeckMatch, load_candidates, rank_matching_decks
from rl_mcts.model import card_count


HAND_SCALAR_SIZE = 10
DEFAULT_HAND_MODEL = Path(__file__).with_name("hand_model.pth")


@dataclass(frozen=True)
class HiddenCards:
    """Concrete hidden cards supplied to one simulator determinization."""

    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


class HandScoreModel(torch.nn.Module):
    """Predict per-card residual hand-retention scores.

    Physical availability is applied outside the model. A zero-score model therefore
    reduces exactly to sampling from the remaining deck multiset.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 96) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        input_size = 2 * vocab_size + HAND_SCALAR_SIZE
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _card_identity(card: Card | Pokemon) -> tuple[int, int]:
    return int(card.serial), int(card.id)


def pokemon_cards(pokemon: Pokemon | None) -> list[Card | Pokemon]:
    if pokemon is None:
        return []
    return [
        pokemon,
        *pokemon.preEvolution,
        *pokemon.tools,
        *pokemon.energyCards,
    ]


def visible_player_cards(player: PlayerState) -> list[Card | Pokemon]:
    cards: list[Card | Pokemon] = list(player.discard)
    for pokemon in player.active:
        cards.extend(pokemon_cards(pokemon))
    for pokemon in player.bench:
        cards.extend(pokemon_cards(pokemon))
    cards.extend(card for card in player.prize if card is not None)
    return cards


def current_public_counter(obs: Observation, player_index: int) -> Counter[int]:
    cards = visible_player_cards(obs.current.players[player_index])
    cards.extend(card for card in obs.current.stadium if card.playerIndex == player_index)
    return Counter(int(card.id) for card in cards)


def make_hand_features(
    deck_counts: Counter[int],
    public_counts: Counter[int],
    obs: Observation,
    target_index: int,
    vocab_size: int,
) -> torch.Tensor:
    """Build the dense input shared by training and submission inference."""

    features = torch.zeros(2 * vocab_size + HAND_SCALAR_SIZE, dtype=torch.float32)
    for card_id, count in deck_counts.items():
        if 0 <= card_id < vocab_size:
            features[card_id] = min(float(count) / 4.0, 3.0)
    for card_id, count in public_counts.items():
        if 0 <= card_id < vocab_size:
            features[vocab_size + card_id] = min(float(count) / 4.0, 3.0)

    state = obs.current
    player = state.players[target_index]
    scalar = 2 * vocab_size
    values = (
        min(state.turn / 20.0, 2.0),
        min(player.handCount / 15.0, 2.0),
        player.deckCount / 60.0,
        len(player.prize) / 6.0,
        min(len(player.discard) / 30.0, 2.0),
        len(player.bench) / 5.0,
        float(state.supporterPlayed),
        float(state.energyAttached),
        float(state.stadiumPlayed),
        float(state.retreated),
    )
    features[scalar : scalar + HAND_SCALAR_SIZE] = torch.tensor(values)
    return features


class HandModelPredictor:
    def __init__(self, path: Path = DEFAULT_HAND_MODEL) -> None:
        self.path = path
        self.model: HandScoreModel | None = None
        self.vocab_size = card_count()
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.vocab_size = int(checkpoint["vocab_size"])
            model = HandScoreModel(self.vocab_size, int(checkpoint["hidden_size"]))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.model = model

    def scores(
        self,
        deck_counts: Counter[int],
        public_counts: Counter[int],
        obs: Observation,
        target_index: int,
    ) -> list[float]:
        if self.model is None:
            return [0.0] * self.vocab_size
        features = make_hand_features(
            deck_counts, public_counts, obs, target_index, self.vocab_size
        )
        with torch.inference_mode():
            values = self.model(features.unsqueeze(0))[0].clamp(-8.0, 8.0)
        return values.tolist()


class OpponentHandTracker:
    """Track cards that are guaranteed to remain in the opponent's hand."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.opponent_index: int | None = None
        self.known_hand_by_serial: dict[int, int] = {}
        self.seen_by_serial: dict[int, int] = {}

    def _remember(self, serial: int | None, card_id: int | None) -> None:
        if serial is not None and card_id is not None:
            self.seen_by_serial[int(serial)] = int(card_id)

    def update(self, obs: Observation) -> None:
        your_index = obs.current.yourIndex
        opponent_index = 1 - your_index
        if self.opponent_index is not None and self.opponent_index != opponent_index:
            self.reset()
        self.opponent_index = opponent_index

        for card in visible_player_cards(obs.current.players[opponent_index]):
            serial, card_id = _card_identity(card)
            self.seen_by_serial[serial] = card_id
        for card in obs.current.stadium:
            if card.playerIndex == opponent_index:
                self.seen_by_serial[int(card.serial)] = int(card.id)

        for log in obs.logs:
            if log.playerIndex != opponent_index:
                continue

            if log.type in {
                LogType.MOVE_CARD,
                LogType.PLAY,
                LogType.ATTACH,
                LogType.EVOLVE,
                LogType.DEVOLVE,
                LogType.MOVE_ATTACHED,
                LogType.ATTACK,
            }:
                self._remember(log.serial, log.cardId)

            if log.type == LogType.MOVE_CARD:
                if log.toArea == AreaType.HAND and log.cardId is not None and log.serial is not None:
                    self.known_hand_by_serial[int(log.serial)] = int(log.cardId)
                if log.fromArea == AreaType.HAND and log.serial is not None:
                    self.known_hand_by_serial.pop(int(log.serial), None)
            elif log.type == LogType.MOVE_CARD_REVERSE and log.fromArea == AreaType.HAND:
                # The moved serial is hidden. None of the previous exact identities is
                # guaranteed to have remained, so discard the exact-hand belief.
                self.known_hand_by_serial.clear()
            elif log.type in {LogType.PLAY, LogType.ATTACH, LogType.EVOLVE}:
                if log.serial is not None:
                    self.known_hand_by_serial.pop(int(log.serial), None)

        hand_count = obs.current.players[opponent_index].handCount
        if hand_count == 0:
            self.known_hand_by_serial.clear()
        elif len(self.known_hand_by_serial) > hand_count:
            self.known_hand_by_serial.clear()

    def known_hand(self) -> list[int]:
        return list(self.known_hand_by_serial.values())

    def observed_cards(self) -> list[int]:
        return list(self.seen_by_serial.values())


def reconcile_deck(candidate: Iterable[int], required: Counter[int]) -> list[int]:
    """Replace unmatched candidate cards so every observed physical card is present."""

    counts = Counter(int(card_id) for card_id in candidate)
    missing = Counter(
        {card_id: count - counts.get(card_id, 0) for card_id, count in required.items()}
    )
    missing = +missing
    if not missing:
        return list(counts.elements())

    removable = counts - required
    for card_id in list(missing.elements()):
        if not removable:
            break
        remove_id = max(removable, key=lambda value: removable[value])
        counts[remove_id] -= 1
        removable[remove_id] -= 1
        if removable[remove_id] <= 0:
            del removable[remove_id]
        counts[card_id] += 1
    return list(counts.elements())


def subtract_cards(pool: Counter[int], cards: Iterable[int]) -> Counter[int]:
    result = pool.copy()
    for card_id in cards:
        if result[card_id] > 0:
            result[card_id] -= 1
            if result[card_id] <= 0:
                del result[card_id]
    return result


def weighted_sample_without_replacement(
    pool: Counter[int], count: int, scores: list[float], rng: random.Random
) -> list[int]:
    result: list[int] = []
    available = pool.copy()
    for _ in range(min(count, sum(available.values()))):
        ids = list(available)
        weights = [
            available[card_id]
            * math.exp(scores[card_id] if 0 <= card_id < len(scores) else 0.0)
            for card_id in ids
        ]
        selected = rng.choices(ids, weights=weights, k=1)[0]
        result.append(selected)
        available[selected] -= 1
        if available[selected] <= 0:
            del available[selected]
    return result


class OpponentBelief:
    """Create consistent hidden-state particles for imperfect-information MCTS."""

    def __init__(self, seed: int | None = None, candidate_limit: int = 24) -> None:
        self.rng = random.Random(seed)
        self.candidate_limit = candidate_limit
        self.records = load_candidates()
        self.tracker = OpponentHandTracker()
        self.hand_predictor = HandModelPredictor()
        self.basic_pokemon = {
            int(card.cardId) for card in all_card_data() if bool(card.basic)
        }

    def reset(self) -> None:
        self.tracker.reset()

    def observe(self, obs: Observation) -> None:
        """Consume public logs even when the current prompt needs no search."""

        self.tracker.update(obs)

    def _candidate_matches(self) -> list[DeckMatch]:
        return rank_matching_decks(
            self.tracker.observed_cards(), self.records, limit=self.candidate_limit
        )

    def _sample_match(self, matches: list[DeckMatch]) -> DeckMatch:
        best_missing = matches[0].missing_cards
        viable = [match for match in matches if match.missing_cards == best_missing]
        weights = [max(1.0, float(match.record.get("games", 1))) ** 0.5 for match in viable]
        return self.rng.choices(viable, weights=weights, k=1)[0]

    def _sample_opponent(
        self, obs: Observation, match: DeckMatch
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        opponent_index = 1 - obs.current.yourIndex
        player = obs.current.players[opponent_index]
        public = current_public_counter(obs, opponent_index)
        required = Counter(self.tracker.observed_cards())
        required |= public
        required |= Counter(self.tracker.known_hand())
        deck = reconcile_deck(match.deck, required)
        pool = subtract_cards(Counter(deck), public.elements())

        known_hand = self.tracker.known_hand()[: player.handCount]
        pool = subtract_cards(pool, known_hand)

        opponent_active: list[int] = []
        if player.active and player.active[0] is None:
            basics = Counter(
                {card_id: count for card_id, count in pool.items() if card_id in self.basic_pokemon}
            )
            if basics:
                active_id = weighted_sample_without_replacement(
                    basics, 1, [0.0] * self.hand_predictor.vocab_size, self.rng
                )[0]
            else:
                active_id = 1072
            opponent_active = [active_id]
            pool = subtract_cards(pool, opponent_active)

        scores = self.hand_predictor.scores(
            Counter(deck), public, obs, opponent_index
        )
        unknown_count = max(0, player.handCount - len(known_hand))
        unknown_hand = weighted_sample_without_replacement(pool, unknown_count, scores, self.rng)
        pool = subtract_cards(pool, unknown_hand)
        opponent_hand = known_hand + unknown_hand

        prize: list[int | None] = [card.id if card is not None else None for card in player.prize]
        unknown_prize_count = sum(card_id is None for card_id in prize)
        sampled_prize = weighted_sample_without_replacement(
            pool, unknown_prize_count, [0.0] * self.hand_predictor.vocab_size, self.rng
        )
        pool = subtract_cards(pool, sampled_prize)
        sampled_iter = iter(sampled_prize)
        opponent_prize = [
            int(card_id) if card_id is not None else next(sampled_iter, 1) for card_id in prize
        ]

        opponent_deck = list(pool.elements())
        self.rng.shuffle(opponent_deck)
        if len(opponent_deck) < player.deckCount:
            opponent_deck.extend([1072] * (player.deckCount - len(opponent_deck)))
        return (
            opponent_deck[: player.deckCount],
            opponent_prize,
            opponent_hand,
            opponent_active,
        )

    def _sample_your_hidden(
        self, obs: Observation, your_deck: list[int]
    ) -> tuple[list[int], list[int]]:
        your_index = obs.current.yourIndex
        player = obs.current.players[your_index]
        public = current_public_counter(obs, your_index)
        hand = [card.id for card in (player.hand or [])]
        pool = subtract_cards(Counter(your_deck), public.elements())
        pool = subtract_cards(pool, hand)

        prize: list[int | None] = [card.id if card is not None else None for card in player.prize]
        sampled = weighted_sample_without_replacement(
            pool,
            sum(card_id is None for card_id in prize),
            [0.0] * self.hand_predictor.vocab_size,
            self.rng,
        )
        pool = subtract_cards(pool, sampled)
        sampled_iter = iter(sampled)
        your_prize = [
            int(card_id) if card_id is not None else next(sampled_iter, 1) for card_id in prize
        ]
        deck = list(pool.elements())
        self.rng.shuffle(deck)
        if len(deck) < player.deckCount:
            deck.extend([1] * (player.deckCount - len(deck)))
        return deck[: player.deckCount], your_prize

    def sample(
        self, obs: Observation, your_deck: list[int], count: int
    ) -> list[HiddenCards]:
        self.observe(obs)
        matches = self._candidate_matches()
        particles: list[HiddenCards] = []
        for _ in range(max(1, count)):
            match = self._sample_match(matches)
            opponent_deck, opponent_prize, opponent_hand, opponent_active = (
                self._sample_opponent(obs, match)
            )
            own_deck, own_prize = self._sample_your_hidden(obs, your_deck)
            particles.append(
                HiddenCards(
                    your_deck=own_deck,
                    your_prize=own_prize,
                    opponent_deck=opponent_deck,
                    opponent_prize=opponent_prize,
                    opponent_hand=opponent_hand,
                    opponent_active=opponent_active,
                )
            )
        return particles
