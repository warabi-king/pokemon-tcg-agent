"""Focused tests for opponent hidden-card reconstruction."""

from __future__ import annotations

import random
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from cg.api import AreaType, LogType  # noqa: E402
from rl_mcts.deck_reconstructor import rank_matching_decks  # noqa: E402
from rl_mcts.opponent_hand import (  # noqa: E402
    DEFAULT_HAND_MODEL,
    HandModelPredictor,
    OpponentHandTracker,
    reconcile_deck,
    weighted_sample_without_replacement,
)


def player(hand_count: int = 0):
    return SimpleNamespace(
        active=[],
        bench=[],
        discard=[],
        prize=[],
        handCount=hand_count,
        deckCount=60 - hand_count,
    )


def observation(logs, hand_count: int = 1):
    return SimpleNamespace(
        current=SimpleNamespace(
            yourIndex=0,
            players=[player(), player(hand_count)],
            stadium=[],
            turn=3,
            supporterPlayed=False,
            energyAttached=False,
            stadiumPlayed=False,
            retreated=False,
        ),
        logs=logs,
    )


class DeckReconstructorTest(unittest.TestCase):
    def test_rank_prefers_consistency_then_population(self) -> None:
        records = [
            {"deck": [10] + [1] * 59, "games": 2, "wins": 2},
            {"deck": [10, 10] + [2] * 58, "games": 20, "wins": 1},
            {"deck": [3] * 60, "games": 1000, "wins": 1000},
        ]
        ranked = rank_matching_decks([10, 10], records)
        self.assertEqual(ranked[0].candidate_index, 1)
        self.assertEqual(ranked[-1].missing_cards, 2)

    def test_reconcile_includes_observed_multiset(self) -> None:
        repaired = reconcile_deck([1] * 60, Counter({9: 2, 1: 1}))
        self.assertEqual(len(repaired), 60)
        self.assertGreaterEqual(Counter(repaired)[9], 2)


class HandTrackerTest(unittest.TestCase):
    def test_public_move_to_hand_then_play(self) -> None:
        tracker = OpponentHandTracker()
        tracker.update(
            observation(
                [
                    SimpleNamespace(
                        playerIndex=1,
                        type=LogType.MOVE_CARD,
                        serial=42,
                        cardId=741,
                        fromArea=AreaType.DECK,
                        toArea=AreaType.HAND,
                    )
                ]
            )
        )
        self.assertEqual(tracker.known_hand(), [741])
        tracker.update(
            observation(
                [
                    SimpleNamespace(
                        playerIndex=1,
                        type=LogType.PLAY,
                        serial=42,
                        cardId=741,
                        fromArea=None,
                        toArea=None,
                    )
                ],
                hand_count=0,
            )
        )
        self.assertEqual(tracker.known_hand(), [])
        self.assertIn(741, tracker.observed_cards())

    def test_hidden_hand_move_clears_exact_belief(self) -> None:
        tracker = OpponentHandTracker()
        tracker.known_hand_by_serial[4] = 100
        tracker.opponent_index = 1
        tracker.update(
            observation(
                [
                    SimpleNamespace(
                        playerIndex=1,
                        type=LogType.MOVE_CARD_REVERSE,
                        serial=None,
                        cardId=None,
                        fromArea=AreaType.HAND,
                        toArea=AreaType.DECK,
                    )
                ]
            )
        )
        self.assertEqual(tracker.known_hand(), [])


class SamplingTest(unittest.TestCase):
    def test_sampling_never_exceeds_available_copies(self) -> None:
        pool = Counter({5: 2, 7: 1})
        scores = [0.0] * 8
        scores[5] = 5.0
        sample = weighted_sample_without_replacement(pool, 3, scores, random.Random(1))
        self.assertEqual(len(sample), 3)
        self.assertLessEqual(Counter(sample)[5], 2)
        self.assertEqual(Counter(sample), pool)


class HandModelTest(unittest.TestCase):
    def test_feat_hands_generator_checkpoint_loads(self) -> None:
        predictor = HandModelPredictor(DEFAULT_HAND_MODEL, enabled=True)
        self.assertIsNotNone(predictor.model)
        scores = predictor.scores(
            Counter({3: 4, 17: 4}),
            Counter({3: 1}),
            observation([], hand_count=4),
            1,
        )
        self.assertTrue(any(abs(score) > 1e-6 for score in scores))

    def test_db_only_baseline_returns_zero_scores(self) -> None:
        predictor = HandModelPredictor(DEFAULT_HAND_MODEL, enabled=False)
        scores = predictor.scores(Counter(), Counter(), observation([]), 1)
        self.assertEqual(len(scores), predictor.vocab_size)
        self.assertFalse(any(scores))


if __name__ == "__main__":
    unittest.main()
