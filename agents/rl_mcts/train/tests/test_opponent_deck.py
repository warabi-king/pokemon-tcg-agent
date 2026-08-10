from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


TEST_PATH = Path(__file__).resolve()
AGENT_ROOT = TEST_PATH.parents[2]
SRC_ROOT = AGENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_mcts.opponent_deck import (  # noqa: E402
    OpponentDeckMLP,
    PublicCardTracker,
    complete_deck,
    make_features,
)

TRAIN_PATH = AGENT_ROOT / "train" / "train_opponent_deck.py"
SPEC = importlib.util.spec_from_file_location("train_opponent_deck", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAIN_MODULE
SPEC.loader.exec_module(TRAIN_MODULE)


def card(card_id: int, serial: int, player: int, **extra):
    return {"id": card_id, "serial": serial, "playerIndex": player, **extra}


def observation(turn: int, opponent_active=None, opponent_discard=None, logs=None):
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "players": [
                {"active": [], "bench": [], "discard": [], "prize": [], "hand": []},
                {
                    "active": opponent_active or [],
                    "bench": [],
                    "discard": opponent_discard or [],
                    "prize": [None] * 6,
                    "hand": None,
                },
            ],
            "stadium": [],
            "looking": None,
        },
        "logs": logs or [],
        "select": None,
    }


class PublicCardTrackerTest(unittest.TestCase):
    def test_tracks_nested_and_logged_public_cards_once_by_serial(self) -> None:
        tracker = PublicCardTracker()
        active = card(
            100,
            10,
            1,
            preEvolution=[card(90, 9, 1)],
            tools=[card(200, 20, 1)],
            energyCards=[card(3, 30, 1)],
        )
        public = tracker.update(
            observation(
                3,
                opponent_active=[active],
                opponent_discard=[card(300, 40, 1)],
                logs=[
                    {"cardId": 400, "serial": 50, "playerIndex": 1},
                    {"serial": 51, "playerIndex": 1},
                    {"cardId": 999, "serial": 52, "playerIndex": 0},
                ],
            )
        )
        self.assertEqual(Counter(public.cards), Counter([3, 90, 100, 200, 300, 400]))
        self.assertEqual(Counter(public.current_public_cards), Counter([3, 90, 100, 200, 300]))

        public = tracker.update(observation(4, opponent_discard=[card(100, 10, 1)]))
        self.assertEqual(Counter(public.cards), Counter([3, 90, 100, 200, 300, 400]))
        self.assertEqual(public.current_public_cards, (100,))

    def test_tracks_public_selection_effect_owned_by_opponent(self) -> None:
        tracker = PublicCardTracker()
        value = observation(2)
        value["select"] = {"effect": card(777, 70, 1), "contextCard": card(888, 80, 0)}
        public = tracker.update(value)
        self.assertEqual(public.cards, (777,))
        self.assertEqual(public.current_public_cards, ())


class EpisodeSamplesTest(unittest.TestCase):
    def test_uses_opponents_submitted_deck_and_last_observation_per_turn(self) -> None:
        deck0 = [1] * 60
        deck1 = [2] * 60
        episode = {
            "steps": [
                [{"action": deck0, "observation": {"current": None}}, {"action": deck1, "observation": {"current": None}}],
                [
                    {"action": [], "observation": observation(1, opponent_active=[card(10, 10, 1)])},
                    {"action": [], "observation": {"current": None}},
                ],
                [
                    {"action": [], "observation": observation(1, opponent_active=[card(10, 10, 1)], opponent_discard=[card(20, 20, 1)])},
                    {"action": [], "observation": {"current": None}},
                ],
                [
                    {"action": [], "observation": observation(2, opponent_discard=[card(20, 20, 1)])},
                    {"action": [], "observation": {"current": None}},
                ],
            ]
        }
        samples = list(TRAIN_MODULE.iter_episode_samples(episode, "date/episode.json"))
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].turn, 1)
        self.assertEqual(list(samples[0].observed), [10, 20])
        self.assertEqual(list(samples[0].deck), deck1)
        self.assertEqual(samples[1].turn, 2)
        self.assertEqual(list(samples[1].observed), [10, 20])

    def test_default_ratio_selects_about_one_tenth_deterministically(self) -> None:
        keys = [f"2026-07-{index % 31 + 1:02d}/{index}.json" for index in range(10_000)]
        first = [key for key in keys if TRAIN_MODULE.is_selected_episode(key, 0.1)]
        second = [key for key in keys if TRAIN_MODULE.is_selected_episode(key, 0.1)]
        self.assertEqual(first, second)
        self.assertGreater(len(first), 900)
        self.assertLess(len(first), 1100)
        self.assertTrue(all(TRAIN_MODULE.is_selected_episode(key, 1.0) for key in keys))


class ModelAndCompletionTest(unittest.TestCase):
    def test_model_feature_and_output_shapes(self) -> None:
        scales = torch.full((32,), 4.0)
        features = make_features([3, 3, 10], 5, 32, scales)
        model = OpponentDeckMLP(32, hidden_size=16, layers=2, dropout=0, output_size=12)
        presence, counts = model(features.unsqueeze(0))
        self.assertEqual(features.shape, (34,))
        self.assertEqual(presence.shape, (1, 12))
        self.assertEqual(counts.shape, (1, 12))

    def test_completion_preserves_observed_and_copy_rules(self) -> None:
        prediction = torch.zeros(16)
        prediction[10] = 20
        prediction[3] = 10
        deck = complete_deck(
            [10, 10],
            prediction,
            [3, 10],
            basic_energy_ids={3},
            ace_spec_ids=set(),
            card_names={3: "Water Energy", 10: "Pokemon"},
        )
        self.assertEqual(len(deck), 60)
        self.assertGreaterEqual(deck.count(10), 2)
        self.assertLessEqual(deck.count(10), 4)
        self.assertEqual(deck.count(3), 56)

    def test_diffuse_sub_one_predictions_do_not_create_sixty_singletons(self) -> None:
        prediction = torch.full((100,), 0.2)
        presence = torch.full((100,), 0.5)
        known_card_ids = [3, *range(10, 80)]
        deck = complete_deck(
            [],
            prediction,
            known_card_ids,
            basic_energy_ids={3},
            ace_spec_ids=set(),
            card_names={card_id: str(card_id) for card_id in known_card_ids},
            presence_scores=presence,
            max_unique_cards=20,
        )
        counts = Counter(deck)
        self.assertEqual(len(deck), 60)
        self.assertLessEqual(len(counts), 20)
        self.assertEqual(sum(count == 1 for count in counts.values()), 0)


if __name__ == "__main__":
    unittest.main()
