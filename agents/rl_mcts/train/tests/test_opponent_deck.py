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


class ModelAndCompletionTest(unittest.TestCase):
    def test_model_feature_and_output_shapes(self) -> None:
        scales = torch.full((32,), 4.0)
        features = make_features([3, 3, 10], 5, 32, scales)
        model = OpponentDeckMLP(32, hidden_size=16, layers=2, dropout=0)
        self.assertEqual(features.shape, (34,))
        self.assertEqual(model(features.unsqueeze(0)).shape, (1, 32))

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


if __name__ == "__main__":
    unittest.main()
