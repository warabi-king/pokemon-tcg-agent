from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAIN_ROOT))

from imitation_data import (  # noqa: E402
    MINIMAL_EPISODE_FORMAT,
    extract_samples_from_episode,
    minimize_episode,
)


def card(card_id: int, player: int) -> dict:
    return {"id": card_id, "serial": card_id, "playerIndex": player}


def pokemon(card_id: int, player: int) -> dict:
    return {
        "id": card_id,
        "serial": card_id,
        "hp": 100,
        "maxHp": 100,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [card(3, player)],
        "tools": [card(4, player)],
        "preEvolution": [],
    }


def player_state(player: int, hand_visible: bool) -> dict:
    hand = [card(5 + player, player)] if hand_visible else None
    return {
        "active": [pokemon(1 + player, player)],
        "bench": [pokemon(10 + player, player)],
        "benchMax": 5,
        "deckCount": 52,
        "discard": [card(7 + player, player)],
        "prize": [card(12 + player, player), None, None, None, None, None],
        "handCount": 1,
        "hand": hand,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def observation(viewpoint: int) -> dict:
    options = [
        {"type": 0, "number": 2},
        {"type": 1},
        {"type": 2},
    ]
    options.extend(
        {"type": 3, "area": area, "index": 0, "playerIndex": viewpoint}
        for area in (1, 2, 3, 4, 5, 6, 7, 12)
    )
    options.extend(
        [
            {"type": 4, "area": 4, "index": 0, "playerIndex": viewpoint, "toolIndex": 0},
            {"type": 5, "area": 4, "index": 0, "playerIndex": viewpoint, "energyIndex": 0},
            {
                "type": 6,
                "area": 4,
                "index": 0,
                "playerIndex": viewpoint,
                "energyIndex": 0,
                "count": 2,
            },
            {"type": 7, "index": 0},
            {"type": 8, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 9, "area": 2, "index": 0, "inPlayArea": 4, "inPlayIndex": 0},
            {"type": 10, "area": 4, "index": 0},
            {"type": 11, "area": 4, "index": 0},
            {"type": 12},
            {"type": 13, "attackId": 1},
            {"type": 14},
            {"type": 15, "cardId": 15, "serial": 99},
            {"type": 16, "specialConditionType": 0},
        ]
    )
    return {
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": options,
            "deck": [card(14, viewpoint)],
            "contextCard": None,
            "effect": None,
        },
        "logs": [{"type": 2, "playerIndex": viewpoint}],
        "current": {
            "turn": 3,
            "turnActionCount": 4,
            "yourIndex": viewpoint,
            "firstPlayer": 0,
            "supporterPlayed": True,
            "stadiumPlayed": True,
            "energyAttached": True,
            "retreated": True,
            "result": -1,
            "stadium": [card(9, 0)],
            "looking": [card(16, viewpoint)],
            "players": [
                player_state(0, viewpoint == 0),
                player_state(1, viewpoint == 1),
            ],
        },
        "search_begin_input": "unused",
        "remainingOverageTime": 42,
    }


def raw_episode() -> bytes:
    deck0 = [1] * 60
    deck1 = [2] * 60
    no_selection = {"select": None}
    episode = {
        "configuration": {"unused": True},
        "rewards": [1, -1],
        "steps": [
            [
                {"action": deck0, "observation": no_selection},
                {"action": deck1, "observation": no_selection},
            ],
            [
                {"action": deck0, "observation": no_selection},
                {"action": deck1, "observation": no_selection},
            ],
            [
                {"action": None, "observation": observation(0)},
                {"action": None, "observation": observation(1)},
            ],
            [
                {"action": [0], "observation": no_selection},
                {"action": [0], "observation": no_selection},
            ],
        ],
    }
    return json.dumps(episode).encode()


class MinimalImitationEpisodeTest(unittest.TestCase):
    def test_minimal_json_produces_identical_training_samples(self) -> None:
        raw = raw_episode()
        minimal = minimize_episode(raw)
        self.assertIsNotNone(minimal)

        encoded = json.dumps(minimal, separators=(",", ":")).encode()
        self.assertEqual(
            extract_samples_from_episode(raw),
            extract_samples_from_episode(encoded),
        )

    def test_minimal_json_contains_only_training_schema(self) -> None:
        minimal = minimize_episode(raw_episode())
        self.assertEqual(minimal["format"], MINIMAL_EPISODE_FORMAT)
        self.assertEqual(set(minimal), {"format", "rewards", "decks", "decisions"})
        self.assertEqual(len(minimal["decisions"]), 2)

        decision = minimal["decisions"][0]
        self.assertEqual(set(decision), {"player", "observation", "chosenIndex"})
        self.assertNotIn("logs", decision["observation"])
        self.assertNotIn("action", decision)
        self.assertNotIn("steps", minimal)


if __name__ == "__main__":
    unittest.main()
