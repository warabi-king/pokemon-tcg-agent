from __future__ import annotations

import unittest
from unittest.mock import patch

from app.room_manager import RoomManager, available_decks


class FakeProcess:
    def is_alive(self) -> bool:
        return True


class FakeWorker:
    def __init__(self, _deck_names: list[str]) -> None:
        self.process = FakeProcess()

    def close(self) -> None:
        pass


class RoomLimitTests(unittest.TestCase):
    def test_pin_validation_and_five_room_limit(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=5)
        with patch("app.room_manager.WorkerClient", FakeWorker):
            with self.assertRaisesRegex(ValueError, "4桁"):
                manager.create("123", [deck, deck])
            for index in range(5):
                manager.create(f"{index:04d}", [deck, deck])
            with self.assertRaisesRegex(ValueError, "満室"):
                manager.create("9999", [deck, deck])


class WorkerIntegrationTests(unittest.TestCase):
    def test_two_private_players_can_advance_the_same_match(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=2)
        try:
            room = manager.create("1234", [deck, deck])
            other_room = manager.create("5678", [deck, deck])
            with self.assertRaisesRegex(ValueError, "暗証番号"):
                manager.join(room.room_id, "9999")
            manager.join(room.room_id.lower(), "1234")
            manager.join(other_room.room_id, "5678")

            progressed = 0
            seen_observation = [False, False]
            for _ in range(20):
                states = [manager.state(room.room_id, role) for role in (0, 1)]
                for role, state in enumerate(states):
                    seen_observation[role] |= state.get("observation") is not None
                active_role = next(
                    (role for role, state in enumerate(states) if state["humanTurn"]), None
                )
                if active_role is None:
                    break
                selection = states[active_role]["observation"]["select"]
                indices = list(range(selection["minCount"]))
                manager.action(
                    room.room_id, active_role, indices, states[active_role]["step"]
                )
                progressed += 1

            self.assertGreaterEqual(progressed, 10)
            self.assertEqual(seen_observation, [True, True])
            guest = manager.state(room.room_id, 1)["observation"]["current"]
            self.assertIsNotNone(guest["players"][0]["hand"])
            self.assertIsNone(guest["players"][1]["hand"])

            other_state = manager.state(other_room.room_id, 0)
            self.assertNotEqual(room.room_id, other_room.room_id)
            self.assertEqual(other_state["roomId"], other_room.room_id)
            self.assertTrue(other_state["humanTurn"])
        finally:
            manager.close_all()


if __name__ == "__main__":
    unittest.main()
