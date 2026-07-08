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

    def request(self, _payload):
        return {"active": True, "step": 0}


class FinishingFakeWorker(FakeWorker):
    def __init__(self, _deck_names: list[str]) -> None:
        super().__init__(_deck_names)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def request(self, payload):
        return {"active": False, "finished": True, "role": payload.get("role"), "step": 1}


class RoomLimitTests(unittest.TestCase):
    def test_five_room_limit_and_room_id_length(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=5)
        with patch("app.room_manager.WorkerClient", FakeWorker):
            room_ids = []
            for index in range(5):
                room_ids.append(manager.create([deck, deck]).room_id)
            self.assertTrue(all(len(room_id) == 8 for room_id in room_ids))
            self.assertEqual(len(set(room_ids)), 5)
            with self.assertRaisesRegex(ValueError, "満室"):
                manager.create([deck, deck])


class PlayerTokenTests(unittest.TestCase):
    def test_each_player_has_an_independent_token(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=1)
        with patch("app.room_manager.WorkerClient", FakeWorker):
            room = manager.create([deck, deck])
            manager.join(room.room_id)
            self.assertEqual(manager.authenticate(room.player_tokens[0]), (room.room_id, 0))
            self.assertEqual(manager.authenticate(room.player_tokens[1]), (room.room_id, 1))
            self.assertNotEqual(room.player_tokens[0], room.player_tokens[1])
            with self.assertRaises(ValueError):
                manager.authenticate("invalid-token")

    def test_finished_room_releases_worker_immediately(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=1)
        with patch("app.room_manager.WorkerClient", FinishingFakeWorker):
            room = manager.create([deck, deck])
            worker = room.worker
            manager.join(room.room_id)
            final_state = manager.state(room.room_id, 0)
            self.assertTrue(final_state["finished"])
            self.assertTrue(worker.closed)
            self.assertIsNone(room.worker)
            self.assertEqual(manager.active_room_count(), 0)
            self.assertTrue(manager.state(room.room_id, 1)["finished"])


class WorkerIntegrationTests(unittest.TestCase):
    def test_two_private_players_can_advance_the_same_match(self) -> None:
        deck = available_decks()[0]
        manager = RoomManager(max_rooms=2)
        try:
            room = manager.create([deck, deck])
            other_room = manager.create([deck, deck])
            manager.join(room.room_id.lower())
            manager.join(other_room.room_id)

            progressed = 0
            seen_observation = [False, False]
            for _ in range(20):
                states = [manager.state(room.room_id, role) for role in (0, 1)]
                host_observation = states[0].get("observation")
                guest_observation = states[1].get("observation")
                if host_observation and guest_observation:
                    host_players = host_observation["current"]["players"]
                    guest_players = guest_observation["current"]["players"]
                else:
                    host_players = guest_players = []
                guest_active = guest_players[0].get("active") if guest_players else None
                if (
                    host_players
                    and host_players[0].get("active")
                    and guest_active
                    and isinstance(guest_active[0], dict)
                    and guest_active[0].get("id") is not None
                ):
                    self.assertEqual(
                        host_players[1]["active"][0].get("id"),
                        guest_players[0]["active"][0].get("id"),
                    )
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
