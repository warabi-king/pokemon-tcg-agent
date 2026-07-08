from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path

from app.cloud_agent_manager import CloudAgentManager, available_agents


class FinishingAgentWorker:
    def __init__(self, _mode, _agents, _human_deck=None) -> None:
        self.closed = False

    def request(self, _payload):
        return {"started": True, "finished": True, "step": 99}

    def close(self) -> None:
        self.closed = True


class CloudAgentModeIntegrationTests(unittest.TestCase):
    def test_torch_agent_worker_starts(self) -> None:
        if "rl_mcts" not in available_agents():
            self.skipTest("rl_mcts is not available")
        manager = CloudAgentManager(lambda: len(manager.matches), max_matches=1)
        try:
            match, state = manager.create("play", ["rl_mcts"], "rule_Lucario")
            self.assertTrue(state["started"])
            self.assertIn(match.token, manager.matches)
        finally:
            manager.close_all()

    def test_finished_agent_match_releases_worker_immediately(self) -> None:
        agent = available_agents()[0]
        manager = CloudAgentManager(lambda: len(manager.matches), max_matches=1)
        with patch("app.cloud_agent_manager.AgentWorkerClient", FinishingAgentWorker):
            match, result = manager.create("play", [agent])
            self.assertTrue(result["finished"])
            self.assertTrue(match.worker.closed)
            self.assertNotIn(match.token, manager.matches)
            self.assertEqual(
                manager.request(match.token, "play", "state")["step"], 99
            )

    def test_play_and_watch_workers_advance(self) -> None:
        agent = available_agents()[0]
        manager = CloudAgentManager(lambda: len(manager.matches), max_matches=5)
        try:
            deck_path = Path("agents") / agent / "src" / "deck.csv"
            uploaded_deck = [int(value) for value in deck_path.read_text().splitlines() if value]
            play, play_state = manager.create("play", [agent], uploaded_deck)
            self.assertTrue(play_state["started"])
            ai_steps = 0
            for _ in range(20):
                if play_state["finished"]:
                    break
                previous_step = play_state["step"]
                if play_state["humanTurn"]:
                    selection = play_state["observation"]["select"]
                    indices = list(range(selection["minCount"]))
                    play_state = manager.request(
                        play.token, "play", "action", indices=indices
                    )
                else:
                    play_state = manager.request(play.token, "play", "step")
                    ai_steps += 1
                self.assertGreater(play_state["step"], previous_step)
            self.assertGreater(ai_steps, 0)

            watch, watch_state = manager.create("watch", [agent, agent])
            self.assertTrue(watch_state["started"])
            next_state = manager.request(watch.token, "watch", "step")
            self.assertGreaterEqual(next_state["step"], 1)
            self.assertEqual(next_state["agentNames"], [agent, agent])
        finally:
            manager.close_all()


if __name__ == "__main__":
    unittest.main()
