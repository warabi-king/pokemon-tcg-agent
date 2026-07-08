from __future__ import annotations

import unittest

from app.cloud_agent_manager import CloudAgentManager, available_agents


class CloudAgentModeIntegrationTests(unittest.TestCase):
    def test_play_and_watch_workers_advance(self) -> None:
        agent = available_agents()[0]
        manager = CloudAgentManager(lambda: len(manager.matches), max_matches=5)
        try:
            play, play_state = manager.create("play", [agent])
            self.assertTrue(play_state["started"])
            if play_state["humanTurn"]:
                selection = play_state["observation"]["select"]
                indices = list(range(selection["minCount"]))
                play_state = manager.request(
                    play.token, "play", "action", indices=indices
                )
                self.assertTrue(play_state["started"])

            watch, watch_state = manager.create("watch", [agent, agent])
            self.assertTrue(watch_state["started"])
            next_state = manager.request(watch.token, "watch", "step")
            self.assertGreaterEqual(next_state["step"], 1)
            self.assertEqual(next_state["agentNames"], [agent, agent])
        finally:
            manager.close_all()


if __name__ == "__main__":
    unittest.main()
