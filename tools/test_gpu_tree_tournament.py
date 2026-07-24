"""device常駐MCTS総当たりbackendのテスト。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gpu_tree_tournament import run_gpu_tree_tournament


class GpuTreeTournamentIntegrationTest(unittest.TestCase):
    def test_cpu_device_tree_finishes_matches(self) -> None:
        src = ROOT / "agents" / "rl_mcts_r_robin1" / "src"
        specs = [
            SimpleNamespace(
                name=name,
                agent_path=src / "main.py",
                deck_path=src / "deck.csv",
            )
            for name in ("a", "b")
        ]
        output = run_gpu_tree_tournament(
            specs,
            [("a", "b")],
            2,
            device_name="cpu",
            batch_size=8,
            lanes=2,
            search_count=1,
            max_selections=500,
            seed=321,
        )

        self.assertEqual(len(output.results), 2)
        self.assertTrue(all(result.error is None for result in output.results))
        self.assertTrue(all(result.result in (0, 1, 2) for result in output.results))
        self.assertGreater(output.profile.nn_evaluations, 0)
        self.assertGreater(output.profile.gpu_tree_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
