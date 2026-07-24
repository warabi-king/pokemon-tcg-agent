"""共有libcg＋試合横断NN batch backendのテスト。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from batched_tournament import (
    _enumerate_actions,
    _pad_sparse_offsets,
    run_batched_tournament,
)


class ActionBatchTest(unittest.TestCase):
    def test_enumerate_actions_respects_limit(self) -> None:
        actions = _enumerate_actions(option_count=10, select_count=2, limit=7)
        self.assertEqual(len(actions), 7)
        self.assertEqual(actions[0], [0, 1])
        self.assertEqual(len({tuple(action) for action in actions}), 7)

    def test_sparse_offsets_are_padded_with_empty_bags(self) -> None:
        sparse = SimpleNamespace(index=[3, 5], offset=[0, 1])
        _pad_sparse_offsets(sparse, 4)
        self.assertEqual(sparse.offset, [0, 1, 2, 2])


class BatchedTournamentIntegrationTest(unittest.TestCase):
    def test_two_matches_share_one_model_batch(self) -> None:
        src = ROOT / "agents" / "rl_mcts_r_robin1" / "src"
        specs = [
            SimpleNamespace(
                name=name,
                agent_path=src / "main.py",
                deck_path=src / "deck.csv",
            )
            for name in ("a", "b")
        ]
        output = run_batched_tournament(
            specs,
            [("a", "b")],
            2,
            device_name="cpu",
            batch_size=8,
            lanes=2,
            search_count=1,
            max_selections=500,
            seed=123,
        )

        self.assertEqual(len(output.results), 2)
        self.assertTrue(all(result.error is None for result in output.results))
        self.assertTrue(all(result.result in (0, 1, 2) for result in output.results))
        self.assertGreaterEqual(output.profile.max_batch_size, 2)
        self.assertGreater(output.profile.search_steps, 0)


if __name__ == "__main__":
    unittest.main()
