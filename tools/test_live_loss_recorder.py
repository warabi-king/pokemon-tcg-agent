"""Notebook向けリアルタイムLoss記録のテスト。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from live_loss_recorder import LiveLossRecorder
from run_train_round_robin import loss_plateau_status


class LiveLossRecorderTest(unittest.TestCase):
    def test_records_batch_csv_and_atomic_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            recorder = LiveLossRecorder(
                Path(directory),
                ["agent_a", "agent_b"],
                target_loss=0.03,
            )
            recorder.update_status(
                "collecting",
                "samples",
                iteration=0,
            )
            recorder.record_batch(
                iteration=0,
                agent="agent_a",
                batch=2,
                batches=10,
                batch_loss=0.04,
                running_loss=0.05,
                value_loss=0.01,
                policy_loss=0.03,
            )

            with recorder.csv_path.open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["agent"], "agent_a")
            self.assertEqual(rows[0]["batch"], "2")
            self.assertEqual(float(rows[0]["running_loss"]), 0.05)

            status = json.loads(recorder.status_path.read_text(encoding="utf-8"))
            self.assertEqual(status["phase"], "training")
            self.assertEqual(status["agent"], "agent_a")
            self.assertEqual(status["batch"], 2)

            recorder.finish_training()
            status = json.loads(recorder.status_path.read_text(encoding="utf-8"))
            self.assertTrue(status["completed"])
            self.assertEqual(status["phase"], "completed")

    def test_plateau_requires_every_agent_to_stabilize(self) -> None:
        plateau, ranges = loss_plateau_status(
            {
                "stable_a": [0.0200, 0.0195, 0.0198],
                "stable_b": [0.0100, 0.0103, 0.0101],
            },
            window=3,
            delta=0.001,
        )
        self.assertTrue(plateau)
        self.assertLessEqual(max(ranges.values()), 0.001)

        plateau, ranges = loss_plateau_status(
            {
                "stable": [0.0200, 0.0195, 0.0198],
                "moving": [0.0300, 0.0250, 0.0200],
            },
            window=3,
            delta=0.001,
        )
        self.assertFalse(plateau)
        self.assertGreater(ranges["moving"], 0.001)


if __name__ == "__main__":
    unittest.main()
