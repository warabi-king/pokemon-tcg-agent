"""固定リーグ集約と提出アーカイブ静的検査の回帰テスト。"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from summarize_league import aggregate_reports  # noqa: E402
from validate_submission import inspect_members  # noqa: E402


class StrategyToolTests(unittest.TestCase):
    """統計集約とアーカイブ境界の重要な振る舞いを検証する。"""

    @staticmethod
    def _report(wins_a: int, wins_b: int, draws: int = 0) -> dict:
        """テスト用の最小評価JSONを作る。"""

        completed = wins_a + wins_b + draws
        return {
            "format": "pokemon-tcg-agent/fixed-league-evaluation-v1",
            "configuration": {"name_a": "a", "name_b": "b"},
            "summary": {
                "games_requested": completed,
                "games_completed": completed,
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
                "errors": 0,
                "score_a": 0.0,
                "wilson_lower_95": 0.0,
                "wilson_upper_95": 1.0,
                "elapsed_seconds": float(completed),
                "mean_seconds_per_game": 1.0,
            },
        }

    def test_aggregate_recomputes_draw_score_and_wilson(self) -> None:
        """drawを0.5勝として合算し、成分の平均で代用しないことを確認する。"""

        aggregate = aggregate_reports(
            [self._report(6, 3, 1), self._report(4, 6, 0)],
            "test",
        )
        summary = aggregate["summary"]
        self.assertEqual(summary["games_completed"], 20)
        self.assertEqual(summary["wins_a"], 10)
        self.assertEqual(summary["draws"], 1)
        self.assertAlmostEqual(summary["score_a"], 0.525)
        self.assertLess(summary["wilson_lower_95"], summary["score_a"])
        self.assertGreater(summary["wilson_upper_95"], summary["score_a"])

    def test_archive_inspection_rejects_training_cache(self) -> None:
        """学習物やcacheが提出物へ混入した場合に失敗することを確認する。"""

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name in ("main.py", "deck.csv", "cg/api.py", "train/checkpoint.pth"):
                info = tarfile.TarInfo(name)
                info.size = 0
                archive.addfile(info, io.BytesIO())
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as archive:
            with self.assertRaisesRegex(ValueError, "提出不要物"):
                inspect_members(archive)


if __name__ == "__main__":
    unittest.main()
