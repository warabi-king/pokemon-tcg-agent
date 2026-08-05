"""デッキ候補DBの統合と多様性選抜を検証する。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents" / "belief_puct" / "train"))

from deck_database import load_records, select_diverse_records  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from search_deck_fixed_league import Opponent, aggregate_candidate, collect_candidate_decks, ranking_key  # noqa: E402


def record(card_id: int, cluster_id: int, wins: int, games: int) -> dict:
    """60枚の単純なテスト用候補レコードを作る。"""

    return {"deck": [card_id] * 60, "cluster_id": cluster_id, "wins": wins, "losses": games - wins, "draws": 0, "games": games, "cluster_win_rate": 0.5}


class DeckDatabaseTests(unittest.TestCase):
    """公式DBを壊さない候補選抜の最小振る舞いを確認する。"""

    def test_duplicate_decks_are_merged(self) -> None:
        """同じ60枚構成は勝敗を合算し、別候補へ増殖させない。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "candidates.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in (record(1, 0, 4, 5), record(1, 0, 3, 5))), encoding="utf-8")
            records = load_records([path])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["games"], 10)
        self.assertEqual(records[0]["wins"], 7)

    def test_selection_prefers_distinct_clusters_before_duplicates(self) -> None:
        """上位候補が同一クラスタでも、候補枠があれば別クラスタを先に残す。"""

        records = [record(1, 0, 90, 100), record(2, 0, 80, 100), record(3, 1, 60, 100)]
        selected = select_diverse_records(records, 2)
        self.assertEqual({item["cluster_id"] for item in selected}, {0, 1})

    def test_selection_excludes_single_game_winner(self) -> None:
        """単発勝利だけの候補は、より十分な試合数の候補より優先しない。"""

        records = [record(1, 0, 1, 1), record(2, 1, 60, 100)]
        selected = select_diverse_records(records, 1, min_games=20)
        self.assertEqual(selected[0]["deck"][0], 2)


class FixedLeagueRankingTests(unittest.TestCase):
    """固定リーグの順位付けが平均勝率だけへ偏らないことを確認する。"""

    def test_ranking_uses_worst_opponent_after_weighted_score(self) -> None:
        """平均が同じ候補では、最苦手相手にも勝てる候補を先に選ぶ。"""

        opponents = [
            Opponent("a", Path("agent-a"), Path("deck-a"), None, 1.0),
            Opponent("b", Path("agent-b"), Path("deck-b"), None, 1.0),
        ]
        unstable = aggregate_candidate(0, Path("unstable.csv"), opponents, [
            {"score_a": 1.0, "wins_a": 20, "wins_b": 0, "draws": 0, "games_completed": 20},
            {"score_a": 0.0, "wins_a": 0, "wins_b": 20, "draws": 0, "games_completed": 20},
        ])
        robust = aggregate_candidate(1, Path("robust.csv"), opponents, [
            {"score_a": 0.5, "wins_a": 10, "wins_b": 10, "draws": 0, "games_completed": 20},
            {"score_a": 0.5, "wins_a": 10, "wins_b": 10, "draws": 0, "games_completed": 20},
        ])
        self.assertEqual([item["candidate_index"] for item in sorted([unstable, robust], key=ranking_key)], [1, 0])

    def test_timeout_can_be_scored_as_candidate_loss(self) -> None:
        """短時間リーグでは未判定試合を候補側の敗戦として採点できる。"""

        opponent = Opponent("fast", Path("agent"), Path("deck"), None, 1.0)
        candidate = aggregate_candidate(0, Path("candidate.csv"), [opponent], [{
            "score_a": 1.0, "wins_a": 9, "wins_b": 0, "draws": 0,
            "games_completed": 9, "games_requested": 10, "errors": 1,
        }], errors_as_losses=True)
        self.assertEqual(candidate["weighted_score"], 0.9)

    def test_candidate_directory_collects_sorted_decks(self) -> None:
        """pipeline出力の候補ディレクトリを番号順で全件読める。"""

        with tempfile.TemporaryDirectory() as temporary_directory:
            candidates = Path(temporary_directory) / "candidates"
            for name in ("candidate_010", "candidate_002"):
                deck_path = candidates / name / "deck.csv"
                deck_path.parent.mkdir(parents=True)
                deck_path.write_text("1\n" * 60, encoding="utf-8")
            decks = collect_candidate_decks([], [candidates])
        self.assertEqual([deck.parent.name for deck in decks], ["candidate_002", "candidate_010"])


if __name__ == "__main__":
    unittest.main()
