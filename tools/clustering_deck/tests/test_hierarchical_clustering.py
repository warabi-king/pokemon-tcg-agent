"""clustering_deckの階層クラスタリングと主要デッキ選出を検証する。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from hierarchical_clustering import (  # noqa: E402
    add_hierarchical_fields,
    agglomerative_average_linkage,
    count_histogram,
    histogram_intersection_distance,
    load_records,
    write_records_atomic,
)
from evaluate_hierarchical_clustering import (  # noqa: E402
    CardMetadata,
    evaluate_records,
    render_markdown,
)
from render_hierarchical_dendrogram import build_cut_cluster_dendrogram  # noqa: E402
from select_cluster_major_decks import (  # noqa: E402
    select_cluster_decks,
    write_selection_outputs,
)


def make_record(card_counts: dict[int, int], games: int = 1) -> dict:
    """テスト用の60枚デッキレコードを作る。"""

    deck = [card_id for card_id, count in card_counts.items() for _ in range(count)]
    return {
        "deck": deck,
        "deck_counts": {str(card_id): count for card_id, count in card_counts.items()},
        "games": games,
        "wins": games,
        "losses": 0,
        "draws": 0,
        "win_rate": 1.0,
        "cluster_id": 999,
    }


class HistogramDistanceTest(unittest.TestCase):
    """カード枚数ヒストグラム距離の境界値を検証する。"""

    def test_identical_decks_have_zero_distance(self) -> None:
        """完全一致する60枚デッキの距離が0になる。"""

        histogram = {1: 60}
        self.assertEqual(histogram_intersection_distance(histogram, histogram), 0.0)

    def test_disjoint_decks_have_unit_distance(self) -> None:
        """共通カードがない60枚デッキの距離が1になる。"""

        self.assertEqual(histogram_intersection_distance({1: 60}, {2: 60}), 1.0)

    def test_forty_five_overlapping_cards_have_quarter_distance(self) -> None:
        """45枚一致するデッキの距離が0.25になる。"""

        distance = histogram_intersection_distance({1: 60}, {1: 45, 2: 15})
        self.assertEqual(distance, 0.25)


class AverageLinkageTest(unittest.TestCase):
    """平均連結法の統合結果と入力検証を確認する。"""

    def test_related_decks_form_two_expected_clusters(self) -> None:
        """近い2組のデッキがそれぞれ同じクラスタになる。"""

        histograms = [
            {1: 60},
            {1: 45, 2: 15},
            {3: 60},
            {3: 45, 4: 15},
        ]
        result = agglomerative_average_linkage(histograms, target_cluster_count=2)
        clusters = {frozenset(cluster) for cluster in result.clusters}
        self.assertEqual(clusters, {frozenset({0, 1}), frozenset({2, 3})})
        self.assertEqual(result.last_merge_distance, 0.25)
        self.assertEqual(result.next_merge_distance, 1.0)

    def test_invalid_cluster_count_is_rejected(self) -> None:
        """デッキ数を超えるクラスタ数が明示的なエラーになる。"""

        with self.assertRaisesRegex(ValueError, "clustersは1以上2以下"):
            agglomerative_average_linkage([{1: 60}, {2: 60}], 3)

    def test_non_sixty_card_record_is_rejected(self) -> None:
        """60枚でない入力レコードを処理しない。"""

        with self.assertRaisesRegex(ValueError, "60枚デッキ"):
            count_histogram(make_record({1: 59}))


class HierarchicalFieldsTest(unittest.TestCase):
    """JSONLへ追加する階層クラスタ情報と既存項目の保持を検証する。"""

    def test_fields_are_added_without_overwriting_existing_cluster(self) -> None:
        """既存cluster_idを残し、2クラスタ分の統計を追加する。"""

        records = [
            make_record({1: 60}, games=10),
            make_record({1: 45, 2: 15}, games=5),
            make_record({3: 60}, games=8),
            make_record({3: 45, 4: 15}, games=4),
        ]
        add_hierarchical_fields(records, target_cluster_count=2)

        self.assertTrue(all(record["cluster_id"] == 999 for record in records))
        self.assertEqual(
            {record["hierarchical_cluster_id"] for record in records},
            {0, 1},
        )
        self.assertTrue(
            all(record["hierarchical_cluster_count"] == 2 for record in records)
        )
        self.assertTrue(all(record["hierarchical_linkage"] == "average" for record in records))
        self.assertEqual(
            sum(record["hierarchical_cluster_representative"] for record in records),
            2,
        )

    def test_atomic_round_trip_preserves_unicode_jsonl(self) -> None:
        """原子的な書き込み後も日本語を含むJSONLを読み直せる。"""

        records = [make_record({1: 60})]
        records[0]["label"] = "主要デッキ"
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "decks.jsonl"
            write_records_atomic(output_path, records)
            loaded = load_records(output_path)
        self.assertEqual(loaded, json.loads(json.dumps(records, ensure_ascii=False)))


class HierarchicalEvaluationTest(unittest.TestCase):
    """定量・定性評価の主要指標と異常系を検証する。"""

    def setUp(self) -> None:
        """2種類の明確なクラスタを持つ4デッキを準備する。"""

        self.records = [
            make_record({1: 60}, games=10),
            make_record({1: 45, 2: 15}, games=5),
            make_record({3: 60}, games=8),
            make_record({3: 45, 4: 15}, games=4),
        ]
        add_hierarchical_fields(self.records, target_cluster_count=2)
        self.card_metadata = {
            1: CardMetadata(1, "超ポケモンA", "ポケモン/たね", "超"),
            2: CardMetadata(2, "超ポケモンB", "ポケモン/たね", "超"),
            3: CardMetadata(3, "炎ポケモンA", "ポケモン/たね", "炎"),
            4: CardMetadata(4, "炎ポケモンB", "ポケモン/たね", "炎"),
        }

    def test_evaluation_separates_two_obvious_clusters(self) -> None:
        """明確に異なる2群で、期待する類似度と純度が得られる。"""

        report = evaluate_records(self.records, self.card_metadata)

        self.assertEqual(report["method"]["cluster_count"], 2)
        self.assertEqual(
            report["deck_similarity"]["within_cluster"]["mean"],
            0.75,
        )
        self.assertEqual(
            report["deck_similarity"]["between_clusters"]["mean"],
            0.0,
        )
        self.assertEqual(report["silhouette"]["mean"], 0.75)
        self.assertEqual(
            report["pokemon_composition"]["within_cluster_jaccard"]["mean"],
            0.6,
        )
        self.assertEqual(
            report["pokemon_composition"]["dominant_type_purity"],
            1.0,
        )

    def test_markdown_contains_quantitative_and_qualitative_sections(self) -> None:
        """Markdownに定量・定性とクラスタ別概要が含まれる。"""

        markdown = render_markdown(evaluate_records(self.records, self.card_metadata))
        self.assertIn("## 定量評価", markdown)
        self.assertIn("## 定性評価の補助指標", markdown)
        self.assertIn("## クラスタ別概要", markdown)

    def test_missing_cluster_field_is_rejected(self) -> None:
        """階層クラスタIDが欠けた入力を明示的なエラーにする。"""

        records = [make_record({1: 60})]
        with self.assertRaisesRegex(ValueError, "hierarchical_cluster_id"):
            evaluate_records(records, self.card_metadata)

    def test_cluster_report_contains_card_and_type_ratios(self) -> None:
        """カード種別、ポケモンタイプ、エネルギータイプの割合を集計する。"""

        records = [make_record({1: 30, 2: 20, 3: 10}, games=3)]
        records[0]["hierarchical_cluster_id"] = 0
        metadata = {
            1: CardMetadata(1, "超ポケモン", "ポケモン/たね", "超"),
            2: CardMetadata(2, "テストグッズ", "グッズ", "n/a"),
            3: CardMetadata(3, "基本炎エネルギー", "基本エネルギー", "炎"),
        }

        report = evaluate_records(records, metadata)
        cluster = report["clusters"][0]
        categories = {
            item["category"]: item["ratio"]
            for item in cluster["card_category_distribution"]
        }
        self.assertEqual(categories["ポケモン"], 0.5)
        self.assertEqual(categories["グッズ"], 1 / 3)
        self.assertEqual(categories["エネルギー"], 1 / 6)
        self.assertEqual(cluster["pokemon_type_distribution"][0]["type"], "超")
        self.assertEqual(cluster["pokemon_type_distribution"][0]["ratio"], 1.0)
        self.assertEqual(cluster["energy_type_distribution"][0]["type"], "炎")
        self.assertEqual(cluster["energy_type_distribution"][0]["ratio"], 1.0)

    def test_cut_cluster_dendrogram_merges_all_cut_clusters(self) -> None:
        """切断済み2クラスタから上位階層の統合が1件生成される。"""

        cluster_ids, merges, root_node_id = build_cut_cluster_dendrogram(self.records)
        self.assertEqual(cluster_ids, [0, 1])
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0].member_count, 4)
        self.assertEqual(root_node_id, merges[0].new_node_id)


class ClusterMajorDeckSelectionTest(unittest.TestCase):
    """クラスタの代表・競技向け60枚デッキの選出を検証する。"""

    def test_representative_and_competitive_decks_use_different_priorities(self) -> None:
        """代表は中心性と実績、競技向けは中心性範囲内の補正勝率を優先する。"""

        records = [
            make_record({1: 60}, games=100),
            make_record({1: 57, 2: 3}, games=30),
        ]
        records[0].update(wins=40, losses=60, win_rate=0.4)
        records[1].update(wins=24, losses=6, win_rate=0.8)
        for record in records:
            record["hierarchical_cluster_id"] = 0

        selections = select_cluster_decks(records)

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0].representative.record_index, 0)
        self.assertEqual(selections[0].competitive.record_index, 1)
        self.assertTrue(records[0]["hierarchical_cluster_representative"])
        self.assertTrue(records[1]["hierarchical_cluster_competitive"])
        self.assertGreater(
            records[0]["hierarchical_cluster_centrality"],
            records[1]["hierarchical_cluster_centrality"],
        )

    def test_competitive_selection_falls_back_when_minimum_games_is_unmet(self) -> None:
        """最低試合数を満たす候補がなくても中心的な実在デッキを選出する。"""

        records = [make_record({1: 60}, games=2)]
        records[0]["hierarchical_cluster_id"] = 0

        selection = select_cluster_decks(
            records,
            minimum_competitive_games=20,
        )[0]

        self.assertEqual(selection.competitive.record_index, 0)
        self.assertFalse(selection.competitive_minimum_games_applied)

    def test_selection_outputs_are_sixty_card_csv_files(self) -> None:
        """代表・競技向けCSVがヘッダーなし60行で生成される。"""

        records = [make_record({1: 40, 2: 20}, games=100)]
        records[0]["hierarchical_cluster_id"] = 0
        selections = select_cluster_decks(records)
        parameters = {
            "cluster_field": "hierarchical_cluster_id",
            "representative_tolerance": 0.01,
            "competitive_tolerance": 0.03,
            "win_rate_prior_games": 20.0,
            "minimum_competitive_games": 20,
            "minimum_major_game_share": 0.005,
        }
        metadata = {
            1: CardMetadata(1, "ポケモンA", "ポケモン/たね", "超"),
            2: CardMetadata(2, "テストグッズ", "グッズ", ""),
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_selection_outputs(
                output_dir,
                records,
                selections,
                metadata,
                parameters,
            )
            representative_lines = (
                output_dir / "cluster_00_representative.csv"
            ).read_text(encoding="utf-8").splitlines()
            competitive_lines = (
                output_dir / "cluster_00_competitive.csv"
            ).read_text(encoding="utf-8").splitlines()
            self.assertTrue((output_dir / "manifest.csv").exists())
            self.assertTrue((output_dir / "selection_summary.json").exists())

        self.assertEqual(len(representative_lines), 60)
        self.assertEqual(len(competitive_lines), 60)


if __name__ == "__main__":
    unittest.main()
