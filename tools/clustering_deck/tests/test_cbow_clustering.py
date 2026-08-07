"""CBOWデッキ類似度とspherical k-meansを検証する。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from cbow_clustering import (  # noqa: E402
    build_deck_features,
    choose_cluster_count,
    deck_similarity,
    silhouette_score,
    spherical_kmeans,
    render_markdown_report,
)


class DeckFeatureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.99, 0.10, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.histograms = [{1: 60}, {2: 60}, {3: 60}]

    def test_identical_deck_similarity_is_one(self) -> None:
        model = build_deck_features(self.histograms, self.embeddings)
        self.assertAlmostEqual(deck_similarity(model, 0, 0), 1.0, places=6)

    def test_cbow_near_substitution_is_closer_than_unrelated_card(self) -> None:
        model = build_deck_features(
            self.histograms,
            self.embeddings,
            semantic_weight=0.9,
        )
        self.assertGreater(
            deck_similarity(model, 0, 1),
            deck_similarity(model, 0, 2),
        )

    def test_zero_semantic_weight_uses_only_card_composition(self) -> None:
        model = build_deck_features(
            self.histograms,
            self.embeddings,
            semantic_weight=0.0,
        )
        self.assertAlmostEqual(deck_similarity(model, 0, 0), 1.0, places=6)
        self.assertAlmostEqual(deck_similarity(model, 0, 1), 0.0, places=6)

    def test_out_of_vocabulary_card_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "語彙サイズ"):
            build_deck_features([{99: 60}], self.embeddings)


class SphericalKMeansTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = torch.tensor(
            [
                [1.0, 0.00],
                [1.0, 0.05],
                [1.0, -0.05],
                [0.95, 0.10],
                [-1.0, 0.00],
                [-1.0, 0.05],
                [-1.0, -0.05],
                [-0.95, 0.10],
            ],
            dtype=torch.float32,
        )
        self.features = F.normalize(raw, dim=1)

    def test_two_separated_groups_are_recovered(self) -> None:
        result = spherical_kmeans(self.features, 2, n_init=4, seed=7)
        left_labels = set(result.labels[:4].tolist())
        right_labels = set(result.labels[4:].tolist())
        self.assertEqual(len(left_labels), 1)
        self.assertEqual(len(right_labels), 1)
        self.assertNotEqual(left_labels, right_labels)
        self.assertGreater(silhouette_score(self.features, result.labels), 0.9)

    def test_initialization_is_deterministic(self) -> None:
        first = spherical_kmeans(self.features, 2, n_init=3, seed=11)
        second = spherical_kmeans(self.features, 2, n_init=3, seed=11)
        self.assertTrue(torch.equal(first.labels, second.labels))
        self.assertAlmostEqual(first.inertia, second.inertia, places=7)

    def test_silhouette_selects_two_groups(self) -> None:
        _, clusters, evaluations = choose_cluster_count(
            self.features,
            [2, 3],
            n_init=4,
            max_iterations=50,
            seed=0,
        )
        self.assertEqual(clusters, 2)
        self.assertEqual([item["clusters"] for item in evaluations], [2, 3])


class MarkdownReportTest(unittest.TestCase):
    def test_report_contains_cluster_results_and_deck_composition(self) -> None:
        report = {
            "method": {
                "selected_clusters": 1,
                "selected_silhouette": 0.75,
                "semantic_weight": 0.7,
                "composition_weight": 0.3,
            },
            "source": {"decks": 2, "cards": 2},
            "clusters": [
                {
                    "cluster_id": 0,
                    "members": 2,
                    "games": 10,
                    "wins": 6,
                    "losses": 3,
                    "draws": 1,
                    "win_rate": 0.6,
                    "mean_centroid_similarity": 0.9,
                    "representative_record_index": 1,
                    "top_cards": [
                        {"card_id": 7, "name": "Card A", "decks": 2, "copies": 8}
                    ],
                    "representative_cards": [
                        {"card_id": 7, "name": "Card A", "count": 4}
                    ],
                }
            ],
        }
        markdown = render_markdown_report(report)
        self.assertIn("|0|2|10|6-3-1|60.00%|0.9000|Card A|", markdown)
        self.assertIn("|7|Card A|2|100.0%|8|4.00|", markdown)
        self.assertIn("代表デッキ", markdown)


if __name__ == "__main__":
    unittest.main()
