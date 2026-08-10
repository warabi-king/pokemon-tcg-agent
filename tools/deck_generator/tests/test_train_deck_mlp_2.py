from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "train_deck_mlp_2.py"
SPEC = importlib.util.spec_from_file_location("train_deck_mlp_2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MaskedDeckDatasetTest(unittest.TestCase):
    def test_target_is_only_the_masked_remainder(self) -> None:
        record = MODULE.DeckRecord(
            deck=tuple([1] * 20 + [10] * 40),
            cluster_id=4,
            wins=10,
            games=20,
            cluster_weight=1.0,
        )
        counts = torch.tensor([[20.0, 40.0]])
        dataset = MODULE.MaskedDeckDataset(
            [record],
            counts,
            torch.tensor([20.0, 4.0]),
            {4: 0},
            samples_per_deck=1,
            min_observed=12,
            max_observed=12,
            seed=0,
        )
        features, missing, cluster, observed = dataset.generate(0)
        self.assertEqual(int(observed.sum()), 12)
        self.assertTrue(torch.equal(observed + missing, counts[0]))
        self.assertEqual(int(missing.sum()), 48)
        self.assertEqual(int(cluster), 0)
        self.assertEqual(features.shape, (3,))


class CompatibilityTest(unittest.TestCase):
    def test_cluster_prior_suppresses_an_unrelated_card(self) -> None:
        presence_logits = torch.zeros((1, 2))
        cluster_logits = torch.tensor([[8.0, -8.0]])
        prior = torch.tensor([[0.9, 0.02], [0.02, 0.9]])
        presence, cluster_probability, compatibility = MODULE.combine_presence_with_cluster_prior(
            presence_logits,
            cluster_logits,
            prior,
            compatibility_strength=1.0,
        )
        self.assertGreater(float(cluster_probability[0, 0]), 0.99)
        self.assertGreater(float(compatibility[0, 0]), 0.89)
        self.assertLess(float(compatibility[0, 1]), 0.03)
        self.assertGreater(float(presence[0, 0]), float(presence[0, 1]) * 20)

    def test_observed_card_cooccurrence_suppresses_a_foreign_family(self) -> None:
        presence_logits = torch.zeros((1, 3))
        cluster_logits = torch.zeros((1, 1))
        cluster_prior = torch.ones((1, 3))
        observed_counts = torch.tensor([[1.0, 0.0, 0.0]])
        conditional = torch.tensor(
            [
                [1.0, 0.8, 0.01],
                [0.8, 1.0, 0.02],
                [0.01, 0.02, 1.0],
            ]
        )
        presence, _, compatibility = MODULE.combine_presence_with_cluster_prior(
            presence_logits,
            cluster_logits,
            cluster_prior,
            compatibility_strength=1.0,
            observed_counts=observed_counts,
            card_conditional_prior=conditional,
            cooccurrence_strength=1.0,
        )
        self.assertGreater(float(compatibility[0, 1]), 0.79)
        self.assertLess(float(compatibility[0, 2]), 0.02)
        self.assertGreater(float(presence[0, 1]), float(presence[0, 2]) * 50)

    def test_completion_keeps_mixing_within_selected_compatible_cards(self) -> None:
        known = [1, 10, 20, 30]
        meta = {
            1: MODULE.CardMeta(1, "Energy", "Basic Energy", ""),
            10: MODULE.CardMeta(10, "Observed", "Basic", ""),
            20: MODULE.CardMeta(20, "Related A", "Item", ""),
            30: MODULE.CardMeta(30, "Foreign", "Item", ""),
        }
        predicted = torch.tensor([12.0, 2.0, 4.0, 4.0])
        presence = torch.tensor([0.4, 0.7, 0.9, 0.001])
        deck = MODULE.complete_deck(
            [10, 10],
            predicted,
            presence,
            known,
            meta,
            min_unique_cards=3,
            max_unique_cards=3,
        )
        counts = Counter(deck)
        self.assertEqual(len(deck), 60)
        self.assertIn(20, counts)
        self.assertNotIn(30, counts)
        self.assertLessEqual(counts[10], 4)


class ModelAndSplitTest(unittest.TestCase):
    def test_model_has_three_expected_heads(self) -> None:
        model = MODULE.DeckCompletionMLP2(5, 3, hidden_size=8, layers=2, dropout=0)
        presence, counts, clusters = model(torch.zeros((2, 6)))
        self.assertEqual(presence.shape, (2, 5))
        self.assertEqual(counts.shape, (2, 5))
        self.assertEqual(clusters.shape, (2, 3))

    def test_stratified_split_keeps_each_cluster_in_training(self) -> None:
        records = [
            MODULE.DeckRecord(tuple([index + 1] * 60), cluster, 1, 1, 1.0)
            for cluster in range(3)
            for index in range(3)
        ]
        train, valid = MODULE.split_records_by_cluster(records, 0.34, seed=0)
        self.assertEqual({record.cluster_id for record in train}, {0, 1, 2})
        self.assertEqual({record.cluster_id for record in valid}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
