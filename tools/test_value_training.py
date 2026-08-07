"""valueのlogit空間学習に関する軽量テスト。"""

from __future__ import annotations

import unittest

import torch

from .value_training import (
    build_value_loss,
    forward_for_training,
    value_targets_for_loss,
)


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.d_model = 4
        self.num_words_encoder = 2
        self.encoder_bag = torch.nn.EmbeddingBag(16, 4, mode="sum")
        self.encoder = torch.nn.Identity()
        self.encoder_fc = torch.nn.Linear(4, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(16, 4, mode="sum")
        self.decoder = torch.nn.ModuleList()
        self.decoder_fc = torch.nn.Linear(4, 1)

    def forward(self, *inputs):
        raw_value, policy = forward_for_training(self, *inputs, space="logit")
        return torch.tanh(raw_value), policy


class ValueTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _ToyModel()
        self.inputs = (
            torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            torch.ones(4),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            torch.tensor([5, 6, 7, 8], dtype=torch.int32),
            torch.ones(4),
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

    def test_training_value_round_trips_to_inference_value(self) -> None:
        inference_value, inference_policy = self.model(*self.inputs)
        training_value, training_policy = forward_for_training(
            self.model,
            *self.inputs,
            space="logit",
        )

        torch.testing.assert_close(torch.tanh(training_value), inference_value)
        torch.testing.assert_close(training_policy, inference_policy)

    def test_target_logit_round_trips_without_infinite_endpoints(self) -> None:
        targets = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
        logits = value_targets_for_loss(targets, space="logit")

        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(
            torch.tanh(logits[1:4]),
            targets[1:4],
        )
        self.assertLess(float(torch.tanh(logits[0])), -0.99)
        self.assertGreater(float(torch.tanh(logits[-1])), 0.99)

    def test_logit_training_avoids_tanh_gradient_suppression(self) -> None:
        raw = torch.tensor([[4.0]], requires_grad=True)
        bounded_loss = torch.nn.functional.huber_loss(
            torch.tanh(raw),
            torch.tensor([[-1.0]]),
            delta=0.2,
        )
        bounded_loss.backward()
        bounded_gradient = abs(float(raw.grad))

        raw.grad = None
        logit_loss = torch.nn.functional.huber_loss(
            raw,
            value_targets_for_loss(torch.tensor([[-1.0]]), space="logit"),
            delta=0.2,
        )
        logit_loss.backward()

        self.assertGreater(abs(float(raw.grad)), bounded_gradient * 100.0)

    def test_bce_optimum_matches_expected_outcome(self) -> None:
        # 勝率75%なら期待valueは +0.5。対応するlogitをBCEの停留点として確認する。
        raw_value = torch.tensor([[torch.atanh(torch.tensor(0.5))]], requires_grad=True)
        targets = torch.tensor([[1.0], [1.0], [1.0], [-1.0]])
        predictions = raw_value.expand_as(targets)

        loss = build_value_loss(torch, kind="bce")(predictions, targets)
        loss.backward()

        self.assertAlmostEqual(float(raw_value.grad), 0.0, places=6)

    def test_bce_step_updates_transformer_side_parameters(self) -> None:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        before = self.model.encoder_fc.weight.detach().clone()
        raw_value, _ = forward_for_training(
            self.model,
            *self.inputs,
            space="logit",
        )

        loss = build_value_loss(torch, kind="bce")(
            raw_value,
            torch.tensor([[1.0], [-1.0]]),
        )
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, self.model.encoder_fc.weight))


if __name__ == "__main__":
    unittest.main()
