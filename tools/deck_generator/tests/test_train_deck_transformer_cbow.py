"""勝率重み付きTransformerデッキ補完の学習対象と生成制約を検証する。"""

from __future__ import annotations

import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from train_deck_transformer_cbow import (  # noqa: E402
    CardMeta,
    DeckRecord,
    DeckTransformerCBOW,
    adjusted_win_rate_weight,
    choose_next_card,
    fallback_card_id,
    generate_deck,
    generation_candidate_allowed,
    load_resume_checkpoint,
    split_context_and_remaining_distribution,
    validate_observed_cards,
)


def card_metadata() -> dict[int, CardMeta]:
    """同名、基本エネルギー、ACE SPECを含む最小カード情報を返す。"""
    return {
        1: CardMeta(card_id=1, name="Alpha", kind="Basic", rule=""),
        2: CardMeta(card_id=2, name="Alpha", kind="Stage 1", rule=""),
        3: CardMeta(card_id=3, name="Energy", kind="Basic Energy", rule=""),
        4: CardMeta(card_id=4, name="Ace A", kind="Item", rule="ACE SPEC"),
        5: CardMeta(card_id=5, name="Ace B", kind="Item", rule="ACE SPEC"),
        6: CardMeta(card_id=6, name="Beta", kind="Basic", rule=""),
    }


class ConstantLogitModel(torch.nn.Module):
    """指定カードを常に最高logitにする生成ループ確認用モデル。"""

    def __init__(self, vocab_size: int, preferred_card_id: int) -> None:
        """語彙数と最高logitにするカードIDを保持する。"""
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_card_id = preferred_card_id
        self.pad_id = vocab_size
        self.bos_id = vocab_size + 1

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """batchごとに固定logitを返す。"""
        del padding_mask
        logits = torch.zeros((token_ids.size(0), self.vocab_size), device=token_ids.device)
        logits[:, self.preferred_card_id] = 10.0
        return logits


class TrainingTargetTests(unittest.TestCase):
    """残り枚数分布と補正勝率loss重みを検証する。"""

    def test_consumed_copies_have_zero_remaining_probability(self) -> None:
        """文脈で全枚数を消費したカードの教師確率が0になることを確認する。"""
        context, remaining = split_context_and_remaining_distribution(
            deck=[1, 1, 1, 1, 2, 2],
            context_positions={0, 1, 2, 3},
            vocab_size=7,
        )

        self.assertEqual(context, [1, 1, 1, 1])
        self.assertEqual(float(remaining[1]), 0.0)
        self.assertEqual(float(remaining[2]), 1.0)

    def test_adjusted_win_rate_weight_uses_beta_and_clipping(self) -> None:
        """補正勝率の指数重みと上下限制限を確認する。"""
        strong = DeckRecord(deck=[], adjusted_win_rate=0.6)
        very_strong = DeckRecord(deck=[], adjusted_win_rate=1.0)

        self.assertAlmostEqual(adjusted_win_rate_weight(strong, 3.0, 0.25, 4.0), math.exp(0.3))
        self.assertEqual(adjusted_win_rate_weight(very_strong, 10.0, 0.25, 4.0), 4.0)


class ResumeCheckpointTests(unittest.TestCase):
    """旧形式の重み継続と新形式の完全再開を検証する。"""

    def setUp(self) -> None:
        """小さいTransformerと対応する設定を用意する。"""
        self.config = {
            "vocab_size": 7,
            "pad_id": 7,
            "bos_id": 8,
            "embedding_dim": 4,
            "heads": 2,
            "layers": 1,
            "ff_dim": 8,
            "dropout": 0.0,
            "max_context": 3,
            "target_mode": "remaining-count-distribution",
        }

    def create_model(self) -> DeckTransformerCBOW:
        """テスト設定と一致する小さいモデルを返す。"""
        return DeckTransformerCBOW(
            vocab_size=self.config["vocab_size"],
            embedding_dim=self.config["embedding_dim"],
            heads=self.config["heads"],
            layers=self.config["layers"],
            ff_dim=self.config["ff_dim"],
            dropout=self.config["dropout"],
            pad_id=self.config["pad_id"],
            bos_id=self.config["bos_id"],
        )

    def test_legacy_checkpoint_restores_weights_only(self) -> None:
        """training_stateがない既存checkpointからモデル重みを引き継ぐ。"""
        source_model = self.create_model()
        target_model = self.create_model()
        optimizer = torch.optim.AdamW(target_model.parameters())
        checkpoint = {
            "model_state": source_model.state_dict(),
            "config": self.config,
            "metrics": {"best_valid_loss": 1.25, "best_valid_top1": 0.2, "best_valid_top5": 0.5},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pt"
            torch.save(checkpoint, path)
            resumed = load_resume_checkpoint(path, target_model, optimizer, self.config, torch.device("cpu"))

        self.assertFalse(resumed.optimizer_restored)
        self.assertEqual(resumed.completed_epochs, 0)
        self.assertEqual(resumed.best_valid_loss, 1.25)
        for key, expected in source_model.state_dict().items():
            self.assertTrue(torch.equal(target_model.state_dict()[key], expected))

    def test_new_checkpoint_restores_latest_model_optimizer_and_epoch(self) -> None:
        """新形式checkpointから最終重み・AdamW状態・完了epochを復元する。"""
        source_model = self.create_model()
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.01)
        loss = sum(parameter.sum() for parameter in source_model.parameters())
        loss.backward()
        source_optimizer.step()

        target_model = self.create_model()
        target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.5)
        checkpoint = {
            "model_state": source_model.state_dict(),
            "config": self.config,
            "metrics": {"best_valid_loss": 0.8, "best_valid_top1": 0.3, "best_valid_top5": 0.6},
            "training_state": {
                "latest_model_state": source_model.state_dict(),
                "optimizer_state": source_optimizer.state_dict(),
                "completed_epochs": 7,
                "best_valid_loss": 0.8,
                "best_valid_top1": 0.3,
                "best_valid_top5": 0.6,
                "best_model_state": source_model.state_dict(),
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "full.pt"
            torch.save(checkpoint, path)
            resumed = load_resume_checkpoint(
                path,
                target_model,
                target_optimizer,
                self.config,
                torch.device("cpu"),
            )

        self.assertTrue(resumed.optimizer_restored)
        self.assertEqual(resumed.completed_epochs, 7)
        self.assertTrue(target_optimizer.state_dict()["state"])
        self.assertEqual(target_optimizer.param_groups[0]["lr"], 0.01)
        for key, expected in source_model.state_dict().items():
            self.assertTrue(torch.equal(target_model.state_dict()[key], expected))

    def test_resume_rejects_different_architecture(self) -> None:
        """モデル構造が異なるcheckpointを読み込まず明示エラーにする。"""
        model = self.create_model()
        optimizer = torch.optim.AdamW(model.parameters())
        mismatched_config = dict(self.config)
        mismatched_config["layers"] = 2
        checkpoint = {"model_state": model.state_dict(), "config": mismatched_config}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mismatch.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "layers"):
                load_resume_checkpoint(path, model, optimizer, self.config, torch.device("cpu"))


class GenerationConstraintTests(unittest.TestCase):
    """greedy・sample共通の候補制約と入力検証を確認する。"""

    def setUp(self) -> None:
        """各テストで使うカード情報を初期化する。"""
        self.meta = card_metadata()
        self.known = sorted(self.meta)

    def test_same_name_limit_applies_across_card_ids(self) -> None:
        """別IDでも同名カードが合計4枚なら追加できないことを確認する。"""
        deck = [1, 1, 1, 2]

        self.assertFalse(generation_candidate_allowed(1, deck, self.meta, 18))
        self.assertFalse(generation_candidate_allowed(2, deck, self.meta, 18))

    def test_ace_spec_and_basic_energy_limits_are_shared(self) -> None:
        """ACE SPECと基本エネルギー上限が共通候補判定へ入ることを確認する。"""
        self.assertFalse(generation_candidate_allowed(5, [4], self.meta, 18))
        self.assertFalse(generation_candidate_allowed(3, [3, 3], self.meta, 2))
        self.assertTrue(generation_candidate_allowed(3, [3, 3], self.meta, -1))

    def test_greedy_chooses_highest_scoring_legal_card(self) -> None:
        """最大logitが違反カードの場合に次点の合法カードを選ぶことを確認する。"""
        logits = torch.zeros(7)
        logits[1] = 10.0
        logits[6] = 9.0

        chosen = choose_next_card(
            logits=logits,
            known_card_ids=self.known,
            deck=[1, 1, 1, 2],
            card_meta=self.meta,
            strategy="greedy",
            temperature=1.0,
            top_k=5,
            copy_penalty=0.0,
            energy_penalty=0.0,
            max_basic_energy=18,
            rng=random.Random(0),
        )

        self.assertEqual(chosen, 6)

    def test_sample_requires_positive_temperature(self) -> None:
        """確率サンプリングで0以下のtemperatureを拒否することを確認する。"""
        with self.assertRaisesRegex(ValueError, "temperature"):
            choose_next_card(
                logits=torch.zeros(7),
                known_card_ids=self.known,
                deck=[6],
                card_meta=self.meta,
                strategy="sample",
                temperature=0.0,
                top_k=5,
                copy_penalty=0.0,
                energy_penalty=0.0,
                max_basic_energy=18,
                rng=random.Random(0),
            )

    def test_sample_is_reproducible_with_the_same_seed(self) -> None:
        """同じseedの確率サンプリングが同じカードを返すことを確認する。"""
        arguments = {
            "logits": torch.arange(7, dtype=torch.float32),
            "known_card_ids": self.known,
            "deck": [6],
            "card_meta": self.meta,
            "strategy": "sample",
            "temperature": 1.2,
            "top_k": 5,
            "copy_penalty": 0.0,
            "energy_penalty": 0.0,
            "max_basic_energy": 18,
        }

        first = choose_next_card(**arguments, rng=random.Random(42))
        second = choose_next_card(**arguments, rng=random.Random(42))

        self.assertEqual(first, second)

    def test_fallback_never_bypasses_constraints(self) -> None:
        """合法候補がないfallbackが固定カードを不正追加せず失敗することを確認する。"""
        with self.assertRaisesRegex(RuntimeError, "no legal card"):
            fallback_card_id(
                known_card_ids=[1, 2, 3, 4, 5],
                card_meta=self.meta,
                deck=[1, 1, 1, 2, 3, 4],
                max_basic_energy=1,
            )

    def test_observed_cards_must_be_partial_known_and_legal(self) -> None:
        """部分デッキの枚数、既知ID、既存制約を生成前に検証する。"""
        validate_observed_cards([1, 1, 2, 6], self.known, self.meta, 18)
        validate_observed_cards([3] * 59, self.known, self.meta, -1)

        with self.assertRaisesRegex(ValueError, "between 1 and 59"):
            validate_observed_cards([], self.known, self.meta, 18)
        with self.assertRaisesRegex(ValueError, "between 1 and 59"):
            validate_observed_cards([3] * 60, self.known, self.meta, -1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_observed_cards([99], self.known, self.meta, 18)
        with self.assertRaisesRegex(ValueError, "constraints"):
            validate_observed_cards([1, 1, 1, 1, 2], self.known, self.meta, 18)

    def test_generate_deck_completes_a_59_card_input(self) -> None:
        """59枚の合法な入力を保持して60枚へ補完することを確認する。"""
        model = ConstantLogitModel(vocab_size=7, preferred_card_id=6)

        deck = generate_deck(
            model=model,
            observed_cards=[3] * 59,
            known_card_ids=self.known,
            card_meta=self.meta,
            device=torch.device("cpu"),
            strategy="greedy",
            temperature=1.0,
            top_k=5,
            copy_penalty=0.0,
            energy_penalty=0.0,
            max_basic_energy=-1,
            seed=0,
        )

        self.assertEqual(len(deck), 60)
        self.assertEqual(deck[:59], [3] * 59)
        self.assertEqual(deck[-1], 6)


if __name__ == "__main__":
    unittest.main()
