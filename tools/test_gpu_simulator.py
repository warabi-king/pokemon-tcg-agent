"""gpu_simulatorの静的テーブルと基礎戦闘カーネルのテスト。"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gpu_simulator import (  # noqa: E402
    AttackBatchKernel,
    AttackBatchState,
    EffectOpcode,
    TensorCardDatabase,
    compile_effect_text,
)


class EffectCompilerTest(unittest.TestCase):
    def test_known_effects(self) -> None:
        self.assertEqual(
            compile_effect_text("Draw 3 cards.").opcode,
            EffectOpcode.DRAW,
        )
        self.assertEqual(
            compile_effect_text("Draw 3 cards.").argument,
            3,
        )
        self.assertEqual(
            compile_effect_text("This Pokémon also does 30 damage to itself.").opcode,
            EffectOpcode.SELF_DAMAGE,
        )
        self.assertEqual(
            compile_effect_text("Your opponent’s Active Pokémon is now Poisoned.").opcode,
            EffectOpcode.APPLY_POISON,
        )

    def test_complex_effect_is_not_silently_accepted(self) -> None:
        effect = compile_effect_text(
            "Flip a coin. If heads, this attack does 20 more damage."
        )
        self.assertEqual(effect.opcode, EffectOpcode.UNSUPPORTED)
        self.assertFalse(effect.supported)


class TensorCardDatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = TensorCardDatabase.from_cg(device="cpu")

    def test_every_sdk_card_and_attack_is_loaded(self) -> None:
        self.assertEqual(self.db.card_count, 1267)
        self.assertEqual(int(self.db.card_exists.sum()), self.db.card_count)
        self.assertEqual(self.db.attack_count, 1556)
        self.assertEqual(int(self.db.attack_exists.sum()), self.db.attack_count)

    def test_all_attack_owners_are_valid_cards(self) -> None:
        owners = self.db.attack_owner[self.db.attack_exists]
        self.assertTrue(bool(torch.all(owners >= 0)))
        self.assertTrue(bool(torch.all(self.db.card_exists[owners])))

    def test_base_damage_knocks_out_and_takes_correct_prizes(self) -> None:
        db = self.db
        attack_candidates = torch.nonzero(
            db.attack_exists & (db.attack_damage > 0) & db.attack_effect_supported,
            as_tuple=False,
        ).flatten()
        attack_id = int(attack_candidates[0])
        attacker_id = int(db.attack_owner[attack_id])
        defender_candidates = torch.nonzero(
            db.card_exists & (db.card_type == 0) & (db.hp > 0), as_tuple=False
        ).flatten()
        defender_id = int(defender_candidates[0])
        state = AttackBatchState.create(
            torch.tensor([[attacker_id, defender_id]], dtype=torch.int64), db
        )
        state.hp[0, 1] = 1
        AttackBatchKernel(db).step(state, torch.tensor([attack_id]))
        self.assertEqual(int(state.hp[0, 1]), 0)
        expected_prizes = 3 if bool(db.is_mega_ex[defender_id]) else 2 if bool(db.is_ex[defender_id]) else 1
        self.assertEqual(int(state.prizes_taken[0, 0]), expected_prizes)


if __name__ == "__main__":
    unittest.main()
