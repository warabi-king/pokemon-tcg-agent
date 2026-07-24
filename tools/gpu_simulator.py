"""全cabtカードをTensor化するGPUルールエンジンの検証基盤。

このモジュールは、cabt SDKが公開する全カード・全ワザの静的データをGPU上の
lookup tableへ変換する。また、ワザの基礎ダメージ、弱点、抵抗力、きぜつ、
サイド取得と、機械的に解釈できる一部の効果をバッチ適用できる。

重要:
    cabt SDKが公開するカード効果は自然言語テキストだけであり、実行可能な
    opcodeは公開されていない。したがって、このファイルは現時点では完全な
    cabt代替ではない。``audit``で、GPU命令へ正確に変換できる効果と未変換の
    効果を明示する。未変換効果を暗黙に無視して学習へ使わないこと。

使用例:
    .venv/bin/python tools/gpu_simulator.py audit --device mps
    .venv/bin/python tools/gpu_simulator.py benchmark --device mps --batch-size 4096
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_SRC = ROOT / "agents" / "rl_mcts_r_robin1" / "src"


class EffectOpcode(IntEnum):
    """GPU上で直接扱える効果命令。0は効果なし、-1は未対応。"""

    UNSUPPORTED = -1
    NONE = 0
    DRAW = 1
    SELF_DAMAGE = 2
    APPLY_POISON = 3
    APPLY_BURN = 4
    APPLY_CONFUSION = 5
    APPLY_PARALYSIS = 6
    DISCARD_ENERGY = 7
    DISCARD_ALL_ENERGY = 8
    CANNOT_ATTACK_NEXT_TURN = 9
    CANNOT_RETREAT_NEXT_TURN = 10
    SWITCH_SELF = 11


KERNEL_EXECUTABLE_OPCODES = frozenset(
    {
        EffectOpcode.NONE,
        EffectOpcode.SELF_DAMAGE,
        EffectOpcode.APPLY_POISON,
        EffectOpcode.APPLY_BURN,
        EffectOpcode.APPLY_CONFUSION,
        EffectOpcode.APPLY_PARALYSIS,
    }
)


@dataclass(frozen=True)
class CompiledEffect:
    opcode: EffectOpcode
    argument: int = 0

    @property
    def supported(self) -> bool:
        return self.opcode is not EffectOpcode.UNSUPPORTED


_DRAW_RE = re.compile(r"Draw (?:a|(?P<count>\d+)) cards?\.", re.IGNORECASE)
_SELF_DAMAGE_RE = re.compile(
    r"This Pok.mon also does (?P<count>\d+) damage to itself\.", re.IGNORECASE
)
_DISCARD_ENERGY_RE = re.compile(
    r"Discard (?:(?:an|1)|(?P<count>\d+)) Energy from this Pok.mon\.", re.IGNORECASE
)


def normalize_effect_text(text: str) -> str:
    """改行・Unicode空白を揃え、完全一致パターンで安全に判定できる形にする。"""
    return " ".join(text.replace("\xa0", " ").split())


def compile_effect_text(text: str) -> CompiledEffect:
    """単一の既知効果テキストをGPU命令へ変換する。

    複数効果を含む文章や条件付き効果は、意味を変えてしまわないよう
    ``UNSUPPORTED``にする。
    """
    normalized = normalize_effect_text(text)
    if not normalized:
        return CompiledEffect(EffectOpcode.NONE)

    match = _DRAW_RE.fullmatch(normalized)
    if match:
        return CompiledEffect(EffectOpcode.DRAW, int(match.group("count") or 1))

    match = _SELF_DAMAGE_RE.fullmatch(normalized)
    if match:
        return CompiledEffect(EffectOpcode.SELF_DAMAGE, int(match.group("count")))

    status_effects = {
        "Your opponent’s Active Pokémon is now Poisoned.": EffectOpcode.APPLY_POISON,
        "Your opponent's Active Pokémon is now Poisoned.": EffectOpcode.APPLY_POISON,
        "Your opponent’s Active Pokémon is now Burned.": EffectOpcode.APPLY_BURN,
        "Your opponent's Active Pokémon is now Burned.": EffectOpcode.APPLY_BURN,
        "Your opponent’s Active Pokémon is now Confused.": EffectOpcode.APPLY_CONFUSION,
        "Your opponent's Active Pokémon is now Confused.": EffectOpcode.APPLY_CONFUSION,
        "Your opponent’s Active Pokémon is now Paralyzed.": EffectOpcode.APPLY_PARALYSIS,
        "Your opponent's Active Pokémon is now Paralyzed.": EffectOpcode.APPLY_PARALYSIS,
    }
    if normalized in status_effects:
        return CompiledEffect(status_effects[normalized])

    match = _DISCARD_ENERGY_RE.fullmatch(normalized)
    if match:
        return CompiledEffect(
            EffectOpcode.DISCARD_ENERGY,
            int(match.group("count") or 1),
        )

    exact_effects = {
        "Discard all Energy from this Pokémon.": EffectOpcode.DISCARD_ALL_ENERGY,
        "During your next turn, this Pokémon can’t use attacks.": EffectOpcode.CANNOT_ATTACK_NEXT_TURN,
        "During your opponent’s next turn, the Defending Pokémon can’t retreat.": EffectOpcode.CANNOT_RETREAT_NEXT_TURN,
        "Switch this Pokémon with 1 of your Benched Pokémon.": EffectOpcode.SWITCH_SELF,
    }
    opcode = exact_effects.get(normalized)
    if opcode is not None:
        return CompiledEffect(opcode)
    return CompiledEffect(EffectOpcode.UNSUPPORTED)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAを利用できません。")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPSを利用できません。")
    return device


def _load_cg_api(agent_src: Path):
    resolved = agent_src.resolve()
    if not (resolved / "cg" / "api.py").exists():
        raise FileNotFoundError(f"cg/api.pyが見つかりません: {resolved}")
    sys.path.insert(0, str(resolved))
    try:
        from cg.api import CardType, all_attack, all_card_data
    finally:
        try:
            sys.path.remove(str(resolved))
        except ValueError:
            pass
    return CardType, all_card_data(), all_attack()


@dataclass
class TensorCardDatabase:
    """全カード・全ワザをIDで参照するdevice常駐Tensor。"""

    card_exists: torch.Tensor
    card_type: torch.Tensor
    hp: torch.Tensor
    retreat_cost: torch.Tensor
    weakness: torch.Tensor
    resistance: torch.Tensor
    energy_type: torch.Tensor
    is_basic: torch.Tensor
    is_stage1: torch.Tensor
    is_stage2: torch.Tensor
    is_ex: torch.Tensor
    is_mega_ex: torch.Tensor
    is_tera: torch.Tensor
    is_ace_spec: torch.Tensor
    card_attacks: torch.Tensor
    card_attack_count: torch.Tensor
    attack_exists: torch.Tensor
    attack_damage: torch.Tensor
    attack_energy: torch.Tensor
    attack_energy_count: torch.Tensor
    attack_owner: torch.Tensor
    attack_effect_opcode: torch.Tensor
    attack_effect_argument: torch.Tensor
    attack_effect_supported: torch.Tensor
    attack_effect_executable: torch.Tensor
    card_count: int
    attack_count: int
    skill_count: int
    supported_skill_count: int
    attack_effect_count: int
    supported_attack_effect_count: int
    executable_attack_effect_count: int
    exact_static_card_count: int

    @property
    def device(self) -> torch.device:
        return self.card_exists.device

    @classmethod
    def from_cg(
        cls,
        agent_src: Path = DEFAULT_AGENT_SRC,
        device: torch.device | str = "cpu",
    ) -> "TensorCardDatabase":
        CardType, cards, attacks = _load_cg_api(agent_src)
        max_card_id = max(card.cardId for card in cards)
        max_attack_id = max(attack.attackId for attack in attacks)
        max_attacks_per_card = max(len(card.attacks) for card in cards)
        max_energy_cost = max(len(attack.energies) for attack in attacks)

        def full_card(value: int, *, dtype=torch.int64) -> torch.Tensor:
            return torch.full((max_card_id + 1,), value, dtype=dtype)

        def full_attack(value: int, *, dtype=torch.int64) -> torch.Tensor:
            return torch.full((max_attack_id + 1,), value, dtype=dtype)

        values: dict[str, Any] = {
            "card_exists": full_card(0, dtype=torch.bool),
            "card_type": full_card(-1),
            "hp": full_card(0),
            "retreat_cost": full_card(0),
            "weakness": full_card(-1),
            "resistance": full_card(-1),
            "energy_type": full_card(-1),
            "is_basic": full_card(0, dtype=torch.bool),
            "is_stage1": full_card(0, dtype=torch.bool),
            "is_stage2": full_card(0, dtype=torch.bool),
            "is_ex": full_card(0, dtype=torch.bool),
            "is_mega_ex": full_card(0, dtype=torch.bool),
            "is_tera": full_card(0, dtype=torch.bool),
            "is_ace_spec": full_card(0, dtype=torch.bool),
            "card_attacks": torch.full(
                (max_card_id + 1, max_attacks_per_card), -1, dtype=torch.int64
            ),
            "card_attack_count": full_card(0),
            "attack_exists": full_attack(0, dtype=torch.bool),
            "attack_damage": full_attack(0),
            "attack_energy": torch.full(
                (max_attack_id + 1, max_energy_cost), -1, dtype=torch.int64
            ),
            "attack_energy_count": full_attack(0),
            "attack_owner": full_attack(-1),
            "attack_effect_opcode": full_attack(int(EffectOpcode.UNSUPPORTED)),
            "attack_effect_argument": full_attack(0),
            "attack_effect_supported": full_attack(0, dtype=torch.bool),
            "attack_effect_executable": full_attack(0, dtype=torch.bool),
        }

        attacks_by_id = {attack.attackId: attack for attack in attacks}
        skill_count = 0
        supported_skill_count = 0
        exact_static_card_count = 0
        for card in cards:
            cid = card.cardId
            values["card_exists"][cid] = True
            values["card_type"][cid] = int(card.cardType)
            values["hp"][cid] = card.hp
            values["retreat_cost"][cid] = card.retreatCost
            values["weakness"][cid] = -1 if card.weakness is None else int(card.weakness)
            values["resistance"][cid] = -1 if card.resistance is None else int(card.resistance)
            values["energy_type"][cid] = int(card.energyType)
            values["is_basic"][cid] = card.basic
            values["is_stage1"][cid] = card.stage1
            values["is_stage2"][cid] = card.stage2
            values["is_ex"][cid] = card.ex
            values["is_mega_ex"][cid] = card.megaEx
            values["is_tera"][cid] = card.tera
            values["is_ace_spec"][cid] = card.aceSpec
            values["card_attack_count"][cid] = len(card.attacks)
            if card.attacks:
                values["card_attacks"][cid, : len(card.attacks)] = torch.tensor(card.attacks)
                for attack_id in card.attacks:
                    if values["attack_owner"][attack_id] < 0:
                        values["attack_owner"][attack_id] = cid

            skill_count += len(card.skills)
            supported_skill_count += sum(
                compile_effect_text(skill.text).supported for skill in card.skills
            )
            if not card.skills and all(
                not attacks_by_id[attack_id].text.strip() for attack_id in card.attacks
            ):
                exact_static_card_count += 1

        attack_effect_count = 0
        supported_attack_effect_count = 0
        executable_attack_effect_count = 0
        for attack in attacks:
            aid = attack.attackId
            compiled = compile_effect_text(attack.text)
            values["attack_exists"][aid] = True
            values["attack_damage"][aid] = attack.damage
            values["attack_energy_count"][aid] = len(attack.energies)
            if attack.energies:
                values["attack_energy"][aid, : len(attack.energies)] = torch.tensor(
                    [int(energy) for energy in attack.energies]
                )
            values["attack_effect_opcode"][aid] = int(compiled.opcode)
            values["attack_effect_argument"][aid] = compiled.argument
            values["attack_effect_supported"][aid] = compiled.supported
            values["attack_effect_executable"][aid] = (
                compiled.opcode in KERNEL_EXECUTABLE_OPCODES
            )
            if attack.text.strip():
                attack_effect_count += 1
                supported_attack_effect_count += int(compiled.supported)
                executable_attack_effect_count += int(
                    compiled.opcode in KERNEL_EXECUTABLE_OPCODES
                )

        target_device = torch.device(device)
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(target_device)

        return cls(
            **values,
            card_count=len(cards),
            attack_count=len(attacks),
            skill_count=skill_count,
            supported_skill_count=supported_skill_count,
            attack_effect_count=attack_effect_count,
            supported_attack_effect_count=supported_attack_effect_count,
            executable_attack_effect_count=executable_attack_effect_count,
            exact_static_card_count=exact_static_card_count,
        )

    def audit(self) -> dict[str, int | float | str]:
        attack_pct = (
            100.0 * self.supported_attack_effect_count / self.attack_effect_count
            if self.attack_effect_count
            else 100.0
        )
        skill_pct = (
            100.0 * self.supported_skill_count / self.skill_count
            if self.skill_count
            else 100.0
        )
        executable_pct = (
            100.0 * self.executable_attack_effect_count / self.attack_effect_count
            if self.attack_effect_count
            else 100.0
        )
        return {
            "device": str(self.device),
            "cards_loaded": self.card_count,
            "attacks_loaded": self.attack_count,
            "attack_effects_total": self.attack_effect_count,
            "attack_effects_parsed": self.supported_attack_effect_count,
            "attack_effect_parse_coverage_percent": round(attack_pct, 2),
            "attack_effects_executable_by_kernel": self.executable_attack_effect_count,
            "attack_effect_kernel_coverage_percent": round(executable_pct, 2),
            "skills_total": self.skill_count,
            "skills_parsed": self.supported_skill_count,
            "skill_parse_coverage_percent": round(skill_pct, 2),
            "exact_static_cards": self.exact_static_card_count,
            "exact_static_card_coverage_percent": round(
                100.0 * self.exact_static_card_count / self.card_count, 2
            ),
        }


@dataclass
class AttackBatchState:
    """GPU基礎戦闘カーネル用の最小バッチ状態。"""

    active_card: torch.Tensor  # [B, 2]
    hp: torch.Tensor  # [B, 2]
    current_player: torch.Tensor  # [B]
    prizes_taken: torch.Tensor  # [B, 2]
    special_condition: torch.Tensor  # [B, 2] bit mask
    done: torch.Tensor  # [B]
    winner: torch.Tensor  # [B], -1 while playing

    @classmethod
    def create(
        cls,
        active_card: torch.Tensor,
        card_db: TensorCardDatabase,
    ) -> "AttackBatchState":
        cards = active_card.to(device=card_db.device, dtype=torch.int64)
        batch_size = cards.shape[0]
        if cards.shape != (batch_size, 2):
            raise ValueError("active_cardは[B, 2]で指定してください。")
        return cls(
            active_card=cards,
            hp=card_db.hp[cards].clone(),
            current_player=torch.zeros(batch_size, dtype=torch.int64, device=card_db.device),
            prizes_taken=torch.zeros((batch_size, 2), dtype=torch.int64, device=card_db.device),
            special_condition=torch.zeros((batch_size, 2), dtype=torch.int64, device=card_db.device),
            done=torch.zeros(batch_size, dtype=torch.bool, device=card_db.device),
            winner=torch.full((batch_size,), -1, dtype=torch.int64, device=card_db.device),
        )


class AttackBatchKernel:
    """全ワザIDを同一カーネル経路で処理するTensor実装。"""

    POISON_BIT = 1 << 0
    BURN_BIT = 1 << 1
    CONFUSION_BIT = 1 << 2
    PARALYSIS_BIT = 1 << 3

    def __init__(self, card_db: TensorCardDatabase) -> None:
        self.card_db = card_db

    def step(self, state: AttackBatchState, attack_id: torch.Tensor) -> None:
        db = self.card_db
        attacks = attack_id.to(device=db.device, dtype=torch.int64)
        batch_size = attacks.numel()
        if batch_size != state.current_player.numel():
            raise ValueError("attack_idの件数がstateのbatch sizeと一致しません。")

        row = torch.arange(batch_size, device=db.device)
        player = state.current_player
        opponent = 1 - player
        live = ~state.done
        attacker_card = state.active_card[row, player]
        defender_card = state.active_card[row, opponent]

        damage = db.attack_damage[attacks]
        attack_type = db.energy_type[attacker_card]
        weakness = db.weakness[defender_card]
        resistance = db.resistance[defender_card]
        damage = torch.where((weakness >= 0) & (weakness == attack_type), damage * 2, damage)
        damage = torch.where(
            (resistance >= 0) & (resistance == attack_type),
            torch.clamp(damage - 30, min=0),
            damage,
        )
        next_defender_hp = torch.clamp(state.hp[row, opponent] - damage, min=0)
        state.hp[row, opponent] = torch.where(
            live, next_defender_hp, state.hp[row, opponent]
        )

        opcode = db.attack_effect_opcode[attacks]
        argument = db.attack_effect_argument[attacks]
        self_damage = opcode == int(EffectOpcode.SELF_DAMAGE)
        next_attacker_hp = torch.clamp(state.hp[row, player] - argument, min=0)
        state.hp[row, player] = torch.where(
            live & self_damage, next_attacker_hp, state.hp[row, player]
        )

        status_bits = torch.zeros_like(opcode)
        status_bits = torch.where(
            opcode == int(EffectOpcode.APPLY_POISON), self.POISON_BIT, status_bits
        )
        status_bits = torch.where(
            opcode == int(EffectOpcode.APPLY_BURN), self.BURN_BIT, status_bits
        )
        status_bits = torch.where(
            opcode == int(EffectOpcode.APPLY_CONFUSION), self.CONFUSION_BIT, status_bits
        )
        status_bits = torch.where(
            opcode == int(EffectOpcode.APPLY_PARALYSIS), self.PARALYSIS_BIT, status_bits
        )
        state.special_condition[row, opponent] |= torch.where(
            live, status_bits, torch.zeros_like(status_bits)
        )

        defender_knocked_out = live & (state.hp[row, opponent] <= 0)
        attacker_knocked_out = live & (state.hp[row, player] <= 0)
        prize_value = torch.where(
            db.is_mega_ex[defender_card],
            3,
            torch.where(db.is_ex[defender_card], 2, 1),
        )
        state.prizes_taken[row, player] += torch.where(
            defender_knocked_out, prize_value, torch.zeros_like(prize_value)
        )
        defender_wins_game = defender_knocked_out & (state.prizes_taken[row, player] >= 6)
        attacker_loses_game = attacker_knocked_out & ~defender_wins_game
        newly_done = defender_wins_game | attacker_loses_game
        next_winner = torch.where(defender_wins_game, player, opponent)
        state.winner = torch.where(newly_done, next_winner, state.winner)
        state.done |= newly_done
        state.current_player = torch.where(live & ~state.done, opponent, player)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_attack_kernel(
    db: TensorCardDatabase,
    *,
    batch_size: int,
    steps: int,
) -> dict[str, float | int | str]:
    """全ワザを混在させ、基礎戦闘Tensor遷移の上限を測る。"""
    device = db.device
    valid_attacks = torch.nonzero(
        db.attack_exists & (db.attack_owner >= 0), as_tuple=False
    ).flatten()
    pokemon_cards = torch.nonzero(
        db.card_exists & (db.card_type == 0) & (db.hp > 0), as_tuple=False
    ).flatten()
    sample_index = torch.arange(batch_size, device=device) % valid_attacks.numel()
    attacks = valid_attacks[sample_index]
    attacker = db.attack_owner[attacks]
    defender = pokemon_cards[torch.arange(batch_size, device=device) % pokemon_cards.numel()]
    state = AttackBatchState.create(torch.stack((attacker, defender), dim=1), db)
    kernel = AttackBatchKernel(db)

    warmup = min(20, steps)
    measured_steps = max(1, steps)
    for _ in range(warmup):
        state.hp = db.hp[state.active_card].clone()
        state.done.zero_()
        state.current_player.zero_()
        kernel.step(state, attacks)
    synchronize(device)

    started = time.perf_counter()
    for _ in range(measured_steps):
        state.hp = db.hp[state.active_card].clone()
        state.done.zero_()
        state.current_player.zero_()
        kernel.step(state, attacks)
    synchronize(device)
    elapsed = time.perf_counter() - started
    transitions = batch_size * measured_steps
    return {
        "device": str(device),
        "batch_size": batch_size,
        "steps": measured_steps,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        "microseconds_per_batch": elapsed * 1e6 / measured_steps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "benchmark"))
    parser.add_argument("--agent-src", type=Path, default=DEFAULT_AGENT_SRC)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.steps < 1:
        raise SystemExit("--batch-sizeと--stepsは1以上で指定してください。")
    device = select_device(args.device)
    database = TensorCardDatabase.from_cg(args.agent_src, device)
    if args.command == "audit":
        result = database.audit()
    else:
        result = {
            **database.audit(),
            **benchmark_attack_kernel(
                database,
                batch_size=args.batch_size,
                steps=args.steps,
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
