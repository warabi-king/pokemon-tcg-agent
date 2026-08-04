"""信念更新、複数決定化MCTS、合意形成を統合する意思決定層。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any

from rl_mcts.belief import (
    DeckBeliefError,
    sample_hidden_zones,
    update_seen_opponent_cards,
)
from rl_mcts.mcts import LearnSample, mcts_agent
from rl_mcts.model import MyModel


@dataclass(frozen=True)
class DecisionDiagnostics:
    """直近判断の信念sample数、合意率、fallback理由を保持する。"""

    requested_determinizations: int
    successful_determinizations: int
    consensus_rate: float
    votes: dict[tuple[int, ...], int]
    fallback_reason: str | None


class BeliefPuctDecisionEngine:
    """観測履歴を保持し、複数の整合状態でMCTSを実行する。"""

    def __init__(
        self,
        determinizations: int = 3,
        search_count_per_determinization: int = 8,
    ) -> None:
        """決定化数と各決定化の探索回数を固定して初期化する。"""

        if determinizations <= 0:
            raise ValueError("determinizationsは1以上である必要があります")
        if search_count_per_determinization <= 0:
            raise ValueError("search_count_per_determinizationは1以上である必要があります")
        self.determinizations = determinizations
        self.search_count_per_determinization = search_count_per_determinization
        self.seen_by_viewer: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self.last_diagnostics: DecisionDiagnostics | None = None

    def reset_match(self) -> None:
        """新しい試合の開始時に相手カード観測履歴と診断を消去する。"""

        self.seen_by_viewer = ({}, {})
        self.last_diagnostics = None

    def _seed(self, observation: dict[str, Any]) -> int:
        """公開情報だけから再現可能な局面seedを作る。"""

        current = observation.get("current") or {}
        seed_payload = {
            "turn": current.get("turn"),
            "viewer": current.get("yourIndex"),
            "first": current.get("firstPlayer"),
            "step": observation.get("step"),
            "logs": len(observation.get("logs") or []),
            "select": (observation.get("select") or {}).get("context"),
        }
        digest = hashlib.sha256(
            json.dumps(seed_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def _seen_opponent_cards(self, your_index: int) -> list[int]:
        """現在のviewerが累積観測した相手カードIDを返す。"""

        return list(self.seen_by_viewer[your_index].values())

    def select_action(
        self,
        observation: dict[str, Any],
        your_deck: list[int],
        model: MyModel,
        opponent_model: MyModel | None = None,
    ) -> tuple[list[int], LearnSample | None]:
        """複数の非公開状態で探索し、最も合意された合法手を返す。

        信念復元に失敗した場合だけ単一の互換MCTSへ縮退し、失敗理由を診断へ
        残す。複数決定化の票が割れるほどconsensus_rateが低くなり、戦略上の
        不確実性指標として利用できる。
        """

        current = observation.get("current") or {}
        your_index = int(current.get("yourIndex", -1))
        if your_index not in (0, 1):
            raise ValueError("observation.current.yourIndexが不正です")

        update_seen_opponent_cards(observation, self.seen_by_viewer)
        base_seed = self._seed(observation)
        actions: list[tuple[int, ...]] = []
        first_sample: LearnSample | None = None
        errors: list[str] = []

        for determinization_index in range(self.determinizations):
            rng = random.Random(base_seed + determinization_index)
            try:
                hidden_zones = sample_hidden_zones(
                    observation,
                    your_index,
                    your_deck,
                    self._seen_opponent_cards(your_index),
                    rng=rng,
                )
                action, sample = mcts_agent(
                    observation,
                    your_deck,
                    model,
                    search_count=self.search_count_per_determinization,
                    hidden_zones=hidden_zones,
                    rng=rng,
                    opponent_model=opponent_model,
                )
                actions.append(tuple(action))
                if first_sample is None:
                    first_sample = sample
            except (DeckBeliefError, IndexError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")

        if not actions:
            fallback_rng = random.Random(base_seed)
            action, sample = mcts_agent(
                observation,
                your_deck,
                model,
                search_count=self.search_count_per_determinization,
                rng=fallback_rng,
                opponent_model=opponent_model,
            )
            self.last_diagnostics = DecisionDiagnostics(
                requested_determinizations=self.determinizations,
                successful_determinizations=0,
                consensus_rate=0.0,
                votes={tuple(action): 1},
                fallback_reason=" | ".join(errors) or "信念sampleを生成できませんでした",
            )
            return action, sample

        votes = Counter(actions)
        # 同票ではoption index列の辞書順を使い、実行ごとの揺れを避ける。
        selected = min(votes, key=lambda action: (-votes[action], action))
        self.last_diagnostics = DecisionDiagnostics(
            requested_determinizations=self.determinizations,
            successful_determinizations=len(actions),
            consensus_rate=votes[selected] / len(actions),
            votes=dict(votes),
            fallback_reason=" | ".join(errors) if errors else None,
        )
        return list(selected), first_sample
