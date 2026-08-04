"""提出用agentとしてモデル読込みとbelief-aware PUCTを接続する。"""

from __future__ import annotations

from pathlib import Path

import torch

from cg.api import Observation, to_observation_class
from rl_mcts.deck import read_deck_csv
from rl_mcts.decision import BeliefPuctDecisionEngine
from rl_mcts.model import MyModel, create_model


class RlMctsAgent:
    """信念推定と設定可能な決定化数のPUCTで手を選ぶ提出用agent。"""

    def __init__(
        self,
        model_path: Path | None = None,
        search_count: int = 24,
        determinizations: int = 1,
        opponent_model_path: Path | None = None,
    ) -> None:
        """自分・相手モデルpath、各探索回数、決定化数を設定する。"""

        src_root = Path(__file__).resolve().parents[1]
        self.model_path = model_path or src_root / "model.pth"
        self.opponent_model_path = opponent_model_path
        self.search_count = search_count
        self.model: MyModel | None = None
        self.opponent_model: MyModel | None = None
        self.decision_engine = BeliefPuctDecisionEngine(
            determinizations=determinizations,
            search_count_per_determinization=search_count,
        )

    def reset_match(self) -> None:
        """新規対戦前に試合を跨いではならないbelief履歴を初期化する。"""

        self.decision_engine.reset_match()

    def select_action(self, obs_dict: dict) -> list[int]:
        """現在局面から合法手を選ぶ。"""
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()

        if len(obs.select.option) == 0 or obs.select.maxCount == 0:
            return []

        model = self.get_model()
        opponent_model = self.get_opponent_model()

        with torch.inference_mode():
            selected, _ = self.decision_engine.select_action(
                obs_dict,
                read_deck_csv(),
                model,
                opponent_model,
            )
        return selected

    def get_model(self) -> MyModel:
        """学習済みモデルを遅延読み込みする。"""
        if self.model is not None:
            return self.model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"rl_mcts_sample requires a trained model, but model.pth was not found: "
                f"{self.model_path}"
            )

        model = create_model()
        state = torch.load(self.model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()
        self.model = model
        return self.model

    def get_opponent_model(self) -> MyModel | None:
        """相手手番専用checkpointを遅延読込みし、未指定ならNoneを返す。"""

        if self.opponent_model_path is None:
            return None
        if self.opponent_model is not None:
            return self.opponent_model
        if not self.opponent_model_path.exists():
            raise FileNotFoundError(
                "相手手番モデルが見つかりません: "
                f"{self.opponent_model_path}"
            )
        model = create_model()
        state = torch.load(self.opponent_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()
        self.opponent_model = model
        return self.opponent_model
