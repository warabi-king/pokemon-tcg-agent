from __future__ import annotations

from pathlib import Path

import torch

from cg.api import Observation, to_observation_class
from rl_mcts.deck import read_deck_csv
from rl_mcts.mcts import mcts_agent
from rl_mcts.model import MyModel, create_model
from rl_mcts.opponent_deck import OpponentDeckPredictor


class RlMctsAgent:
    """学習済みモデルを使ってMCTSで手を選ぶagent。"""

    def __init__(
        self,
        model_path: Path | None = None,
        search_count: int = 10,
        opponent_model_path: Path | None = None,
    ) -> None:
        src_root = Path(__file__).resolve().parents[1]
        self.model_path = model_path or src_root / "model.pth"
        self.search_count = search_count
        self.model: MyModel | None = None
        self.opponent_predictor = OpponentDeckPredictor(
            opponent_model_path or src_root / "opponent_deck_mlp.pth"
        )

    def reset(self) -> None:
        """新しい対戦に向けて、累積した相手の公開カードを破棄する。"""
        self.opponent_predictor.reset()

    def select_action(self, obs_dict: dict) -> list[int]:
        """現在局面から合法手を選ぶ。"""
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv()

        if len(obs.select.option) == 0 or obs.select.maxCount == 0:
            return []

        model = self.get_model()
        opponent_cards = None
        if self.opponent_predictor.available:
            opponent_cards = self.opponent_predictor.predict(obs)

        with torch.inference_mode():
            selected, _ = mcts_agent(
                obs_dict,
                read_deck_csv(),
                model,
                search_count=self.search_count,
                opponent_cards=opponent_cards,
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
