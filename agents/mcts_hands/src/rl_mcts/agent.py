from __future__ import annotations

from pathlib import Path

import torch

from cg.api import Observation, to_observation_class
from rl_mcts.deck import read_deck_csv
from rl_mcts.mcts import mcts_agent
from rl_mcts.model import MyModel, create_model
from rl_mcts.opponent_hand import OpponentBelief


class RlMctsAgent:
    """学習済みモデルを使ってMCTSで手を選ぶagent。"""

    def __init__(
        self,
        model_path: Path | None = None,
        opponent_model_path: Path | None = None,
        search_count: int = 10,
        belief_samples: int = 3,
        use_hand_model: bool = True,
        hand_model_path: Path | None = None,
        deck_database_path: Path | None = None,
        belief_seed: int | None = None,
    ) -> None:
        src_root = Path(__file__).resolve().parents[1]
        self.model_path = model_path or src_root / "model.pth"
        self.opponent_model_path = (
            opponent_model_path or src_root / "opponent_model.pth"
        )
        self.search_count = search_count
        self.belief_samples = belief_samples
        self.use_hand_model = use_hand_model
        self.hand_model_path = hand_model_path
        self.deck_database_path = deck_database_path
        self.belief_seed = belief_seed
        self.model: MyModel | None = None
        self.opponent_model: MyModel | None = None
        self.belief: OpponentBelief | None = None

    def select_action(self, obs_dict: dict) -> list[int]:
        """現在局面から合法手を選ぶ。"""
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            if self.belief is not None:
                self.belief.reset()
            return read_deck_csv()

        if len(obs.select.option) == 0 or obs.select.maxCount == 0:
            self.get_belief().observe(obs)
            return []

        model = self.get_model()
        opponent_model = self.get_opponent_model()
        deck = read_deck_csv()
        determinizations = self.get_belief().sample(
            obs,
            deck,
            self.belief_samples,
        )

        with torch.inference_mode():
            selected, _ = mcts_agent(
                obs_dict,
                deck,
                model,
                opponent_model=opponent_model,
                search_count=self.search_count,
                determinizations=determinizations,
            )
        return selected

    def get_belief(self) -> OpponentBelief:
        """対戦開始後にデッキDBと手札モデルを遅延読み込みする。"""

        if self.belief is None:
            self.belief = OpponentBelief(
                seed=self.belief_seed,
                use_hand_model=self.use_hand_model,
                hand_model_path=self.hand_model_path,
                database_path=self.deck_database_path,
            )
        return self.belief

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

    def get_opponent_model(self) -> MyModel:
        """探索内の相手ターン評価用モデルを遅延読み込みする。"""
        if self.opponent_model is not None:
            return self.opponent_model
        if not self.opponent_model_path.exists():
            return self.get_model()

        model = create_model()
        state = torch.load(
            self.opponent_model_path,
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(state)
        model.eval()
        self.opponent_model = model
        return self.opponent_model
