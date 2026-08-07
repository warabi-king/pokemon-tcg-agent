from __future__ import annotations

from pathlib import Path

import torch

from cg.api import Observation, to_observation_class
from rl_mcts.deck import read_deck_csv
from rl_mcts.mcts import mcts_agent
from rl_mcts.model import MyModel, create_model
from rl_mcts.opponent_hand import OpponentBelief


class RlMctsAgent:
    """学習済みモデルを使ってMCTSで手を選ぶagent。

    opponent_model_pathを指定すると、探索木の中で相手の手番のノード評価だけ
    別モデル(「相手はこう指すはず」という専用モデル)を使う。省略時(None)は
    これまで通りmodelを自分/相手の両方に使う(既存コードと完全互換)。

    相手の非公開カード(手札・山札・サイド・裏向きばけポケ)は、公開情報と
    ログから相手デッキ候補・確定手札を推定し、学習済み手札保持モデルで
    残存確率を補正したbelief_samples個のhidden-state粒子として具体化する
    (rl_mcts.opponent_hand.OpponentBelief)。粒子ごとに独立した完全情報MCTSを
    行い、root訪問数を合算して最終手を選ぶ。
    """

    def __init__(
        self,
        model_path: Path | None = None,
        search_count: int = 50,
        opponent_model_path: Path | None = None,
        belief_samples: int = 3,
    ) -> None:
        src_root = Path(__file__).resolve().parents[1]
        self.model_path = model_path or src_root / "model.pth"
        self.opponent_model_path = opponent_model_path
        self.search_count = search_count
        self.belief_samples = belief_samples
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
        belief = self.get_belief()
        deck = read_deck_csv()
        determinizations = belief.sample(obs, deck, self.belief_samples)

        with torch.inference_mode():
            selected, _ = mcts_agent(
                obs_dict,
                deck,
                model,
                search_count=self.search_count,
                opponent_model=opponent_model,
                determinizations=determinizations,
            )
        return selected

    def get_belief(self) -> OpponentBelief:
        """相手デッキ候補DBと手札保持モデルを初回利用時だけ読み込む。"""
        if self.belief is None:
            self.belief = OpponentBelief()
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

    def get_opponent_model(self) -> MyModel | None:
        """相手用モデルを遅延読み込みする。未指定ならNone(=自分のモデルを使う)を返す。"""
        if self.opponent_model_path is None:
            return None
        if self.opponent_model is not None:
            return self.opponent_model
        if not self.opponent_model_path.exists():
            raise FileNotFoundError(f"opponent_model_path が見つかりません: {self.opponent_model_path}")

        model = create_model()
        state = torch.load(self.opponent_model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state)
        model.eval()
        self.opponent_model = model
        return self.opponent_model
