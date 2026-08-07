"""モデル重みに、それを生成した学習regime向けのpolicy_temperatureを埋め込んで保存/読込する。

背景(分離の理由):
    このリポジトリには「模倣学習」(train_imitation.py。cross_entropyで実際に
    選ばれた手を分類する)と「自己対戦学習」(train.py・run_train_round_robin.py・
    generation_az.py等。HuberLossでMCTSのQ優位度(clamp±1の小さい値)を回帰する)の
    2つの学習regimeがある。両者はpolicyヘッドの出力スケールが大きく異なるため、
    推論側(rl_mcts.mcts)がpriorへ変換する際のsoftmax温度も本来regimeごとに
    別の値が必要になる。

    従来はこの温度が mcts.py に `exp(policy * 10.0)` として固定でハードコードされ、
    自己対戦の重みには合っていたが模倣の重みに使うとpolicyヘッドが飽和し、
    MCTSの探索が実質1〜2手にしか広がらなくなる不具合があった。

    この温度を「呼び出し側が推測する値」ではなく「重みファイル自身が申告する値」に
    することで、どの学習regimeの重みかをRlMctsAgent側が気にせず正しく扱えるようにする。

新形式(このモジュールで保存したファイル)は次の辞書:
    {"format": "rl_mcts_checkpoint_v1", "state_dict": <model.state_dict()>,
     "policy_temperature": <float>}

旧形式(このモジュール導入前に保存された、生のstate_dictそのもの)も引き続き読める。
その場合 policy_temperature は None を返すので、呼び出し側で明示引数か既定値へ
フォールバックすること(既存の全チェックポイント(gen_000〜gen_010、
imitation_group0/1/2、agents/rl_mcts/src/model.pth等)はすべて旧形式)。
"""

from __future__ import annotations

from pathlib import Path

import torch

_FORMAT_TAG = "rl_mcts_checkpoint_v1"


def save_checkpoint(model: torch.nn.Module, path: Path, policy_temperature: float) -> None:
    """model.state_dict()を、policy_temperatureを添えて保存する。"""
    torch.save(
        {
            "format": _FORMAT_TAG,
            "state_dict": model.state_dict(),
            "policy_temperature": float(policy_temperature),
        },
        path,
    )


def load_state_dict_and_temperature(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[dict, float | None]:
    """重みファイルを読み、(state_dict, policy_temperature)を返す。

    新形式ならファイルに埋め込まれた温度を返す。旧形式(生state_dict)は
    policy_temperature=Noneを返す(呼び出し側で既定値にフォールバックする)。
    """
    loaded = torch.load(path, map_location=map_location, weights_only=True)
    if (
        isinstance(loaded, dict)
        and loaded.get("format") == _FORMAT_TAG
        and "state_dict" in loaded
    ):
        return loaded["state_dict"], loaded.get("policy_temperature")
    return loaded, None
