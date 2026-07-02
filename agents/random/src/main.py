import os
import random

from cg.api import Observation, to_observation_class


def read_deck_csv() -> list[int]:
    """deck.csvを読み込む。

    Returns:
        list[int]: デッキに含まれるカードIDのリスト。
    """
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def agent(obs_dict: dict) -> list[int]:
    """ポケモンカードゲームのエージェント本体。

    返すリストの各要素は、0以上かつlen(obs.select.option)未満である必要がある。
    リスト長はobs.select.minCount以上obs.select.maxCount以下で、重複要素は不可。

    Returns:
        list[int]: 選択するoption indexのリスト。
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # 初期選択ではobs.selectがNoneになり、デッキを返す必要がある。
        # デッキは60枚のカードIDリスト。
        # デッキはポケモンカードゲームのルールに準拠している必要がある。
        return read_deck_csv()

    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)  # ランダムに選択
