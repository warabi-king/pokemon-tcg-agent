"""MCTSによる探索、手選択、学習サンプル生成。"""

from __future__ import annotations

import math
import random

import torch

from cg.api import SearchState, search_begin, search_end, search_step, to_observation_class
from rl_mcts.features import SparseVector, get_decoder_input, get_encoder_input
from rl_mcts.model import MyModel

SEARCH_COUNT = 10
MAX_ACTIONS = 64


class LearnSample:
    """学習データ1件。"""

    def __init__(
        self,
        value: float,
        policy: list[float],
        sv_enc: SparseVector,
        sv_dec: SparseVector,
    ) -> None:
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class LearnInput:
    """複数のSparseVectorをEmbeddingBag用のバッチ入力へ結合する。"""

    def __init__(self) -> None:
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, sv: SparseVector) -> None:
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for offset in sv.offset:
            self.offset.append(offset + count)


class Child:
    """MCTSノードから見た候補手。"""

    def __init__(self, select: list[int], prob: float) -> None:
        self.node: Node | None = None
        self.select = select
        self.prob = prob


class Node:
    """MCTSの1局面を表すノード。"""

    def __init__(self, parent: "Node | None", state: SearchState) -> None:
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value: float) -> None:
        """探索結果の評価値を親方向へ伝播する。"""
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def enumerate_actions(option_count: int, select_count: int, limit: int = MAX_ACTIONS) -> list[list[int]]:
    """合法option indexの組み合わせを最大limit個まで列挙する。"""
    if select_count <= 0:
        return [[]]
    if option_count <= 0 or select_count > option_count:
        return []

    actions: list[list[int]] = []
    indices = list(range(select_count))
    for _ in range(limit):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < option_count - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def eval_nn(
    sv_enc: SparseVector,
    sv_dec: SparseVector,
    model: MyModel,
) -> tuple[float, list[float]]:
    """ニューラルネットワークで局面価値と候補手スコアを評価する。"""
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )
    return value.tolist()[0][0], policy.tolist()[0]


def create_node(
    parent: Node | None,
    search_state: SearchState,
    your_index: int,
    your_deck: list[int],
    model: MyModel,
) -> tuple[Node, LearnSample | None]:
    """探索状態からMCTSノードを作り、必要ならNN評価と学習サンプルを作る。"""
    node = Node(parent, search_state)

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        return node, None

    actions = enumerate_actions(len(obs.select.option), obs.select.maxCount)
    if not actions:
        node.value = 0
        node.backprop(node.value)
        return node, None

    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    value, policy = eval_nn(sv_enc, sv_dec, model)
    v = value
    if state.yourIndex != your_index:
        v = -v
    node.value = v
    node.backprop(v)

    prob_sum = 0.0
    for i in range(len(policy)):
        p = math.exp(policy[i] * 4.0)
        node.children.append(Child(actions[i], p))
        prob_sum += p
    if prob_sum > 0:
        for child in node.children:
            child.prob /= prob_sum

    return node, LearnSample(value, policy, sv_enc, sv_dec)


def mcts_agent(
    obs_dict: dict,
    your_deck: list[int],
    model: MyModel,
    search_count: int = SEARCH_COUNT,
) -> tuple[list[int], LearnSample | None]:
    """MCTSで手を選び、root局面の学習サンプルを返す。"""
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return your_deck, None

    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=random.sample(your_deck, min(len(your_deck), state.players[your_index].deckCount)),
        your_prize=random.sample(your_deck, min(len(your_deck), len(state.players[your_index].prize))),
        opponent_deck=[1072] * state.players[1 - your_index].deckCount,
        opponent_prize=[1] * len(state.players[1 - your_index].prize),
        opponent_hand=[1] * state.players[1 - your_index].handCount,
        opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
    )

    try:
        root, sample = create_node(None, search_state, your_index, your_deck, model)
        if not root.children:
            return random.sample(list(range(len(obs.select.option))), obs.select.maxCount), sample

        for _ in range(search_count):
            current = root
            while True:
                best_value = -1e9
                best_child: Child | None = None
                c = 0.4 * math.sqrt(max(current.visit, 1))
                for child in current.children:
                    visit = 0
                    if child.node is None:
                        v = current.total / max(current.visit, 1)
                    else:
                        v = child.node.total / max(child.node.visit, 1)
                        visit = child.node.visit
                    if current.state.observation.current.yourIndex != your_index:
                        v = -v
                    v += c * child.prob / (1 + visit)
                    if best_value < v:
                        best_value = v
                        best_child = child

                if best_child is None:
                    break

                if best_child.node is None:
                    next_state = search_step(current.state.searchId, best_child.select)
                    best_child.node, _ = create_node(current, next_state, your_index, your_deck, model)
                    break

                current = best_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

        # VISIT_TIE_BREAK_V1
        # search_count=10ではPUCTの探索項が1回のvisitで半減するため、探索は事実上
        # 合法手への幅優先スイープになり、visit数が同数で並ぶ。2026-08-05計測(305局面):
        # 最多visitが同数になる局面が80.1%、並ぶ子の数は平均4.42。
        # 元実装は厳密な "<" だったので、その80%の局面で「合法手の列挙順で最初の手」が
        # 選ばれていた(enumerate_actionsは組合せを辞書順に返すため、常にoption列の先頭)。
        # 同数のときはQ値(同値ならNN prior)で解く。同一重み・10,240試合の直接対戦で
        # +7.4pt(deck対応づけ検定 t(15)=+2.97, p=0.0096)。
        # visit数が多い子を優先する規則自体は変えていない(tieのときだけの判定)。
        max_child: Child | None = None
        best_key: tuple[int, float, float] | None = None
        min_value = 10.0
        flip_value = root.state.observation.current.yourIndex != your_index
        for child in root.children:
            if child.node is None:
                continue
            v = child.node.total / max(child.node.visit, 1)
            if min_value > v:
                min_value = v
            quality = -v if flip_value else v
            key = (child.node.visit, quality, child.prob)
            if best_key is None or key > best_key:
                best_key = key
                max_child = child

        if max_child is None:
            max_child = max(root.children, key=lambda child: child.prob)
            min_value = root.total / max(root.visit, 1)

        if sample is not None:
            sample.value = root.total / max(root.visit, 1)
            for i, child in enumerate(root.children):
                v = sample.value
                if child.node is None:
                    v = min_value - v - 0.03
                else:
                    v = child.node.total / max(child.node.visit, 1) - v
                sample.policy[i] = max(-1.0, min(1.0, v))

        return max_child.select, sample
    finally:
        search_end()
