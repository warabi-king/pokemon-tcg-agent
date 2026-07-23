"""MCTSによる探索、手選択、学習サンプル生成。"""

from __future__ import annotations

import math
import random

import torch

from cg.api import SearchState, search_begin, search_end, search_step, to_observation_class
from rl_mcts.features import SparseVector, get_decoder_input, get_encoder_input
from rl_mcts.model import MyModel

SEARCH_COUNT = 10
WORLD_COUNT = 10
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
        p = math.exp(policy[i] * 10.0)
        node.children.append(Child(actions[i], p))
        prob_sum += p
    if prob_sum > 0:
        for child in node.children:
            child.prob /= prob_sum

    return node, LearnSample(value, policy, sv_enc, sv_dec)


def search_one_world(
    obs: "Observation",
    your_deck: list[int],
    model: MyModel,
    your_index: int,
    search_count: int,
) -> tuple[Node, LearnSample | None]:
    """隠れ情報を1通りに決定化(determinize)し、その1世界でMCTS探索を行う。

    毎回 search_begin を引き直すため、呼び出しごとに異なる世界線でツリーを張る。
    集計に使う root.children の訪問数・累計値だけを利用するので、探索終了後
    (search_end 後) でも安全に統計を読める。
    """
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
            return root, sample

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

        return root, sample
    finally:
        search_end()


def mcts_agent(
    obs_dict: dict,
    your_deck: list[int],
    model: MyModel,
    search_count: int = SEARCH_COUNT,
    world_count: int = WORLD_COUNT,
) -> tuple[list[int], LearnSample | None]:
    """情報集合MCTS(ISMCTS)風に複数世界を張り、root統計を平均して手を選ぶ。

    world_count 個の決定化世界でそれぞれ search_count 回の探索を行い、root候補手ごとに
    訪問数・累計値を世界横断で集計する。ばらつく隠れ情報を平均化した最善手を選ぶ。
    root観測は全世界で共通なので、children は同順・同数で index を揃えて集計できる。
    """
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return your_deck, None

    your_index = obs.current.yourIndex

    child_count = 0
    ref_children: list[Child] = []          # select/prob は世界非依存なので参照用に保持
    agg_visit: list[int] = []               # 候補手ごとの総訪問数(全世界合算)
    agg_total: list[float] = []             # 候補手ごとの累計評価値(全世界合算)
    root_visit = 0
    root_total = 0.0
    sample: LearnSample | None = None

    for _ in range(max(world_count, 1)):
        root, s = search_one_world(obs, your_deck, model, your_index, search_count)
        if sample is None and s is not None:
            sample = s
        if not root.children:
            continue
        if not ref_children:
            ref_children = root.children
            child_count = len(root.children)
            agg_visit = [0] * child_count
            agg_total = [0.0] * child_count
        root_visit += root.visit
        root_total += root.total
        for i, child in enumerate(root.children):
            if child.node is not None:
                agg_visit[i] += child.node.visit
                agg_total[i] += child.node.total

    if not ref_children:
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount), sample

    # 集計訪問数が最大の手を選ぶ。どの世界でも展開されなかった場合はNN事前分布で代替。
    best_i = max(range(child_count), key=lambda i: agg_visit[i])
    if agg_visit[best_i] <= 0:
        best_i = max(range(child_count), key=lambda i: ref_children[i].prob)

    if sample is not None:
        value = root_total / max(root_visit, 1)
        min_value = 10.0
        for i in range(child_count):
            if agg_visit[i] <= 0:
                continue
            v = agg_total[i] / agg_visit[i]
            if min_value > v:
                min_value = v
        if min_value == 10.0:
            min_value = value

        sample.value = value
        for i in range(child_count):
            if agg_visit[i] <= 0:
                v = min_value - value - 0.03
            else:
                v = agg_total[i] / agg_visit[i] - value
            sample.policy[i] = max(-1.0, min(1.0, v))

    return ref_children[best_i].select, sample
