"""複数試合を1プロセスで進め、MCTSのNN評価を試合横断でバッチ化する。

``kaggle_environments.env.run`` はエージェントを同期的に呼び出すため、別々の
試合で発生したNN評価要求を1回のforwardへまとめられない。このモジュールは
cabt SDKのBattle/Searchハンドルを直接使い、次のように処理する。

* 1つのlibcgを全試合で共有する。
* 複数のBattleハンドルをGPU laneに対応する進行中試合として保持する。
* 各試合のMCTSを1 simulationずつ進める。
* 同じモデルを使う葉ノードをまとめて1回のGPU forwardで評価する。

Battle/Searchのルール処理自体はCPU版libcgで行う。GPUが保持するのはNN入力と
評価処理であり、libcg側の不透明な状態はBattleポインタ/Search IDで参照する。
"""

from __future__ import annotations

from collections import defaultdict, deque
import ctypes
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import torch


MAX_ACTIONS = 64


@dataclass(frozen=True)
class BatchedGameRequest:
    name0: str
    name1: str
    game_index: int
    swap: bool


@dataclass(frozen=True)
class BatchedGameResult:
    name0: str
    name1: str
    game_index: int
    swap: bool
    result: int | None
    turns: int
    error: str | None = None


@dataclass
class BatchedProfile:
    battle_start_seconds: float = 0.0
    battle_step_seconds: float = 0.0
    search_begin_seconds: float = 0.0
    search_step_seconds: float = 0.0
    feature_seconds: float = 0.0
    nn_seconds: float = 0.0
    search_finalize_seconds: float = 0.0
    nn_evaluations: int = 0
    nn_batches: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    battle_steps: int = 0
    search_steps: int = 0

    @property
    def mean_batch_size(self) -> float:
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    @property
    def max_batch_size(self) -> int:
        return max(self.batch_sizes, default=0)


@dataclass
class BatchedTournamentOutput:
    results: list[BatchedGameResult]
    profile: BatchedProfile
    device: str


@dataclass
class _Runtime:
    lib: Any
    to_observation_class: Any
    search_begin: Any
    search_step: Any
    search_end: Any
    get_encoder_input: Any
    get_decoder_input: Any
    create_model: Any


@dataclass
class _Participant:
    name: str
    deck: list[int]
    model: torch.nn.Module | None
    model_key: str | None
    random_policy: bool


@dataclass
class _MatchSession:
    request: BatchedGameRequest
    players: tuple[_Participant, _Participant]
    battle_ptr: int
    observation: dict[str, Any]
    select_player: int
    selections: int = 0


class _Child:
    def __init__(self, select: list[int], probability: float) -> None:
        self.node: _Node | None = None
        self.select = select
        self.probability = probability
        # parallel MCTSで同じroot branchを1つのbatchへ重複投入しないための予約。
        self.in_flight = False


class _Node:
    def __init__(self, parent: _Node | None, state: Any) -> None:
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[_Child] = []
        self.state = state

    def backprop(self, value: float) -> None:
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


@dataclass
class _SearchContext:
    session: _MatchSession
    participant: _Participant
    your_index: int
    root: _Node | None = None


@dataclass
class _EvalRequest:
    context: _SearchContext
    node: _Node
    actions: list[list[int]]
    encoder: Any
    decoder: Any
    reserved_child: _Child | None = None


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


def _load_runtime(agent_src: Path) -> _Runtime:
    """1つのagent実装を型・特徴量・libcgの共通実装として読み込む。"""
    resolved = agent_src.resolve()
    if not (resolved / "cg" / "sim.py").exists():
        raise FileNotFoundError(f"cg SDKが見つかりません: {resolved}")

    # batched backendでは全agentがこの1つのcg/rl_mctsモジュールを共有する。
    # legacy backendのagentローダーとは同じプロセスで併用しない。
    sys.path.insert(0, str(resolved))
    try:
        from cg.api import search_begin, search_end, search_step, to_observation_class
        from cg.sim import lib
        from rl_mcts.features import get_decoder_input, get_encoder_input
        from rl_mcts.model import create_model
    finally:
        try:
            sys.path.remove(str(resolved))
        except ValueError:
            pass

    return _Runtime(
        lib=lib,
        to_observation_class=to_observation_class,
        search_begin=search_begin,
        search_step=search_step,
        search_end=search_end,
        get_encoder_input=get_encoder_input,
        get_decoder_input=get_decoder_input,
        create_model=create_model,
    )


def _read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csvは60枚必要です: {path} ({len(deck)}枚)")
    return deck


def _is_random_agent(agent_path: Path) -> bool:
    return agent_path.parent.parent.name == "random"


def _load_participants(
    specs: Iterable[Any],
    runtime: _Runtime,
    device: torch.device,
) -> dict[str, _Participant]:
    participants: dict[str, _Participant] = {}
    model_cache: dict[str, torch.nn.Module] = {}

    for spec in specs:
        agent_path = Path(spec.agent_path).resolve()
        deck_path = Path(spec.deck_path).resolve()
        deck = _read_deck(deck_path)
        model_path = agent_path.parent / "model.pth"

        if model_path.exists():
            model_key = str(model_path.resolve())
            model = model_cache.get(model_key)
            if model is None:
                model = runtime.create_model()
                state = torch.load(model_path, map_location=torch.device("cpu"))
                model.load_state_dict(state)
                model.eval()
                model.to(device)
                model_cache[model_key] = model
            participants[spec.name] = _Participant(
                name=spec.name,
                deck=deck,
                model=model,
                model_key=model_key,
                random_policy=False,
            )
        elif _is_random_agent(agent_path):
            participants[spec.name] = _Participant(
                name=spec.name,
                deck=deck,
                model=None,
                model_key=None,
                random_policy=True,
            )
        else:
            raise ValueError(
                "batched backendはmodel.pthを持つrl_mcts agentまたはrandom agentのみ"
                f"対応しています: {spec.name} ({agent_path})"
            )
    return participants


def _battle_observation(runtime: _Runtime, battle_ptr: int) -> tuple[dict[str, Any], int]:
    serial = runtime.lib.GetBattleData(battle_ptr)
    observation = json.loads(serial.json.decode())
    observation["search_begin_input"] = ctypes.string_at(serial.data, serial.count).decode(
        "ascii"
    )
    return observation, int(serial.selectPlayer)


def _start_session(
    runtime: _Runtime,
    request: BatchedGameRequest,
    participants: dict[str, _Participant],
    profile: BatchedProfile,
) -> _MatchSession:
    canonical0 = participants[request.name0]
    canonical1 = participants[request.name1]
    players = (canonical1, canonical0) if request.swap else (canonical0, canonical1)
    cards = players[0].deck + players[1].deck
    argument = (ctypes.c_int * len(cards))(*cards)

    started = time.perf_counter()
    start_data = runtime.lib.BattleStart(argument)
    profile.battle_start_seconds += time.perf_counter() - started
    battle_ptr = int(start_data.battlePtr or 0)
    if battle_ptr == 0:
        raise RuntimeError(
            f"BattleStart失敗: errorPlayer={start_data.errorPlayer}, "
            f"errorType={start_data.errorType}"
        )
    observation, select_player = _battle_observation(runtime, battle_ptr)
    return _MatchSession(
        request=request,
        players=players,
        battle_ptr=battle_ptr,
        observation=observation,
        select_player=select_player,
    )


def _enumerate_actions(
    option_count: int,
    select_count: int,
    limit: int = MAX_ACTIONS,
) -> list[list[int]]:
    if select_count <= 0:
        return [[]]
    if option_count <= 0 or select_count > option_count:
        return []

    actions: list[list[int]] = []
    indices = list(range(select_count))
    for _ in range(limit):
        actions.append(indices.copy())
        for reverse_index in range(len(indices)):
            index = len(indices) - reverse_index - 1
            if indices[index] < option_count - reverse_index - 1:
                indices[index] += 1
                for following in range(index + 1, len(indices)):
                    indices[following] = indices[following - 1] + 1
                break
        else:
            break
    return actions


def _pad_sparse_offsets(sparse: Any, target: int) -> None:
    """EmbeddingBagの空bagを末尾へ足し、全局面の候補手数を揃える。"""
    if len(sparse.offset) > target:
        raise ValueError(f"decoder action数が上限を超えました: {len(sparse.offset)} > {target}")
    sparse.offset.extend([len(sparse.index)] * (target - len(sparse.offset)))


def _prepare_node(
    runtime: _Runtime,
    context: _SearchContext,
    parent: _Node | None,
    search_state: Any,
    profile: BatchedProfile,
) -> tuple[_Node, _EvalRequest | None]:
    node = _Node(parent, search_state)
    observation = search_state.observation
    state = observation.current

    if state.result >= 0:
        if state.result == 2:
            node.value = 0.0
        elif state.result == context.your_index:
            node.value = 1.0
        else:
            node.value = -1.0
        node.backprop(node.value)
        return node, None

    actions = _enumerate_actions(
        len(observation.select.option),
        observation.select.maxCount,
    )
    if not actions:
        node.value = 0.0
        node.backprop(node.value)
        return node, None

    started = time.perf_counter()
    encoder = runtime.get_encoder_input(observation, context.participant.deck)
    decoder = runtime.get_decoder_input(observation, actions)
    profile.feature_seconds += time.perf_counter() - started
    return node, _EvalRequest(context, node, actions, encoder, decoder)


def _combine_sparse(vectors: list[Any]) -> tuple[list[int], list[float], list[int]]:
    indices: list[int] = []
    values: list[float] = []
    offsets: list[int] = []
    for vector in vectors:
        base = len(indices)
        indices.extend(vector.index)
        values.extend(vector.value)
        offsets.extend(offset + base for offset in vector.offset)
    return indices, values, offsets


def _apply_evaluations(
    requests: list[_EvalRequest],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
) -> None:
    grouped: dict[str, list[_EvalRequest]] = defaultdict(list)
    for request in requests:
        key = request.context.participant.model_key
        if key is None:
            raise RuntimeError("NN評価要求にモデルがありません。")
        grouped[key].append(request)

    for model_requests in grouped.values():
        model = model_requests[0].context.participant.model
        if model is None:
            raise RuntimeError("NNモデルがありません。")
        for offset in range(0, len(model_requests), batch_size):
            chunk = model_requests[offset : offset + batch_size]
            # 常にMAX_ACTIONSまでpadすると、候補が1～2個しかない選択でも
            # decoderを64 token分実行してしまう。forwardごとの最大長だけに揃える。
            required_decoder_words = max(
                len(request.decoder.offset) for request in chunk
            )
            # MPSはshapeが毎回変わるとgraph/kernel準備コストが大きいので、
            # 1,2,4,...,64のbucketへ丸める。固定64より計算量を抑えつつ、
            # 動的shapeの種類も最大7個に制限できる。
            decoder_words = 1 << (required_decoder_words - 1).bit_length()
            for request in chunk:
                _pad_sparse_offsets(request.decoder, decoder_words)
            encoder = _combine_sparse([request.encoder for request in chunk])
            decoder = _combine_sparse([request.decoder for request in chunk])

            started = time.perf_counter()
            with torch.inference_mode():
                values, policies = model(
                    torch.tensor(encoder[0], dtype=torch.int32, device=device),
                    torch.tensor(encoder[1], dtype=torch.float32, device=device),
                    torch.tensor(encoder[2], dtype=torch.int32, device=device),
                    torch.tensor(decoder[0], dtype=torch.int32, device=device),
                    torch.tensor(decoder[1], dtype=torch.float32, device=device),
                    torch.tensor(decoder[2], dtype=torch.int32, device=device),
                )
                value_rows = values.detach().cpu().tolist()
                policy_rows = policies.detach().cpu().tolist()
            profile.nn_seconds += time.perf_counter() - started
            profile.nn_evaluations += len(chunk)
            profile.nn_batches += 1
            profile.batch_sizes.append(len(chunk))

            for request, value_row, policy_row in zip(
                chunk, value_rows, policy_rows, strict=True
            ):
                value = float(value_row[0])
                state = request.node.state.observation.current
                propagated = value
                if state.yourIndex != request.context.your_index:
                    propagated = -propagated
                request.node.value = propagated
                request.node.backprop(propagated)

                probabilities = [math.exp(float(policy_row[i]) * 10.0) for i in range(len(request.actions))]
                probability_sum = sum(probabilities)
                for action, probability in zip(
                    request.actions, probabilities, strict=True
                ):
                    if probability_sum > 0:
                        probability /= probability_sum
                    request.node.children.append(_Child(action, probability))
                if request.reserved_child is not None:
                    request.reserved_child.in_flight = False


def _begin_search(
    runtime: _Runtime,
    context: _SearchContext,
    profile: BatchedProfile,
) -> _EvalRequest | None:
    observation = runtime.to_observation_class(context.session.observation)
    state = observation.current
    active = state.players[1 - context.your_index].active
    deck = context.participant.deck

    started = time.perf_counter()
    search_state = runtime.search_begin(
        observation,
        your_deck=random.sample(deck, min(len(deck), state.players[context.your_index].deckCount)),
        your_prize=random.sample(deck, min(len(deck), len(state.players[context.your_index].prize))),
        opponent_deck=[1072] * state.players[1 - context.your_index].deckCount,
        opponent_prize=[1] * len(state.players[1 - context.your_index].prize),
        opponent_hand=[1] * state.players[1 - context.your_index].handCount,
        opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
    )
    profile.search_begin_seconds += time.perf_counter() - started
    root, request = _prepare_node(runtime, context, None, search_state, profile)
    context.root = root
    return request


def _select_leaf(
    runtime: _Runtime,
    context: _SearchContext,
    profile: BatchedProfile,
) -> tuple[_EvalRequest | None, bool]:
    current = context.root
    if current is None:
        return None, False
    reserved_root_child: _Child | None = None

    while True:
        best_value = -1e9
        best_child: _Child | None = None
        exploration = 0.4 * math.sqrt(max(current.visit, 1))
        for child in current.children:
            if child.in_flight:
                continue
            visit = 0
            if child.node is None:
                value = current.total / max(current.visit, 1)
            else:
                value = child.node.total / max(child.node.visit, 1)
                visit = child.node.visit
            if current.state.observation.current.yourIndex != context.your_index:
                value = -value
            value += exploration * child.probability / (1 + visit)
            if best_value < value:
                best_value = value
                best_child = child

        if best_child is None:
            return None, False

        if reserved_root_child is None:
            # 1つのNN batch内ではrootの異なる枝を探索する。評価が終わるまで
            # この枝を予約することで、同じ未評価葉を何度も選ぶのを防ぐ。
            reserved_root_child = best_child

        if best_child.node is None:
            started = time.perf_counter()
            next_state = runtime.search_step(current.state.searchId, best_child.select)
            profile.search_step_seconds += time.perf_counter() - started
            profile.search_steps += 1
            child_node, request = _prepare_node(
                runtime, context, current, next_state, profile
            )
            best_child.node = child_node
            if request is not None and reserved_root_child is not None:
                reserved_root_child.in_flight = True
                request.reserved_child = reserved_root_child
            return request, True

        current = best_child.node
        if current.state.observation.current.result >= 0:
            current.backprop(current.value)
            return None, True


def _finish_search(context: _SearchContext) -> list[int]:
    root = context.root
    if root is None or not root.children:
        observation = context.session.observation
        select = observation.get("select") or {}
        option_count = len(select.get("option", []))
        select_count = int(select.get("maxCount", 0))
        return random.sample(range(option_count), select_count)

    max_child: _Child | None = None
    max_visit = -1
    for child in root.children:
        if child.node is not None and child.node.visit > max_visit:
            max_child = child
            max_visit = child.node.visit
    if max_child is None:
        max_child = max(root.children, key=lambda child: child.probability)
    return max_child.select


def _run_search_wave(
    runtime: _Runtime,
    contexts: list[_SearchContext],
    search_count: int,
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
) -> dict[int, list[int]]:
    actions: dict[int, list[int]] = {}
    if not contexts:
        return actions

    search_started = False
    try:
        root_requests: list[_EvalRequest] = []
        for context in contexts:
            request = _begin_search(runtime, context, profile)
            search_started = True
            if request is not None:
                root_requests.append(request)
        _apply_evaluations(root_requests, device, batch_size, profile)

        remaining = {id(context): search_count for context in contexts}
        while any(count > 0 for count in remaining.values()):
            leaf_requests: list[_EvalRequest] = []
            made_progress = False
            for context in contexts:
                # 1試合から複数の独立root branchを同時に発行する。これにより
                # 少数試合でもGPU batchを十分大きくできる。
                while remaining[id(context)] > 0:
                    request, advanced = _select_leaf(runtime, context, profile)
                    if not advanced:
                        break
                    made_progress = True
                    remaining[id(context)] -= 1
                    if request is not None:
                        leaf_requests.append(request)
            _apply_evaluations(leaf_requests, device, batch_size, profile)
            if not made_progress:
                break

        for context in contexts:
            actions[context.session.battle_ptr] = _finish_search(context)
        return actions
    finally:
        if search_started:
            started = time.perf_counter()
            runtime.search_end()
            profile.search_finalize_seconds += time.perf_counter() - started


def _random_action(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select")
    if select is None:
        return []
    return random.sample(range(len(select["option"])), int(select["maxCount"]))


def _raw_result(session: _MatchSession) -> int | None:
    current = session.observation.get("current")
    if current is None:
        return None
    result = int(current.get("result", -1))
    return result if result in (0, 1, 2) else None


def _to_result(session: _MatchSession, error: str | None = None) -> BatchedGameResult:
    raw_result = _raw_result(session)
    if raw_result in (0, 1):
        result = 1 - raw_result if session.request.swap else raw_result
    else:
        result = raw_result
    return BatchedGameResult(
        name0=session.request.name0,
        name1=session.request.name1,
        game_index=session.request.game_index,
        swap=session.request.swap,
        result=result,
        turns=session.selections,
        error=error,
    )


def run_batched_tournament(
    specs: list[Any],
    pairings: list[tuple[str, str]],
    num_games: int,
    *,
    alternate_sides: bool = True,
    device_name: str = "auto",
    batch_size: int = 128,
    lanes: int = 128,
    search_count: int = 10,
    max_selections: int = 2000,
    seed: int = 0,
) -> BatchedTournamentOutput:
    """総当たりの全試合をlane pool上で進める。"""
    if batch_size < 1:
        raise ValueError("batch_sizeは1以上で指定してください。")
    if lanes < 1:
        raise ValueError("lanesは1以上で指定してください。")
    if search_count < 0:
        raise ValueError("search_countは0以上で指定してください。")

    random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device_name)
    canonical_spec = next(
        (
            spec
            for spec in specs
            if (Path(spec.agent_path).resolve().parent / "model.pth").exists()
        ),
        None,
    )
    if canonical_spec is None:
        raise ValueError("batched backendには少なくとも1つrl_mcts agentが必要です。")
    canonical_src = Path(canonical_spec.agent_path).resolve().parent
    runtime = _load_runtime(canonical_src)
    participants = _load_participants(specs, runtime, device)
    profile = BatchedProfile()

    requests = deque(
        BatchedGameRequest(
            name0=name0,
            name1=name1,
            game_index=game_index + 1,
            swap=alternate_sides and game_index % 2 == 1,
        )
        for name0, name1 in pairings
        for game_index in range(num_games)
    )
    active: list[_MatchSession] = []
    results: list[BatchedGameResult] = []

    def fill_lanes() -> None:
        while requests and len(active) < lanes:
            request = requests.popleft()
            try:
                active.append(_start_session(runtime, request, participants, profile))
            except Exception as exc:  # noqa: BLE001 - 失敗した試合だけunresolvedにする
                results.append(
                    BatchedGameResult(
                        name0=request.name0,
                        name1=request.name1,
                        game_index=request.game_index,
                        swap=request.swap,
                        result=None,
                        turns=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

    try:
        fill_lanes()
        while active:
            survivors: list[_MatchSession] = []
            for session in active:
                if _raw_result(session) is not None:
                    results.append(_to_result(session))
                    runtime.lib.BattleFinish(session.battle_ptr)
                elif session.selections >= max_selections:
                    results.append(
                        _to_result(
                            session,
                            error=f"max_selections={max_selections}を超えました。",
                        )
                    )
                    runtime.lib.BattleFinish(session.battle_ptr)
                else:
                    survivors.append(session)
            active = survivors
            fill_lanes()
            if not active:
                break

            selected_actions: dict[int, list[int]] = {}
            contexts: list[_SearchContext] = []
            for session in active:
                observation = session.observation
                select = observation.get("select")
                if select is None:
                    selected_actions[session.battle_ptr] = []
                    continue
                if len(select["option"]) == 0 or int(select["maxCount"]) == 0:
                    selected_actions[session.battle_ptr] = []
                    continue

                your_index = int(observation["current"]["yourIndex"])
                participant = session.players[your_index]
                if participant.random_policy:
                    selected_actions[session.battle_ptr] = _random_action(observation)
                else:
                    contexts.append(
                        _SearchContext(
                            session=session,
                            participant=participant,
                            your_index=your_index,
                        )
                    )

            selected_actions.update(
                _run_search_wave(
                    runtime,
                    contexts,
                    search_count,
                    device,
                    batch_size,
                    profile,
                )
            )

            next_active: list[_MatchSession] = []
            for session in active:
                action = selected_actions[session.battle_ptr]
                argument = (ctypes.c_int * len(action))(*action)
                try:
                    started = time.perf_counter()
                    error = runtime.lib.Select(
                        session.battle_ptr, argument, len(action)
                    )
                    if error != 0:
                        raise RuntimeError(f"libcg.Select error={error}")
                    session.observation, session.select_player = _battle_observation(
                        runtime, session.battle_ptr
                    )
                    profile.battle_step_seconds += time.perf_counter() - started
                    profile.battle_steps += 1
                    session.selections += 1
                    next_active.append(session)
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        _to_result(session, error=f"{type(exc).__name__}: {exc}")
                    )
                    runtime.lib.BattleFinish(session.battle_ptr)
            active = next_active
    finally:
        for session in active:
            runtime.lib.BattleFinish(session.battle_ptr)

    results.sort(key=lambda result: (result.name0, result.name1, result.game_index))
    return BatchedTournamentOutput(
        results=results,
        profile=profile,
        device=str(device),
    )
