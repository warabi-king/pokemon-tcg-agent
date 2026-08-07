"""共有libcgと試合横断CPUバッチ推論による学習サンプル収集。"""

from __future__ import annotations

from collections import deque
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
import random
import time
from typing import Any

import torch

from batched_tournament import (
    BatchedGameRequest,
    BatchedGameResult,
    BatchedLearnSample,
    BatchedProfile,
    _MatchSession,
    _Participant,
    _SearchContext,
    _battle_observation,
    _load_runtime,
    _raw_result,
    _run_search_wave,
    _start_session,
    _to_result,
    update_seen_opponent_cards,
)


@dataclass
class BatchedTrainingAgent:
    name: str
    model: torch.nn.Module
    deck: list[int]
    # 探索木の相手ノード評価に使うモデル（self/opp 二重運用）。None なら model を流用。
    opponent_model: torch.nn.Module | None = None
    # policy(raw logits)をprior確率へ変換するsoftmax温度。呼び出し側がmodel/opponent_model
    # 読み込み時にrl_mcts.checkpointから取得した値を渡す(未指定なら自己対戦系の既定10.0)。
    policy_temperature: float = 10.0
    opponent_policy_temperature: float | None = None


@dataclass
class BatchedTrainingOutput:
    samples: dict[str, list[BatchedLearnSample]]
    results: list[BatchedGameResult]
    profile: BatchedProfile
    # 各エージェントを相手にした側が打った手のサンプル（opp モデルの学習用）。
    # 1つの着手サンプルは「打ち手の self」と「相手の opp」の両方へ振り分けられる。
    opp_samples: dict[str, list[BatchedLearnSample]] = field(default_factory=dict)


def _label_finished_game(
    session: _MatchSession,
    per_player: list[list[BatchedLearnSample]],
    samples: dict[str, list[BatchedLearnSample]],
    lambda_value: float,
    opp_samples: dict[str, list[BatchedLearnSample]] | None = None,
) -> None:
    result = _raw_result(session)
    if result is None:
        return

    for player_index, player_samples in enumerate(per_player):
        if result == 2:
            value = 0.0
        else:
            value = 1.0 if player_index == result else -1.0
        opponent_name = session.players[1 - player_index].name
        for sample in reversed(player_samples):
            label = (value + sample.value) * 0.5
            value = value * lambda_value + sample.value * (1.0 - lambda_value)
            sample.value = label
            # 打ち手の self へ。
            samples[session.players[player_index].name].append(sample)
            # 同じサンプルを相手エージェントの opp（相手データ）へも振り分ける。
            if opp_samples is not None:
                opp_samples.setdefault(opponent_name, []).append(sample)


def collect_batched_training_samples(
    agents: list[BatchedTrainingAgent],
    pairings: list[tuple[str, str]],
    games: int,
    *,
    canonical_src: Path,
    device: torch.device,
    batch_size: int,
    lanes: int,
    search_count: int,
    lambda_value: float,
    max_selections: int = 2000,
    alternate_sides: bool = True,
    seed: int = 0,
) -> BatchedTrainingOutput:
    """複数対戦を同時進行し、モデル別にNN評価をCPUバッチ化する。"""
    if games < 1:
        raise ValueError("gamesは1以上で指定してください。")
    if batch_size < 1:
        raise ValueError("batch_sizeは1以上で指定してください。")
    if lanes < 1:
        raise ValueError("lanesは1以上で指定してください。")
    if search_count < 0:
        raise ValueError("search_countは0以上で指定してください。")

    random.seed(seed)
    torch.manual_seed(seed)
    runtime = _load_runtime(canonical_src)
    participants: dict[str, _Participant] = {}
    samples: dict[str, list[BatchedLearnSample]] = {}
    opp_samples: dict[str, list[BatchedLearnSample]] = {}
    for agent in agents:
        agent.model.eval()
        agent.model.to(device)
        opp_model = agent.opponent_model
        opp_key = None
        if opp_model is not None:
            opp_model.eval()
            opp_model.to(device)
            opp_key = f"{agent.name}__opp"
        participants[agent.name] = _Participant(
            name=agent.name,
            deck=agent.deck,
            model=agent.model,
            model_key=agent.name,
            random_policy=False,
            opponent_model=opp_model,
            opponent_model_key=opp_key,
            policy_temperature=agent.policy_temperature,
            opponent_policy_temperature=agent.opponent_policy_temperature,
        )
        samples[agent.name] = []
        opp_samples[agent.name] = []

    profile = BatchedProfile()
    requests = deque(
        BatchedGameRequest(
            name0=name0,
            name1=name1,
            game_index=game_index + 1,
            swap=alternate_sides and game_index % 2 == 1,
        )
        for name0, name1 in pairings
        for game_index in range(games)
    )
    active: list[_MatchSession] = []
    per_battle_samples: dict[int, list[list[BatchedLearnSample]]] = {}
    results: list[BatchedGameResult] = []

    def finish_session(session: _MatchSession, error: str | None = None) -> None:
        if error is None:
            _label_finished_game(
                session,
                per_battle_samples[session.battle_ptr],
                samples,
                lambda_value,
                opp_samples=opp_samples,
            )
        results.append(_to_result(session, error=error))
        runtime.lib.BattleFinish(session.battle_ptr)
        per_battle_samples.pop(session.battle_ptr, None)

    def fill_lanes() -> None:
        while requests and len(active) < lanes:
            request = requests.popleft()
            try:
                session = _start_session(runtime, request, participants, profile)
                active.append(session)
                per_battle_samples[session.battle_ptr] = [[], []]
            except Exception as exc:  # noqa: BLE001
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
                    finish_session(session)
                elif session.selections >= max_selections:
                    finish_session(
                        session,
                        error=f"max_selections={max_selections}を超えました。",
                    )
                else:
                    survivors.append(session)
            active = survivors
            fill_lanes()
            if not active:
                break

            selected_actions: dict[int, list[int]] = {}
            contexts: list[_SearchContext] = []
            for session in active:
                select = session.observation.get("select")
                if (
                    select is None
                    or len(select["option"]) == 0
                    or int(select["maxCount"]) == 0
                ):
                    selected_actions[session.battle_ptr] = []
                    continue
                your_index = int(session.observation["current"]["yourIndex"])
                contexts.append(
                    _SearchContext(
                        session=session,
                        participant=session.players[your_index],
                        your_index=your_index,
                    )
                )

            wave_samples: dict[int, BatchedLearnSample] = {}
            selected_actions.update(
                _run_search_wave(
                    runtime,
                    contexts,
                    search_count,
                    device,
                    batch_size,
                    profile,
                    sample_sink=wave_samples,
                )
            )
            for context in contexts:
                sample = wave_samples.get(context.session.battle_ptr)
                if sample is not None:
                    per_battle_samples[context.session.battle_ptr][
                        context.your_index
                    ].append(sample)

            next_active: list[_MatchSession] = []
            for session in active:
                action = selected_actions[session.battle_ptr]
                argument = (ctypes.c_int * len(action))(*action)
                try:
                    started = time.perf_counter()
                    error = runtime.lib.Select(
                        session.battle_ptr,
                        argument,
                        len(action),
                    )
                    if error != 0:
                        raise RuntimeError(f"libcg.Select error={error}")
                    session.observation, session.select_player = _battle_observation(
                        runtime,
                        session.battle_ptr,
                    )
                    update_seen_opponent_cards(
                        session.observation,
                        session.seen_opponent_cards,
                    )
                    profile.battle_step_seconds += time.perf_counter() - started
                    profile.battle_steps += 1
                    session.selections += 1
                    next_active.append(session)
                except Exception as exc:  # noqa: BLE001
                    finish_session(
                        session,
                        error=f"{type(exc).__name__}: {exc}",
                    )
            active = next_active
    finally:
        for session in active:
            runtime.lib.BattleFinish(session.battle_ptr)

    results.sort(key=lambda result: (result.name0, result.name1, result.game_index))
    return BatchedTrainingOutput(
        samples=samples, results=results, profile=profile, opp_samples=opp_samples
    )
