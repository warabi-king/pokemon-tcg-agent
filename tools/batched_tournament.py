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

import ast
from collections import defaultdict, deque
import copy
import ctypes
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import random
import sys
import time
import traceback
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np

try:
    from deck_belief import sample_hidden_zones, update_seen_opponent_cards
except ModuleNotFoundError:  # ``python -m unittest tools...``用
    from .deck_belief import sample_hidden_zones, update_seen_opponent_cards

try:
    import msgspec
except ImportError:
    msgspec = None


if msgspec is not None:
    class _FastCard(msgspec.Struct):
        id: int


    class _FastPokemon(msgspec.Struct):
        id: int
        hp: int
        energyCards: list[_FastCard]
        tools: list[_FastCard]


    class _FastPlayerState(msgspec.Struct):
        active: list[_FastPokemon | None]
        bench: list[_FastPokemon]
        deckCount: int
        discard: list[_FastCard]
        prize: list[_FastCard | None]
        handCount: int
        hand: list[_FastCard] | None
        poisoned: bool
        burned: bool
        asleep: bool
        paralyzed: bool
        confused: bool


    class _FastState(msgspec.Struct):
        turn: int
        yourIndex: int
        firstPlayer: int
        result: int
        stadium: list[_FastCard]
        looking: list[_FastCard | None] | None
        players: list[_FastPlayerState]


    class _FastOption(msgspec.Struct):
        type: int
        number: int | None = None
        area: int | None = None
        index: int | None = None
        playerIndex: int | None = None
        toolIndex: int | None = None
        energyIndex: int | None = None
        inPlayArea: int | None = None
        inPlayIndex: int | None = None
        attackId: int | None = None
        cardId: int | None = None
        specialConditionType: int | None = None


    class _FastSelectData(msgspec.Struct):
        context: int
        maxCount: int
        option: list[_FastOption]
        deck: list[_FastCard] | None


    class _FastObservation(msgspec.Struct):
        select: _FastSelectData | None
        current: _FastState | None
        search_begin_input: str | None = None


    class _FastSearchState(msgspec.Struct):
        observation: _FastObservation
        searchId: int


    class _FastApiResult(msgspec.Struct):
        state: _FastSearchState | None
        error: int


_LIGHTWEIGHT_WORKER_ENV = "PTCG_BATCHED_LIGHTWEIGHT_WORKER"
if os.environ.get(_LIGHTWEIGHT_WORKER_ENV) == "1":
    # Windows multiprocessing uses spawn.  CPU libcg workers do not execute a
    # neural network, so importing the multi-gigabyte CUDA runtime in every
    # worker only wastes commit memory and can make workers fail at startup.
    torch = None
else:
    import torch


_TorchModuleBase = torch.nn.Module if torch is not None else object


MAX_ACTIONS = 64
SETUP_SELECT_CONTEXTS = frozenset((1, 2))
TRAINING_EPISODE_FORMAT = "pokemon-tcg-agent/imitation-episode-v1"
_TRAINING_OPTION_FIELDS = (
    "type",
    "number",
    "area",
    "index",
    "playerIndex",
    "toolIndex",
    "energyIndex",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
    "cardId",
    "specialConditionType",
)


@dataclass(frozen=True)
class _CpuDevice:
    type: str = "cpu"

    def __str__(self) -> str:
        return self.type


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
    search_step_c_api_seconds: float = 0.0
    search_step_json_seconds: float = 0.0
    search_step_dataclass_seconds: float = 0.0
    feature_seconds: float = 0.0
    nn_seconds: float = 0.0
    nn_merge_seconds: float = 0.0
    nn_input_seconds: float = 0.0
    nn_forward_submit_seconds: float = 0.0
    nn_output_wait_seconds: float = 0.0
    nn_tolist_seconds: float = 0.0
    nn_response_pack_seconds: float = 0.0
    nn_decoder_source_tokens: int = 0
    nn_decoder_padded_tokens: int = 0
    response_put_seconds: float = 0.0
    search_finalize_seconds: float = 0.0
    nn_evaluations: int = 0
    nn_batches: int = 0
    batch_sizes: list[int] = field(default_factory=list)
    battle_steps: int = 0
    search_steps: int = 0
    cuda_ensemble_waves: int = 0
    cuda_per_model_waves: int = 0
    cpu_workers: int = 1
    remote_wait_seconds: float = 0.0
    remote_numpy_pack_seconds: float = 0.0
    batch_collect_seconds: float = 0.0
    ipc_messages: int = 0

    @property
    def mean_batch_size(self) -> float:
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    @property
    def max_batch_size(self) -> int:
        return max(self.batch_sizes, default=0)

    @property
    def decoder_token_efficiency(self) -> float:
        if self.nn_decoder_padded_tokens <= 0:
            return 0.0
        return self.nn_decoder_source_tokens / self.nn_decoder_padded_tokens


@dataclass
class BatchedTournamentOutput:
    results: list[BatchedGameResult]
    profile: BatchedProfile
    device: str


@dataclass
class BatchedLearnSample:
    """学習器が必要とする、1回の着手判断のMCTS教師データ。"""

    value: float
    policy: list[float]
    sv_enc: Any
    sv_dec: Any


@dataclass
class _SparseVectorSnapshot:
    """forward用paddingの影響を受けないSparseVectorのスナップショット。"""

    index: list[int]
    value: list[float]
    offset: list[int]


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
    search_step_timing: Any


@dataclass
class _SearchStepTiming:
    """cg.api.search_step内で行われるJSON変換の累積時間。"""

    active: int = 0
    json_seconds: float = 0.0
    dataclass_seconds: float = 0.0
    fast_decode_fallbacks: int = 0

    def snapshot(self) -> tuple[float, float]:
        return self.json_seconds, self.dataclass_seconds


def _create_fast_api_decoder() -> Any | None:
    """SearchBegin/SearchStepのJSONをreflectionなしで軽量Structへdecodeする。"""
    if msgspec is None:
        return None
    return msgspec.json.Decoder(_FastApiResult)


_FAST_UNTYPED_JSON_DECODER = msgspec.json.Decoder() if msgspec is not None else None


def _decode_json_dict(data: bytes) -> dict[str, Any]:
    if _FAST_UNTYPED_JSON_DECODER is not None:
        return _FAST_UNTYPED_JSON_DECODER.decode(data)
    return json.loads(data.decode())


def _to_search_observation(observation: dict[str, Any], fallback: Any) -> Any:
    if msgspec is not None:
        return msgspec.convert(observation, type=_FastObservation)
    return fallback(observation)


@dataclass
class _Participant:
    name: str
    deck: tuple[int, ...]
    model: torch.nn.Module | None
    model_key: str | None
    random_policy: bool
    opponent_model: torch.nn.Module | None = None
    opponent_model_key: str | None = None


@dataclass
class _MatchSession:
    request: BatchedGameRequest
    players: tuple[_Participant, _Participant]
    battle_ptr: int
    observation: dict[str, Any]
    select_player: int
    selections: int = 0
    seen_opponent_cards: tuple[dict[int, int], dict[int, int]] = field(
        default_factory=lambda: ({}, {})
    )
    training_decisions: list[dict[str, Any]] = field(default_factory=list)


class _Child:
    __slots__ = ("node", "select", "probability", "in_flight")

    def __init__(self, select: list[int], probability: float) -> None:
        self.node: _Node | None = None
        self.select = select
        self.probability = probability
        # parallel MCTSで同じroot branchを1つのbatchへ重複投入しないための予約。
        self.in_flight = False


class _Node:
    __slots__ = (
        "value",
        "total",
        "visit",
        "parent",
        "children",
        "search_id",
        "player_index",
        "result",
    )

    def __init__(
        self,
        parent: _Node | None,
        search_id: int,
        player_index: int,
        result: int,
    ) -> None:
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children: list[_Child] = []
        # 特徴量生成後のObservation全体は保持しない。木探索で必要なのは
        # libcg Search ID、手番、終局結果の3値だけ。
        self.search_id = search_id
        self.player_index = player_index
        self.result = result

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
    root_sample: BatchedLearnSample | None = None
    full_decks: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    policy_only: bool = False


@dataclass
class _EvalRequest:
    context: _SearchContext
    node: _Node
    actions: list[list[int]]
    encoder: Any
    decoder: Any
    model_key: str
    reserved_child: _Child | None = None


@dataclass(frozen=True)
class _RemoteEvalJob:
    """CPU workerから中央NN batcherへ渡す、モデル別の評価入力。"""

    model_key: str
    batch_count: int
    decoder_words: int
    encoder: tuple[np.ndarray, np.ndarray, np.ndarray]
    decoder: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class _CudaEvalJob:
    """1つのCUDA streamへ投入済みのNN評価。

    pinned_inputsを保持して、非同期H2D copyが完了する前にhost Tensorが解放
    されないようにする。value_rows/policy_rowsは同じstream上で非同期D2H
    copyされ、全streamを一度だけ同期した後にPythonへ戻す。
    """

    chunk: list[_EvalRequest]
    pinned_inputs: list[torch.Tensor]
    value_rows: torch.Tensor
    policy_rows: torch.Tensor


class _CudaEnsembleTail(_TorchModuleBase):
    """EmbeddingBagより後ろのdense/attention部分。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.d_model = model.d_model
        self.num_words_encoder = model.num_words_encoder
        self.encoder = copy.deepcopy(model.encoder)
        self.encoder_fc = copy.deepcopy(model.encoder_fc)
        self.decoder = copy.deepcopy(model.decoder)
        self.decoder_fc = copy.deepcopy(model.decoder_fc)

    def forward(
        self,
        encoder_bags: torch.Tensor,
        decoder_bags: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value_hidden = encoder_bags.reshape(
            -1,
            self.num_words_encoder,
            self.d_model,
        ).transpose(0, 1)
        batch_size = value_hidden.size(1)
        encoder_out = self.encoder(value_hidden)
        values = self.encoder_fc(encoder_out)
        values = torch.tanh(values.mean(0))

        policy_hidden = decoder_bags.reshape(
            batch_size,
            -1,
            self.d_model,
        ).transpose(0, 1)
        for layer in self.decoder:
            policy_hidden = layer(policy_hidden, encoder_out)
        policies = self.decoder_fc(policy_hidden)
        policies = policies.transpose(0, 1).reshape(batch_size, -1)
        return values, torch.tanh(policies)


def _models_from_participants(
    participants: Iterable[_Participant],
) -> list[tuple[str, torch.nn.Module]]:
    model_by_key: dict[str, torch.nn.Module] = {}
    for participant in participants:
        for model_key, model in (
            (participant.model_key, participant.model),
            (participant.opponent_model_key, participant.opponent_model),
        ):
            if model_key is not None and model is not None:
                model_by_key.setdefault(model_key, model)
    return list(model_by_key.items())


def _model_axis_embedding_bag(
    weights: torch.Tensor,
    indices: torch.Tensor,
    per_sample_weights: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """model次元を保ったままweighted EmbeddingBag(sum)を計算する。"""
    model_count, value_count = indices.shape
    vocabulary_size = weights.shape[1]
    model_indices = torch.arange(
        model_count,
        dtype=indices.dtype,
        device=indices.device,
    )
    flat_indices = (
        indices + model_indices.unsqueeze(1) * vocabulary_size
    ).reshape(-1)
    model_value_offsets = torch.arange(
        model_count,
        dtype=offsets.dtype,
        device=offsets.device,
    ).unsqueeze(1)
    flat_offsets = (
        offsets + model_value_offsets * value_count
    ).reshape(-1)
    bags = torch.nn.functional.embedding_bag(
        flat_indices,
        weights.reshape(-1, weights.shape[-1]),
        flat_offsets,
        mode="sum",
        per_sample_weights=per_sample_weights.reshape(-1),
    )
    return bags.reshape(model_count, offsets.shape[1], weights.shape[-1])


class _MpsModelAxisEngine:
    """MPS専用。active modelの重みをまとめて手動batched演算する。"""

    def __init__(
        self,
        model_items: Iterable[tuple[str, torch.nn.Module]],
        device: torch.device,
    ) -> None:
        if device.type != "mps":
            raise ValueError("MPS model-axis blockにはMPS deviceが必要です。")
        model_by_key = dict(model_items)
        if not model_by_key:
            raise ValueError("MPS model-axis blockには少なくとも1つNNモデルが必要です。")

        self.model_keys = list(model_by_key)
        self.model_indices = {
            model_key: index for index, model_key in enumerate(self.model_keys)
        }
        models = [model_by_key[model_key] for model_key in self.model_keys]
        first = models[0]
        self.model_count = len(models)
        self.d_model = int(first.d_model)
        self.num_words_encoder = int(first.num_words_encoder)
        required_attributes = (
            "encoder_bag",
            "decoder_bag",
            "encoder",
            "encoder_fc",
            "decoder",
            "decoder_fc",
            "d_model",
            "num_words_encoder",
        )
        if any(not hasattr(model, name) for model in models for name in required_attributes):
            raise TypeError(
                "MPS model-axisはEmbeddingBag + Transformer形式の同一モデルを必要とします。"
            )
        first_state_shapes = {
            key: tuple(value.shape) for key, value in first.state_dict().items()
        }
        for model in models[1:]:
            if {
                key: tuple(value.shape) for key, value in model.state_dict().items()
            } != first_state_shapes:
                raise ValueError("MPS model-axis block内のモデル構造が一致していません。")

        self.encoder_weights = torch.stack(
            [model.encoder_bag.weight.detach() for model in models]
        )
        self.decoder_weights = torch.stack(
            [model.decoder_bag.weight.detach() for model in models]
        )
        self.device = device
        state_dicts = [model.state_dict() for model in models]
        self.stacked_state = {
            key: torch.stack([state[key].detach() for state in state_dicts])
            for key in state_dicts[0]
            if key not in ("encoder_bag.weight", "decoder_bag.weight")
        }
        self.num_encoder_layers = len(first.encoder.layers)
        self.num_decoder_layers = len(first.decoder)
        self.num_heads = int(first.encoder.layers[0].self_attn.num_heads)
        self.encoder_norm_eps = [
            (float(layer.norm1.eps), float(layer.norm2.eps))
            for layer in first.encoder.layers
        ]
        self.decoder_norm_eps = [
            (float(layer.norm1.eps), float(layer.norm2.eps))
            for layer in first.decoder
        ]

    @staticmethod
    def _linear(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """model別Linearを1回のbmmとして計算する。"""
        model_count = inputs.shape[0]
        leading_shape = inputs.shape[1:-1]
        outputs = torch.bmm(
            inputs.reshape(model_count, -1, inputs.shape[-1]),
            weight.transpose(1, 2),
        )
        outputs = outputs + bias.unsqueeze(1)
        return outputs.reshape(model_count, *leading_shape, weight.shape[1])

    @staticmethod
    def _layer_norm(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """model別LayerNorm。最後のfeature軸だけを正規化する。"""
        variance, mean = torch.var_mean(
            inputs,
            dim=-1,
            correction=0,
            keepdim=True,
        )
        normalized = (inputs - mean) * torch.rsqrt(variance + eps)
        expand = (slice(None),) + (None,) * (inputs.ndim - 2) + (slice(None),)
        return normalized * weight[expand] + bias[expand]

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        prefix: str,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """model別MultiheadAttentionを通常のbatched matmulへ畳み込む。"""
        projection_weight = state[f"{prefix}.in_proj_weight"]
        projection_bias = state[f"{prefix}.in_proj_bias"]
        d_model = self.d_model
        q = self._linear(
            query,
            projection_weight[:, :d_model],
            projection_bias[:, :d_model],
        )
        k = self._linear(
            key,
            projection_weight[:, d_model : 2 * d_model],
            projection_bias[:, d_model : 2 * d_model],
        )
        v = self._linear(
            value,
            projection_weight[:, 2 * d_model :],
            projection_bias[:, 2 * d_model :],
        )

        model_count, batch_count, query_words, _ = q.shape
        key_words = k.shape[2]
        head_size = d_model // self.num_heads

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(
                model_count,
                batch_count,
                -1,
                self.num_heads,
                head_size,
            ).permute(0, 1, 3, 2, 4)

        q = split_heads(q).reshape(-1, query_words, head_size)
        k = split_heads(k).reshape(-1, key_words, head_size)
        v = split_heads(v).reshape(-1, key_words, head_size)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(head_size)
        attention = torch.softmax(scores, dim=-1)
        context = torch.bmm(attention, v)
        context = context.reshape(
            model_count,
            batch_count,
            self.num_heads,
            query_words,
            head_size,
        ).permute(0, 1, 3, 2, 4).reshape(
            model_count,
            batch_count,
            query_words,
            d_model,
        )
        return self._linear(
            context,
            state[f"{prefix}.out_proj.weight"],
            state[f"{prefix}.out_proj.bias"],
        )

    def _manual_forward(
        self,
        encoder_bags: torch.Tensor,
        decoder_bags: torch.Tensor,
        state: dict[str, torch.Tensor],
        model_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """vmap非対応のMPS attentionを使わないmodel-axis forward。"""
        encoder = encoder_bags.reshape(
            model_count,
            -1,
            self.num_words_encoder,
            self.d_model,
        )
        for layer_index in range(self.num_encoder_layers):
            prefix = f"encoder.layers.{layer_index}"
            attended = self._attention(
                encoder,
                encoder,
                encoder,
                f"{prefix}.self_attn",
                state,
            )
            residual = self._layer_norm(
                encoder + attended,
                state[f"{prefix}.norm1.weight"],
                state[f"{prefix}.norm1.bias"],
                self.encoder_norm_eps[layer_index][0],
            )
            hidden = self._linear(
                residual,
                state[f"{prefix}.linear1.weight"],
                state[f"{prefix}.linear1.bias"],
            ).relu_()
            hidden = self._linear(
                hidden,
                state[f"{prefix}.linear2.weight"],
                state[f"{prefix}.linear2.bias"],
            )
            encoder = self._layer_norm(
                residual + hidden,
                state[f"{prefix}.norm2.weight"],
                state[f"{prefix}.norm2.bias"],
                self.encoder_norm_eps[layer_index][1],
            )

        values = self._linear(
            encoder,
            state["encoder_fc.weight"],
            state["encoder_fc.bias"],
        ).mean(dim=2)
        values = torch.tanh(values)

        batch_count = encoder.shape[1]
        decoder = decoder_bags.reshape(
            model_count,
            batch_count,
            -1,
            self.d_model,
        )
        for layer_index in range(self.num_decoder_layers):
            prefix = f"decoder.{layer_index}"
            attended = self._attention(
                decoder,
                encoder,
                encoder,
                f"{prefix}.attention",
                state,
            )
            residual = self._layer_norm(
                decoder + attended,
                state[f"{prefix}.norm1.weight"],
                state[f"{prefix}.norm1.bias"],
                self.decoder_norm_eps[layer_index][0],
            )
            hidden = self._linear(
                residual,
                state[f"{prefix}.fc1.weight"],
                state[f"{prefix}.fc1.bias"],
            ).relu_()
            hidden = self._linear(
                hidden,
                state[f"{prefix}.fc2.weight"],
                state[f"{prefix}.fc2.bias"],
            )
            decoder = self._layer_norm(
                residual + hidden,
                state[f"{prefix}.norm2.weight"],
                state[f"{prefix}.norm2.bias"],
                self.decoder_norm_eps[layer_index][1],
            )
        policies = self._linear(
            decoder,
            state["decoder_fc.weight"],
            state["decoder_fc.bias"],
        ).squeeze(-1)
        return values, torch.tanh(policies)

    def evaluate(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_bags = _model_axis_embedding_bag(
            self.encoder_weights,
            index_encoder,
            value_encoder,
            offset_encoder,
        )
        decoder_bags = _model_axis_embedding_bag(
            self.decoder_weights,
            index_decoder,
            value_decoder,
            offset_decoder,
        )
        return self._manual_forward(
            encoder_bags,
            decoder_bags,
            self.stacked_state,
            self.model_count,
        )

class _CudaEnsembleEvaluator:
    """CUDA専用。stackした全モデルをtorch.vmapで1回に評価する。"""

    _embedding_bag = staticmethod(_model_axis_embedding_bag)

    def __init__(
        self,
        participants: Iterable[_Participant],
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA ensembleにはCUDA deviceが必要です。")
        model_items = _models_from_participants(participants)
        if not model_items:
            raise ValueError("CUDA ensembleには少なくとも1つNNモデルが必要です。")
        self.model_keys = [key for key, _model in model_items]
        self.model_indices = {
            model_key: index for index, model_key in enumerate(self.model_keys)
        }
        models = [model for _key, model in model_items]
        first = models[0]
        self.model_count = len(models)
        self.num_words_encoder = int(first.num_words_encoder)
        first_state_shapes = {
            key: tuple(value.shape) for key, value in first.state_dict().items()
        }
        for model in models[1:]:
            if {
                key: tuple(value.shape) for key, value in model.state_dict().items()
            } != first_state_shapes:
                raise ValueError("CUDA ensemble内のモデル構造が一致していません。")

        self.encoder_weights = torch.stack(
            [model.encoder_bag.weight.detach() for model in models]
        )
        self.decoder_weights = torch.stack(
            [model.decoder_bag.weight.detach() for model in models]
        )
        tails = [_CudaEnsembleTail(model) for model in models]
        self.params, self.buffers = torch.func.stack_module_state(tails)
        self.base_model = copy.deepcopy(tails[0]).to("meta")

        def call_one(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            encoder_bags: torch.Tensor,
            decoder_bags: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.func.functional_call(
                self.base_model,
                (params, buffers),
                (encoder_bags, decoder_bags),
            )

        self.call = torch.vmap(call_one, in_dims=(0, 0, 0, 0))

    def evaluate(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_bags = self._embedding_bag(
            self.encoder_weights,
            index_encoder,
            value_encoder,
            offset_encoder,
        )
        decoder_bags = self._embedding_bag(
            self.decoder_weights,
            index_decoder,
            value_decoder,
            offset_decoder,
        )
        return self.call(
            self.params,
            self.buffers,
            encoder_bags,
            decoder_bags,
        )


class _MpsModelAxisEvaluator:
    """MPS専用。固定した全モデルをmanual batched演算で評価する。"""

    def __init__(
        self,
        participants: Iterable[_Participant],
        device: torch.device,
    ) -> None:
        if device.type != "mps":
            raise ValueError("MPS model-axis evaluatorにはMPS deviceが必要です。")
        model_items = _models_from_participants(participants)
        if not model_items:
            raise ValueError("MPS model-axisには少なくとも1つNNモデルが必要です。")
        self.model_count = len(model_items)
        self.model_keys = [key for key, _model in model_items]
        self.model_indices = {
            model_key: index for index, model_key in enumerate(self.model_keys)
        }
        self.engine = _MpsModelAxisEngine(model_items, device)
        self.num_words_encoder = self.engine.num_words_encoder

    def evaluate(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.engine.evaluate(
            index_encoder,
            value_encoder,
            offset_encoder,
            index_decoder,
            value_decoder,
            offset_decoder,
        )

# RTX 3050実測では標準EmbeddingBagで5モデルを統合するとmodel当たり32件まで
# model-axisがper-modelより1.6倍以上速く、64件で同等になる。不均等batchの
# padding余裕を残し、平均32件未満だけmodel-axisへ切り替える。
#
# MPSは固定した全モデルのdense weightを一度だけstackし、waveごとのweight
# コピーを行わない。inactive modelのpaddingを考慮した閾値より小さいwaveだけ
# manual model-axisを使い、それ以上はqueued per-modelにする。
_CUDA_MODEL_AXIS_BATCH_CROSSOVER = 32.0
_MPS_MODEL_AXIS_BATCH_CROSSOVER = 16.0
_MPS_MODEL_AXIS_MIN_ROW_EFFICIENCY = 0.30


def _model_axis_batch_crossover(
    device: torch.device,
    evaluator: _CudaEnsembleEvaluator,
    active_model_count: int,
) -> float:
    del evaluator, active_model_count
    return _CUDA_MODEL_AXIS_BATCH_CROSSOVER if device.type == "cuda" else 0.0


def select_device(requested: str) -> torch.device:
    if torch is None:
        if requested not in ("auto", "cpu"):
            raise RuntimeError(
                "軽量libcg workerはCPU deviceだけを利用できます。"
            )
        return _CpuDevice()

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


def _read_model_feature_constants(agent_src: Path) -> dict[str, int]:
    """model.pyを実行せず、特徴量生成に必要な整数定数だけを読む。"""
    required = {"DECODER_ATTACK_OFFSET", "DECODER_MAIN_FEATURE"}
    constants: dict[str, int] = {}
    tree = ast.parse(
        (agent_src / "rl_mcts" / "model.py").read_text(encoding="utf-8")
    )
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in required:
            continue
        value = ast.literal_eval(statement.value)
        if not isinstance(value, int):
            raise TypeError(f"{target.id}は整数である必要があります。")
        constants[target.id] = value
    missing = required - constants.keys()
    if missing:
        raise RuntimeError(f"model.pyに特徴量定数がありません: {sorted(missing)}")
    return constants


def _install_lightweight_model_shim(agent_src: Path) -> None:
    """特徴量生成に必要なmodel定数だけをTorchなしで提供する。"""
    if "rl_mcts.model" in sys.modules:
        return

    from cg.api import all_attack, all_card_data

    constants = _read_model_feature_constants(agent_src)
    model_module = ModuleType("rl_mcts.model")
    model_module.DECODER_ATTACK_OFFSET = constants["DECODER_ATTACK_OFFSET"]
    model_module.DECODER_MAIN_FEATURE = constants["DECODER_MAIN_FEATURE"]

    @lru_cache(maxsize=1)
    def card_count() -> int:
        all_cards = all_card_data()
        return max(all_cards, key=lambda card: card.cardId).cardId + 1

    @lru_cache(maxsize=1)
    def attack_count() -> int:
        all_attacks = all_attack()
        return max(all_attacks, key=lambda attack: attack.attackId).attackId + 1

    model_module.card_count = card_count
    model_module.attack_count = attack_count
    sys.modules["rl_mcts.model"] = model_module


def _load_runtime(agent_src: Path) -> _Runtime:
    """1つのagent実装を型・特徴量・libcgの共通実装として読み込む。"""
    resolved = agent_src.resolve()
    if not (resolved / "cg" / "sim.py").exists():
        raise FileNotFoundError(f"cg SDKが見つかりません: {resolved}")

    # batched backendでは全agentがこの1つのcg/rl_mctsモジュールを共有する。
    # legacy backendのagentローダーとは同じプロセスで併用しない。
    sys.path.insert(0, str(resolved))
    try:
        import cg.api as cg_api
        from cg.api import search_begin, search_end, to_observation_class
        from cg.sim import lib
        if torch is None:
            _install_lightweight_model_shim(resolved)
        from rl_mcts.features import get_decoder_input, get_encoder_input
        if torch is None:
            create_model = None
        else:
            from rl_mcts.model import create_model
    finally:
        try:
            sys.path.remove(str(resolved))
        except ValueError:
            pass

    # SearchStepは1回ごとにC++の返値をJSONへ変換し、さらに多数のdataclassを
    # 構築する。SDKを変更せずに内訳を測るため、cg.api内の同じ変換処理を
    # 計時版へ一度だけ差し替える。CPU workerは単一threadなのでactiveで対象を
    # SearchStep呼び出し中だけに限定できる。
    timing = getattr(cg_api, "_batched_search_step_timing", None)
    if timing is None:
        timing = _SearchStepTiming()
        original_json_to_dataclass = cg_api.json_to_dataclass
        fast_api_decoder = _create_fast_api_decoder()

        def profiled_json_to_dataclass(bs: bytes, cls: type) -> Any:
            if fast_api_decoder is not None and cls is cg_api.ApiResult:
                started = time.perf_counter()
                try:
                    result = fast_api_decoder.decode(bs)
                except msgspec.DecodeError:
                    timing.fast_decode_fallbacks += 1
                else:
                    if timing.active > 0:
                        timing.json_seconds += time.perf_counter() - started
                    return result

            if timing.active <= 0:
                return original_json_to_dataclass(bs, cls)

            started = time.perf_counter()
            decoded = json.loads(bs.decode())
            timing.json_seconds += time.perf_counter() - started

            started = time.perf_counter()
            result = cg_api.to_dataclass(decoded, cls)
            timing.dataclass_seconds += time.perf_counter() - started
            return result

        def profiled_search_step(search_id: int, select: list[int]) -> Any:
            timing.active += 1
            try:
                return cg_api.search_step(search_id, select)
            finally:
                timing.active -= 1

        cg_api.json_to_dataclass = profiled_json_to_dataclass
        cg_api._batched_search_step_timing = timing
        cg_api._batched_profiled_search_step = profiled_search_step

    return _Runtime(
        lib=lib,
        to_observation_class=lambda observation: _to_search_observation(
            observation,
            to_observation_class,
        ),
        search_begin=search_begin,
        search_step=cg_api._batched_profiled_search_step,
        search_end=search_end,
        get_encoder_input=get_encoder_input,
        get_decoder_input=get_decoder_input,
        create_model=create_model,
        search_step_timing=timing,
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
    *,
    load_models: bool = True,
) -> dict[str, _Participant]:
    participants: dict[str, _Participant] = {}
    model_cache: dict[str, torch.nn.Module] = {}

    def load_model(model_path: Path) -> tuple[str, torch.nn.Module | None]:
        model_key = str(model_path.resolve())
        model = model_cache.get(model_key) if load_models else None
        if load_models and model is None:
            model = runtime.create_model()
            state = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state)
            model.eval()
            model.to(device)
            model_cache[model_key] = model
        return model_key, model

    for spec in specs:
        agent_path = Path(spec.agent_path).resolve()
        deck_path = Path(spec.deck_path).resolve()
        # tuple化しておくと特徴量側のdeck cache keyを評価ごとに再構築しない。
        deck = tuple(_read_deck(deck_path))
        model_path = agent_path.parent / "model.pth"

        if model_path.exists():
            model_key, model = load_model(model_path)
            opponent_model_path = agent_path.parent / "opponent_model.pth"
            if opponent_model_path.exists():
                opponent_model_key, opponent_model = load_model(opponent_model_path)
            else:
                # 別学習済みcheckpointが用意されるまでは従来modelへfallbackする。
                opponent_model_key, opponent_model = model_key, model
            participants[spec.name] = _Participant(
                name=spec.name,
                deck=deck,
                model=model,
                model_key=model_key,
                random_policy=False,
                opponent_model=opponent_model,
                opponent_model_key=opponent_model_key,
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
    observation = _decode_json_dict(serial.json)
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
    session = _MatchSession(
        request=request,
        players=players,
        battle_ptr=battle_ptr,
        observation=observation,
        select_player=select_player,
    )
    update_seen_opponent_cards(observation, session.seen_opponent_cards)
    return session


@lru_cache(maxsize=512)
def _enumerate_actions(
    option_count: int,
    select_count: int,
    limit: int = MAX_ACTIONS,
) -> list[list[int]]:
    """列挙結果を共有するため、返値とその内側のlistは変更してはいけない。"""
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


def _training_card_id(card: dict[str, Any] | None) -> int | None:
    return None if card is None else int(card["id"])


def _minimal_training_pokemon(pokemon: dict[str, Any] | None) -> dict[str, Any] | None:
    if pokemon is None:
        return None
    return {
        "id": int(pokemon["id"]),
        "hp": int(pokemon["hp"]),
        "energyCards": [
            _training_card_id(card) for card in pokemon["energyCards"]
        ],
        "tools": [_training_card_id(card) for card in pokemon["tools"]],
    }


def _minimal_training_player(player: dict[str, Any]) -> dict[str, Any]:
    hand = player["hand"]
    return {
        "active": [
            _minimal_training_pokemon(pokemon) for pokemon in player["active"]
        ],
        "bench": [
            _minimal_training_pokemon(pokemon) for pokemon in player["bench"]
        ],
        "deckCount": int(player["deckCount"]),
        "discard": [_training_card_id(card) for card in player["discard"]],
        "prize": [_training_card_id(card) for card in player["prize"]],
        "handCount": int(player["handCount"]),
        "hand": (
            None
            if hand is None
            else [_training_card_id(card) for card in hand]
        ),
        "poisoned": bool(player["poisoned"]),
        "burned": bool(player["burned"]),
        "asleep": bool(player["asleep"]),
        "paralyzed": bool(player["paralyzed"]),
        "confused": bool(player["confused"]),
    }


def _minimal_training_observation(observation: dict[str, Any]) -> dict[str, Any]:
    current = observation["current"]
    select = observation["select"]
    looking = current["looking"]
    select_deck = select["deck"]
    return {
        "turn": int(current["turn"]),
        "firstPlayer": int(current["firstPlayer"]),
        "stadium": [_training_card_id(card) for card in current["stadium"]],
        "looking": (
            None
            if looking is None
            else [_training_card_id(card) for card in looking]
        ),
        "players": [
            _minimal_training_player(player) for player in current["players"]
        ],
        "select": {
            "context": int(select["context"]),
            "maxCount": int(select["maxCount"]),
            "option": [
                {
                    key: option[key]
                    for key in _TRAINING_OPTION_FIELDS
                    if key in option and option[key] is not None
                }
                for option in select["option"]
            ],
            "deck": (
                None
                if select_deck is None
                else [_training_card_id(card) for card in select_deck]
            ),
        },
    }


def _build_training_decision(
    observation: dict[str, Any],
    selected_action: list[int],
) -> dict[str, Any] | None:
    select = observation.get("select")
    current = observation.get("current")
    if select is None or current is None:
        return None

    actions = _enumerate_actions(
        len(select["option"]),
        int(select["maxCount"]),
    )
    target = tuple(sorted(selected_action))
    chosen_index = next(
        (
            index
            for index, candidate in enumerate(actions)
            if tuple(candidate) == target
        ),
        None,
    )
    if chosen_index is None:
        return None
    return {
        "player": int(current["yourIndex"]),
        "observation": _minimal_training_observation(observation),
        "chosenIndex": chosen_index,
    }


def _safe_training_name(value: str) -> str:
    visible = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )[:40]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{visible or 'agent'}_{digest}"


def _write_training_episode(session: _MatchSession, output_dir: Path) -> Path:
    result = _raw_result(session)
    if result not in (0, 1, 2):
        raise ValueError("終局していない試合は学習JSONへ保存できません。")
    rewards = [0, 0]
    if result in (0, 1):
        rewards[result] = 1
        rewards[1 - result] = -1

    episode = {
        "format": TRAINING_EPISODE_FORMAT,
        "rewards": rewards,
        "decks": [list(player.deck) for player in session.players],
        "decisions": session.training_decisions,
    }
    request = session.request
    matchup = hashlib.sha1(
        f"{request.name0}\0{request.name1}".encode("utf-8")
    ).hexdigest()[:8]
    filename = (
        f"episode_{_safe_training_name(request.name0)}_vs_"
        f"{_safe_training_name(request.name1)}_{matchup}_"
        f"g{request.game_index:06d}_s{int(request.swap)}.json"
    )
    output_path = output_dir / filename
    temporary_path = output_path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(episode, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return output_path


def _pad_sparse_offsets(sparse: Any, target: int) -> None:
    """EmbeddingBagの空bagを末尾へ足し、全局面の候補手数を揃える。"""
    if len(sparse.offset) > target:
        raise ValueError(f"decoder action数が上限を超えました: {len(sparse.offset)} > {target}")
    sparse.offset.extend([len(sparse.index)] * (target - len(sparse.offset)))


def _evaluation_model_key(context: _SearchContext, player_index: int) -> str:
    participant = context.participant
    if player_index != context.your_index:
        model_key = participant.opponent_model_key or participant.model_key
    else:
        model_key = participant.model_key
    if model_key is None:
        raise RuntimeError("NN評価要求にモデルがありません。")
    return model_key


def _evaluation_model(request: _EvalRequest) -> torch.nn.Module:
    participant = request.context.participant
    if request.model_key == participant.opponent_model_key:
        model = participant.opponent_model
    else:
        model = participant.model
    if model is None:
        raise RuntimeError(f"NNモデルがありません: {request.model_key}")
    return model


def _prepare_node(
    runtime: _Runtime,
    context: _SearchContext,
    parent: _Node | None,
    search_state: Any,
    profile: BatchedProfile,
) -> tuple[_Node, _EvalRequest | None]:
    observation = search_state.observation
    state = observation.current
    node = _Node(
        parent,
        int(search_state.searchId),
        int(state.yourIndex),
        int(state.result),
    )

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
    full_deck = context.participant.deck
    if context.full_decks is not None:
        full_deck = context.full_decks[int(state.yourIndex)]
    encoder = runtime.get_encoder_input(observation, full_deck)
    decoder = runtime.get_decoder_input(observation, actions)
    profile.feature_seconds += time.perf_counter() - started
    return node, _EvalRequest(
        context,
        node,
        actions,
        encoder,
        decoder,
        _evaluation_model_key(context, int(state.yourIndex)),
    )


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


def _combine_sparse_numpy(
    vectors: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SparseVector群をQueue転送向けの連続typed bufferへまとめる。"""
    indices, values, offsets = _combine_sparse(vectors)
    return (
        np.asarray(indices, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
        np.asarray(offsets, dtype=np.int32),
    )


def _snapshot_sparse(sparse: Any, offset_count: int | None = None) -> _SparseVectorSnapshot:
    offsets = sparse.offset if offset_count is None else sparse.offset[:offset_count]
    return _SparseVectorSnapshot(
        index=list(sparse.index),
        value=list(sparse.value),
        offset=list(offsets),
    )


def _prepare_evaluation_chunk(
    chunk: list[_EvalRequest],
    *,
    numpy_output: bool = False,
) -> tuple[Any, Any]:
    """同一モデルの要求を1回のforward入力へまとめる。"""
    required_decoder_words = max(len(request.decoder.offset) for request in chunk)
    # MPS/CUDAでshapeの種類を抑えるため1,2,4,...,64のbucketへ丸める。
    decoder_words = 1 << (required_decoder_words - 1).bit_length()
    for request in chunk:
        _pad_sparse_offsets(request.decoder, decoder_words)
    combine = _combine_sparse_numpy if numpy_output else _combine_sparse
    encoder = combine([request.encoder for request in chunk])
    decoder = combine([request.decoder for request in chunk])
    return encoder, decoder


def _merge_combined_sparse(
    vectors: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """複数workerが既に結合したSparseVectorを中央batchへ再結合する。"""
    index_count = sum(len(vector[0]) for vector in vectors)
    offset_count = sum(len(vector[2]) for vector in vectors)
    indices = np.empty(index_count, dtype=np.int32)
    values = np.empty(index_count, dtype=np.float32)
    offsets = np.empty(offset_count, dtype=np.int32)
    index_position = 0
    offset_position = 0
    for vector_indices, vector_values, vector_offsets in vectors:
        next_index_position = index_position + len(vector_indices)
        next_offset_position = offset_position + len(vector_offsets)
        indices[index_position:next_index_position] = vector_indices
        values[index_position:next_index_position] = vector_values
        offsets[offset_position:next_offset_position] = (
            vector_offsets + index_position
        )
        index_position = next_index_position
        offset_position = next_offset_position
    return indices, values, offsets


def _pad_remote_decoder(
    job: _RemoteEvalJob,
    target_words: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """worker内で結合済みdecoderの各局面末尾へ空bagを追加する。"""
    if job.decoder_words > target_words:
        raise ValueError("target_wordsがworker decoder幅より小さいです。")
    if job.decoder_words == target_words:
        return job.decoder

    indices, values, offsets = job.decoder
    padded_offsets = np.empty(job.batch_count * target_words, dtype=np.int32)
    for row in range(job.batch_count):
        start = row * job.decoder_words
        end = start + job.decoder_words
        output_start = row * target_words
        output_end = output_start + job.decoder_words
        padded_offsets[output_start:output_end] = offsets[start:end]
        row_value_end = offsets[end] if end < len(offsets) else len(indices)
        padded_offsets[output_end : output_start + target_words] = row_value_end
    return indices, values, padded_offsets


class _RemoteEvaluationClient:
    """CPU worker内でNN要求を中央batcherへ転送する同期client。"""

    def __init__(
        self,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.request_id = 0

    def evaluate(
        self,
        requests: list[_EvalRequest],
        batch_size: int,
        profile: BatchedProfile,
    ) -> None:
        grouped: dict[str, list[_EvalRequest]] = defaultdict(list)
        for request in requests:
            grouped[request.model_key].append(request)
        if not grouped:
            return

        jobs: list[_RemoteEvalJob] = []
        chunks: list[list[_EvalRequest]] = []
        numpy_pack_started = time.perf_counter()
        for model_key, model_requests in grouped.items():
            for offset in range(0, len(model_requests), batch_size):
                chunk = model_requests[offset : offset + batch_size]
                encoder, decoder = _prepare_evaluation_chunk(
                    chunk,
                    numpy_output=True,
                )
                decoder_words = len(decoder[2]) // len(chunk)
                jobs.append(
                    _RemoteEvalJob(
                        model_key=model_key,
                        batch_count=len(chunk),
                        decoder_words=decoder_words,
                        encoder=encoder,
                        decoder=decoder,
                    )
                )
                chunks.append(chunk)
        profile.remote_numpy_pack_seconds += (
            time.perf_counter() - numpy_pack_started
        )

        request_id = self.request_id
        self.request_id += 1
        started = time.perf_counter()
        self.request_queue.put((self.worker_id, request_id, jobs))
        response_id, responses = self.response_queue.get()
        profile.remote_wait_seconds += time.perf_counter() - started
        profile.ipc_messages += 1
        if response_id != request_id:
            raise RuntimeError(
                f"NN応答IDが不一致です: expected={request_id}, actual={response_id}"
            )
        if len(responses) != len(chunks):
            raise RuntimeError("中央NN batcherの応答数が一致しません。")

        for chunk, response in zip(chunks, responses, strict=True):
            value_rows, policy_rows = response
            _commit_evaluation_rows(chunk, value_rows, policy_rows)


def _commit_evaluation_rows(
    chunk: list[_EvalRequest],
    value_rows: list[list[float]],
    policy_rows: list[list[float]],
) -> None:
    """device評価結果をCPU側のMCTS nodeへ反映する。"""
    for request, value_row, policy_row in zip(
        chunk, value_rows, policy_rows, strict=True
    ):
        value = float(value_row[0])
        if request.node.parent is None and not request.context.policy_only:
            request.context.root_sample = BatchedLearnSample(
                value=value,
                policy=[
                    float(policy_row[index])
                    for index in range(len(request.actions))
                ],
                sv_enc=_snapshot_sparse(request.encoder),
                # decoderはforward前にbucket幅までpaddingされているため、
                # 実際の合法手数へ戻して保存する。
                sv_dec=_snapshot_sparse(
                    request.decoder,
                    offset_count=len(request.actions),
                ),
            )
        propagated = value
        if request.node.player_index != request.context.your_index:
            propagated = -propagated
        request.node.value = propagated
        request.node.backprop(propagated)

        probabilities = [
            math.exp(float(policy_row[index]) * 10.0)
            for index in range(len(request.actions))
        ]
        probability_sum = sum(probabilities)
        for action, probability in zip(
            request.actions, probabilities, strict=True
        ):
            if probability_sum > 0:
                probability /= probability_sum
            request.node.children.append(_Child(action, probability))
        if request.reserved_child is not None:
            request.reserved_child.in_flight = False


def _apply_evaluations_cuda_streams(
    grouped: dict[str, list[_EvalRequest]],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    cuda_streams: dict[str, torch.cuda.Stream],
) -> None:
    """異なるモデルのbatchを別CUDA streamへ非同期投入する。

    各モデルの入力転送、forward、出力転送を専用streamに並べ、全モデルを
    投入してからcurrent streamを一度だけ同期する。モデルごとの`.cpu()`同期を
    行わないため、GPUが複数モデルの小さいkernelを並行scheduleできる。
    """
    if not grouped:
        return

    started = time.perf_counter()
    current_stream = torch.cuda.current_stream(device)
    jobs: list[_CudaEvalJob] = []
    used_streams: dict[str, torch.cuda.Stream] = {}

    for model_key, model_requests in grouped.items():
        model = _evaluation_model(model_requests[0])
        stream = cuda_streams.setdefault(model_key, torch.cuda.Stream(device=device))
        # modelのdefault stream上での初期化と、前waveのcurrent stream処理を待つ。
        stream.wait_stream(current_stream)
        used_streams[model_key] = stream

        for offset in range(0, len(model_requests), batch_size):
            chunk = model_requests[offset : offset + batch_size]
            encoder, decoder = _prepare_evaluation_chunk(chunk)
            host_inputs = [
                torch.tensor(values, dtype=dtype, pin_memory=True)
                for values, dtype in (
                    (encoder[0], torch.int32),
                    (encoder[1], torch.float32),
                    (encoder[2], torch.int32),
                    (decoder[0], torch.int32),
                    (decoder[1], torch.float32),
                    (decoder[2], torch.int32),
                )
            ]

            with torch.cuda.stream(stream), torch.inference_mode():
                cuda_inputs = [
                    tensor.to(device, non_blocking=True) for tensor in host_inputs
                ]
                values, policies = model(*cuda_inputs)
                value_rows = torch.empty(
                    values.shape,
                    dtype=values.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                policy_rows = torch.empty(
                    policies.shape,
                    dtype=policies.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                value_rows.copy_(values, non_blocking=True)
                policy_rows.copy_(policies, non_blocking=True)

            jobs.append(
                _CudaEvalJob(
                    chunk=chunk,
                    pinned_inputs=host_inputs,
                    value_rows=value_rows,
                    policy_rows=policy_rows,
                )
            )

    # 全モデルを投入した後に一度だけCPUを待たせる。
    for stream in used_streams.values():
        current_stream.wait_stream(stream)
    current_stream.synchronize()
    profile.nn_seconds += time.perf_counter() - started

    for job in jobs:
        profile.nn_evaluations += len(job.chunk)
        profile.nn_batches += 1
        profile.batch_sizes.append(len(job.chunk))
        _commit_evaluation_rows(
            job.chunk,
            job.value_rows.tolist(),
            job.policy_rows.tolist(),
        )


def _pad_combined_sparse(
    combined: tuple[Any, Any, Any],
    target_values: int,
) -> tuple[Any, Any, Any]:
    """vmapでmodelごとのflatten長を揃えるためzero-weight要素を足す。"""
    missing = target_values - len(combined[0])
    if missing < 0:
        raise ValueError("target_valuesが現在のsparse長より小さいです。")
    if missing == 0:
        return combined
    if isinstance(combined[0], np.ndarray):
        return (
            np.pad(combined[0], (0, missing), constant_values=0),
            np.pad(combined[1], (0, missing), constant_values=0.0),
            combined[2],
        )
    return (
        combined[0] + [0] * missing,
        combined[1] + [0.0] * missing,
        combined[2],
    )


def _apply_evaluations_cuda_ensemble(
    grouped: dict[str, list[_EvalRequest]],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    evaluator: _CudaEnsembleEvaluator,
) -> None:
    """8モデルの重みと入力をmodel次元へ積み、vmapで一括評価する。"""
    if not grouped:
        return

    chunked = {
        model_key: [
            model_requests[offset : offset + batch_size]
            for offset in range(0, len(model_requests), batch_size)
        ]
        for model_key, model_requests in grouped.items()
    }
    wave_count = max(len(chunks) for chunks in chunked.values())

    for wave_index in range(wave_count):
        active_chunks = {
            model_key: chunks[wave_index]
            for model_key, chunks in chunked.items()
            if wave_index < len(chunks)
        }
        template_request = next(iter(active_chunks.values()))[0]
        model_batch = max(len(chunk) for chunk in active_chunks.values())
        required_decoder_words = max(
            len(request.decoder.offset)
            for chunk in active_chunks.values()
            for request in chunk
        )
        decoder_words = 1 << (required_decoder_words - 1).bit_length()
        for chunk in active_chunks.values():
            for request in chunk:
                _pad_sparse_offsets(request.decoder, decoder_words)

        template_encoder = _snapshot_sparse(template_request.encoder)
        template_decoder = _snapshot_sparse(template_request.decoder)
        combined_rows: list[
            tuple[
                tuple[list[int], list[float], list[int]],
                tuple[list[int], list[float], list[int]],
            ]
        ] = []
        for model_key in evaluator.model_keys:
            chunk = active_chunks.get(model_key, [])
            encoders: list[Any] = [request.encoder for request in chunk]
            decoders: list[Any] = [request.decoder for request in chunk]
            while len(encoders) < model_batch:
                encoders.append(template_encoder)
                decoders.append(template_decoder)
            combined_rows.append(
                (_combine_sparse(encoders), _combine_sparse(decoders))
            )

        encoder_values = max(len(row[0][0]) for row in combined_rows)
        decoder_values = max(len(row[1][0]) for row in combined_rows)
        combined_rows = [
            (
                _pad_combined_sparse(encoder, encoder_values),
                _pad_combined_sparse(decoder, decoder_values),
            )
            for encoder, decoder in combined_rows
        ]

        started = time.perf_counter()
        inputs = (
            torch.tensor(
                [row[0][0] for row in combined_rows],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(
                [row[0][1] for row in combined_rows],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [row[0][2] for row in combined_rows],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(
                [row[1][0] for row in combined_rows],
                dtype=torch.int32,
                device=device,
            ),
            torch.tensor(
                [row[1][1] for row in combined_rows],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [row[1][2] for row in combined_rows],
                dtype=torch.int32,
                device=device,
            ),
        )
        with torch.inference_mode():
            values, policies = evaluator.evaluate(*inputs)
            value_rows = values.detach().cpu().tolist()
            policy_rows = policies.detach().cpu().tolist()
        profile.nn_seconds += time.perf_counter() - started

        for model_key, chunk in active_chunks.items():
            model_index = evaluator.model_indices[model_key]
            profile.nn_evaluations += len(chunk)
            profile.nn_batches += 1
            profile.batch_sizes.append(len(chunk))
            _commit_evaluation_rows(
                chunk,
                value_rows[model_index][: len(chunk)],
                policy_rows[model_index][: len(chunk)],
            )


def _apply_evaluations(
    requests: list[_EvalRequest],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    cuda_streams: dict[str, torch.cuda.Stream] | None = None,
    cuda_ensemble: _CudaEnsembleEvaluator | None = None,
    remote_evaluator: _RemoteEvaluationClient | None = None,
) -> None:
    if remote_evaluator is not None:
        remote_evaluator.evaluate(requests, batch_size, profile)
        return

    grouped: dict[str, list[_EvalRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.model_key].append(request)

    mean_model_batch = len(requests) / max(len(grouped), 1)
    if (
        cuda_ensemble is not None
        and len(grouped) >= 2
        and mean_model_batch
        < _model_axis_batch_crossover(device, cuda_ensemble, len(grouped))
    ):
        profile.cuda_ensemble_waves += 1
        _apply_evaluations_cuda_ensemble(
            grouped, device, batch_size, profile, cuda_ensemble
        )
        return
    if cuda_ensemble is not None:
        profile.cuda_per_model_waves += 1
    if cuda_streams is not None:
        _apply_evaluations_cuda_streams(
            grouped, device, batch_size, profile, cuda_streams
        )
        return

    for model_requests in grouped.values():
        model = _evaluation_model(model_requests[0])
        for offset in range(0, len(model_requests), batch_size):
            chunk = model_requests[offset : offset + batch_size]
            encoder, decoder = _prepare_evaluation_chunk(chunk)

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

            _commit_evaluation_rows(chunk, value_rows, policy_rows)


def _begin_search(
    runtime: _Runtime,
    context: _SearchContext,
    profile: BatchedProfile,
) -> _EvalRequest | None:
    observation = runtime.to_observation_class(context.session.observation)
    belief = sample_hidden_zones(
        context.session.observation,
        context.your_index,
        context.participant.deck,
        context.session.seen_opponent_cards[context.your_index].values(),
    )
    context.full_decks = belief.full_decks

    started = time.perf_counter()
    search_state = runtime.search_begin(
        observation,
        your_deck=belief.your_deck,
        your_prize=belief.your_prize,
        opponent_deck=belief.opponent_deck,
        opponent_prize=belief.opponent_prize,
        opponent_hand=belief.opponent_hand,
        opponent_active=belief.opponent_active,
    )
    profile.search_begin_seconds += time.perf_counter() - started
    root, request = _prepare_node(runtime, context, None, search_state, profile)
    context.root = root
    return request


def _is_setup_context(context: _SearchContext) -> bool:
    observation = context.session.observation
    select = observation.get("select") or {}
    if int(select.get("context", -1)) in SETUP_SELECT_CONTEXTS:
        return True
    state = observation.get("current") or {}
    return any(
        any(pokemon is None for pokemon in (player.get("active") or []))
        for player in (state.get("players") or [])
    )


def _prepare_policy_only_root(
    runtime: _Runtime,
    context: _SearchContext,
    profile: BatchedProfile,
) -> _EvalRequest | None:
    """セットアップをSearchBeginなしの1-step policyで選ぶ。"""
    observation = runtime.to_observation_class(context.session.observation)
    state = observation.current
    node = _Node(None, -1, int(state.yourIndex), int(state.result))
    context.root = node
    context.policy_only = True
    actions = _enumerate_actions(
        len(observation.select.option),
        observation.select.maxCount,
    )
    if not actions:
        return None
    started = time.perf_counter()
    encoder = runtime.get_encoder_input(observation, context.participant.deck)
    decoder = runtime.get_decoder_input(observation, actions)
    profile.feature_seconds += time.perf_counter() - started
    return _EvalRequest(
        context,
        node,
        actions,
        encoder,
        decoder,
        _evaluation_model_key(context, int(state.yourIndex)),
    )


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
            if current.player_index != context.your_index:
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
            json_before, dataclass_before = runtime.search_step_timing.snapshot()
            started = time.perf_counter()
            next_state = runtime.search_step(current.search_id, best_child.select)
            elapsed = time.perf_counter() - started
            json_after, dataclass_after = runtime.search_step_timing.snapshot()
            json_elapsed = json_after - json_before
            dataclass_elapsed = dataclass_after - dataclass_before
            profile.search_step_seconds += elapsed
            profile.search_step_json_seconds += json_elapsed
            profile.search_step_dataclass_seconds += dataclass_elapsed
            profile.search_step_c_api_seconds += max(
                0.0,
                elapsed - json_elapsed - dataclass_elapsed,
            )
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
        if current.result >= 0:
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
    min_value = 10.0
    for child in root.children:
        if child.node is None:
            continue
        if child.node.visit > max_visit:
            max_child = child
            max_visit = child.node.visit
        min_value = min(
            min_value,
            child.node.total / max(child.node.visit, 1),
        )
    if max_child is None:
        max_child = max(root.children, key=lambda child: child.probability)

    sample = context.root_sample
    if sample is not None:
        sample.value = root.total / max(root.visit, 1)
        if min_value == 10.0:
            min_value = sample.value
        for index, child in enumerate(root.children):
            value = sample.value
            if child.node is None:
                value = min_value - value - 0.03
            else:
                value = child.node.total / max(child.node.visit, 1) - value
            sample.policy[index] = max(-1.0, min(1.0, value))
    return max_child.select


def _run_search_wave(
    runtime: _Runtime,
    contexts: list[_SearchContext],
    search_count: int,
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    sample_sink: dict[int, BatchedLearnSample] | None = None,
    cuda_streams: dict[str, torch.cuda.Stream] | None = None,
    cuda_ensemble: _CudaEnsembleEvaluator | None = None,
    remote_evaluator: _RemoteEvaluationClient | None = None,
) -> dict[int, list[int]]:
    actions: dict[int, list[int]] = {}
    if not contexts:
        return actions

    search_started = False
    try:
        root_requests: list[_EvalRequest] = []
        tree_contexts: list[_SearchContext] = []
        for context in contexts:
            if _is_setup_context(context):
                request = _prepare_policy_only_root(runtime, context, profile)
            else:
                request = _begin_search(runtime, context, profile)
                tree_contexts.append(context)
                search_started = True
            if request is not None:
                root_requests.append(request)
        _apply_evaluations(
            root_requests,
            device,
            batch_size,
            profile,
            cuda_streams,
            cuda_ensemble,
            remote_evaluator,
        )

        remaining = {id(context): search_count for context in tree_contexts}
        while any(count > 0 for count in remaining.values()):
            leaf_requests: list[_EvalRequest] = []
            made_progress = False
            for context in tree_contexts:
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
            _apply_evaluations(
                leaf_requests,
                device,
                batch_size,
                profile,
                cuda_streams,
                cuda_ensemble,
                remote_evaluator,
            )
            if not made_progress:
                break

        for context in contexts:
            actions[context.session.battle_ptr] = _finish_search(context)
            if sample_sink is not None and context.root_sample is not None:
                sample_sink[context.session.battle_ptr] = context.root_sample
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


def _raw_turn(session: _MatchSession) -> int:
    current = session.observation.get("current") or {}
    return int(current.get("turn", 0))


def _set_result_from_remaining_prizes(session: _MatchSession) -> int:
    """上限到達時の勝敗を残りサイド枚数で確定する。

    残りサイドが少ない側を勝者とし、同数なら引き分けにする。返すresultは
    Battle内部のplayer index基準で、通常終局時のcurrent.resultと同じ形式。
    """
    current = session.observation.get("current")
    if current is None:
        raise ValueError("現在の対戦状態がないためサイド枚数を比較できません。")
    players = current.get("players") or []
    if len(players) != 2:
        raise ValueError("プレイヤー2人分のサイド枚数を取得できません。")

    remaining_prizes = [len(player.get("prize") or []) for player in players]
    if remaining_prizes[0] == remaining_prizes[1]:
        result = 2
    else:
        result = 0 if remaining_prizes[0] < remaining_prizes[1] else 1
    current["result"] = result
    return result


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
    max_turns: int | None = None,
    seed: int = 0,
    parallel_cuda_models: bool = False,
    cuda_ensemble_models: bool = False,
    training_json_dir: Path | None = None,
    _evaluation_client: _RemoteEvaluationClient | None = None,
    _game_requests: list[BatchedGameRequest] | None = None,
    _load_worker_models: bool = True,
) -> BatchedTournamentOutput:
    """総当たりの全試合をlane pool上で進める。"""
    if batch_size < 1:
        raise ValueError("batch_sizeは1以上で指定してください。")
    if lanes < 1:
        raise ValueError("lanesは1以上で指定してください。")
    if search_count < 0:
        raise ValueError("search_countは0以上で指定してください。")
    if max_selections < 1:
        raise ValueError("max_selectionsは1以上で指定してください。")
    if max_turns is not None and max_turns < 1:
        raise ValueError("max_turnsは1以上またはNoneにしてください。")

    training_output_dir = (
        Path(training_json_dir).resolve()
        if training_json_dir is not None
        else None
    )
    if training_output_dir is not None:
        training_output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    device = select_device(device_name)
    if parallel_cuda_models and device.type != "cuda":
        raise ValueError("parallel_cuda_modelsにはCUDA deviceが必要です。")
    if cuda_ensemble_models and device.type != "cuda":
        raise ValueError("cuda_ensemble_modelsにはCUDA deviceが必要です。")
    if parallel_cuda_models and cuda_ensemble_models:
        raise ValueError("CUDA streamsとCUDA ensembleは同時に指定できません。")
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
    participants = _load_participants(
        specs,
        runtime,
        device,
        load_models=_load_worker_models,
    )
    cuda_streams: dict[str, torch.cuda.Stream] | None = None
    if parallel_cuda_models:
        cuda_streams = {
            model_key: torch.cuda.Stream(device=device)
            for participant in participants.values()
            for model_key in (
                participant.model_key,
                participant.opponent_model_key,
            )
            if model_key is not None
        }
        # model.to(cuda)を行ったdefault streamの初期化を完了してから専用streamを使う。
        torch.cuda.synchronize(device)
    cuda_ensemble: _CudaEnsembleEvaluator | None = None
    if cuda_ensemble_models:
        cuda_ensemble = _CudaEnsembleEvaluator(participants.values(), device)
        torch.cuda.synchronize(device)
    profile = BatchedProfile()

    requests = deque(
        _game_requests
        if _game_requests is not None
        else (
            BatchedGameRequest(
                name0=name0,
                name1=name1,
                game_index=game_index + 1,
                swap=alternate_sides and game_index % 2 == 1,
            )
            for name0, name1 in pairings
            for game_index in range(num_games)
        )
    )
    active: list[_MatchSession] = []
    results: list[BatchedGameResult] = []

    def finish_session(
        session: _MatchSession,
        error: str | None = None,
    ) -> None:
        try:
            if error is None and training_output_dir is not None:
                _write_training_episode(session, training_output_dir)
            results.append(_to_result(session, error=error))
        finally:
            runtime.lib.BattleFinish(session.battle_ptr)

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
                    finish_session(session)
                elif max_turns is not None and _raw_turn(session) >= max_turns:
                    _set_result_from_remaining_prizes(session)
                    finish_session(session)
                elif session.selections >= max_selections:
                    _set_result_from_remaining_prizes(session)
                    finish_session(session)
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
                    cuda_streams=cuda_streams,
                    cuda_ensemble=cuda_ensemble,
                    remote_evaluator=_evaluation_client,
                )
            )

            next_active: list[_MatchSession] = []
            for session in active:
                action = selected_actions[session.battle_ptr]
                if training_output_dir is not None:
                    decision = _build_training_decision(
                        session.observation,
                        action,
                    )
                    if decision is not None:
                        session.training_decisions.append(decision)
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
    return BatchedTournamentOutput(
        results=results,
        profile=profile,
        device=str(device),
    )


def _parallel_worker_main(
    worker_id: int,
    spec_records: list[tuple[str, str, str]],
    game_requests: list[BatchedGameRequest],
    lanes: int,
    search_count: int,
    max_selections: int,
    max_turns: int | None,
    seed: int,
    remote_batch_size: int,
    training_json_dir: str | None,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    """libcgと特徴量生成を担当するspawn workerのentry point。"""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    if torch is not None:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    specs = [
        SimpleNamespace(
            name=name,
            agent_path=Path(agent_path),
            deck_path=Path(deck_path),
        )
        for name, agent_path, deck_path in spec_records
    ]
    client = _RemoteEvaluationClient(worker_id, request_queue, response_queue)
    try:
        output = run_batched_tournament(
            specs=specs,
            pairings=[],
            num_games=0,
            device_name="cpu",
            batch_size=remote_batch_size,
            lanes=min(lanes, len(game_requests)),
            search_count=search_count,
            max_selections=max_selections,
            max_turns=max_turns,
            seed=seed + worker_id,
            training_json_dir=(
                Path(training_json_dir)
                if training_json_dir is not None
                else None
            ),
            _evaluation_client=client,
            _game_requests=game_requests,
            _load_worker_models=False,
        )
        result_queue.put(("done", worker_id, output))
    except BaseException:  # noqa: BLE001 - 親processへtracebackを返す
        result_queue.put(("error", worker_id, traceback.format_exc()))


def _mps_sparse_inputs(
    encoder: tuple[np.ndarray, np.ndarray, np.ndarray],
    decoder: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    """6本のsparse配列をdtype別の2転送にまとめ、MPS上でviewへ戻す。"""
    integer_arrays = (encoder[0], encoder[2], decoder[0], decoder[2])
    float_arrays = (encoder[1], decoder[1])
    integer_sizes = [array.size for array in integer_arrays]
    float_sizes = [array.size for array in float_arrays]
    integer_device = torch.from_numpy(
        np.concatenate([array.reshape(-1) for array in integer_arrays])
    ).to(device)
    float_device = torch.from_numpy(
        np.concatenate([array.reshape(-1) for array in float_arrays])
    ).to(device)

    integer_tensors: list[torch.Tensor] = []
    position = 0
    for array, size in zip(integer_arrays, integer_sizes, strict=True):
        integer_tensors.append(
            integer_device[position : position + size].reshape(array.shape)
        )
        position += size
    float_tensors: list[torch.Tensor] = []
    position = 0
    for array, size in zip(float_arrays, float_sizes, strict=True):
        float_tensors.append(
            float_device[position : position + size].reshape(array.shape)
        )
        position += size
    return (
        integer_tensors[0],
        float_tensors[0],
        integer_tensors[1],
        integer_tensors[2],
        float_tensors[1],
        integer_tensors[3],
    )


def _mps_sparse_input_batches(
    sparse_batches: list[
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
    ],
    device: torch.device,
) -> list[tuple[torch.Tensor, ...]]:
    """同じwaveの全model入力をdtype別の2転送にまとめる。"""
    if not sparse_batches:
        return []

    integer_arrays: list[np.ndarray] = []
    float_arrays: list[np.ndarray] = []
    integer_metadata: list[list[tuple[int, tuple[int, ...]]]] = []
    float_metadata: list[list[tuple[int, tuple[int, ...]]]] = []
    for encoder, decoder in sparse_batches:
        batch_integer_arrays = (encoder[0], encoder[2], decoder[0], decoder[2])
        batch_float_arrays = (encoder[1], decoder[1])
        integer_arrays.extend(batch_integer_arrays)
        float_arrays.extend(batch_float_arrays)
        integer_metadata.append(
            [(array.size, array.shape) for array in batch_integer_arrays]
        )
        float_metadata.append(
            [(array.size, array.shape) for array in batch_float_arrays]
        )

    integer_device = torch.from_numpy(
        np.concatenate([array.reshape(-1) for array in integer_arrays])
    ).to(device)
    float_device = torch.from_numpy(
        np.concatenate([array.reshape(-1) for array in float_arrays])
    ).to(device)

    batches: list[tuple[torch.Tensor, ...]] = []
    integer_position = 0
    float_position = 0
    for batch_integer_metadata, batch_float_metadata in zip(
        integer_metadata,
        float_metadata,
        strict=True,
    ):
        integer_tensors: list[torch.Tensor] = []
        for size, shape in batch_integer_metadata:
            integer_tensors.append(
                integer_device[integer_position : integer_position + size].reshape(shape)
            )
            integer_position += size
        float_tensors: list[torch.Tensor] = []
        for size, shape in batch_float_metadata:
            float_tensors.append(
                float_device[float_position : float_position + size].reshape(shape)
            )
            float_position += size
        batches.append(
            (
                integer_tensors[0],
                float_tensors[0],
                integer_tensors[1],
                integer_tensors[2],
                float_tensors[1],
                integer_tensors[3],
            )
        )
    return batches


def _evaluate_remote_messages_mps_queued(
    grouped: dict[str, list[tuple[int, int, int, _RemoteEvalJob]]],
    responses: dict[tuple[int, int], list[Any]],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
) -> None:
    """per-model forwardを全てMPSへ投入してから、出力をまとめて待つ。

    モデルごとに直後の.cpu()を呼ぶとMPS command queueを毎回同期してしまう。
    NumPyのmergeを先に済ませ、全model/chunkのforwardをsubmitした後で出力を
    CPUへ戻すことで、1つの中央waveにつき実質1回の完了待ちにする。
    """
    prepared: list[
        tuple[
            list[tuple[int, int, int, _RemoteEvalJob]],
            torch.nn.Module,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
            int,
        ]
    ] = []
    for model_key, entries in grouped.items():
        model = models.get(model_key)
        if model is None:
            raise RuntimeError(f"中央NN batcherにモデルがありません: {model_key}")
        entries.sort(key=lambda entry: entry[3].decoder_words)
        central_chunks: list[list[tuple[int, int, int, _RemoteEvalJob]]] = []
        current: list[tuple[int, int, int, _RemoteEvalJob]] = []
        current_size = 0
        for entry in entries:
            job_size = entry[3].batch_count
            if current and current_size + job_size > batch_size:
                central_chunks.append(current)
                current = []
                current_size = 0
            current.append(entry)
            current_size += job_size
        if current:
            central_chunks.append(current)

        for central_chunk in central_chunks:
            merge_started = time.perf_counter()
            jobs = [entry[3] for entry in central_chunk]
            decoder_words = max(job.decoder_words for job in jobs)
            encoder = _merge_combined_sparse([job.encoder for job in jobs])
            decoder = _merge_combined_sparse(
                [_pad_remote_decoder(job, decoder_words) for job in jobs]
            )
            evaluation_count = sum(job.batch_count for job in jobs)
            profile.nn_decoder_source_tokens += sum(
                job.batch_count * job.decoder_words for job in jobs
            )
            profile.nn_decoder_padded_tokens += evaluation_count * decoder_words
            profile.nn_merge_seconds += time.perf_counter() - merge_started
            prepared.append(
                (central_chunk, model, encoder, decoder, evaluation_count)
            )

    submitted: list[
        tuple[
            list[tuple[int, int, int, _RemoteEvalJob]],
            int,
            torch.Tensor,
        ]
    ] = []
    started = time.perf_counter()
    with torch.inference_mode():
        input_started = time.perf_counter()
        input_batches = _mps_sparse_input_batches(
            [(encoder, decoder) for _, _, encoder, decoder, _ in prepared],
            device,
        )
        profile.nn_input_seconds += time.perf_counter() - input_started
        for (
            central_chunk,
            model,
            _encoder,
            _decoder,
            evaluation_count,
        ), inputs in zip(prepared, input_batches, strict=True):
            forward_started = time.perf_counter()
            values, policies = model(*inputs)
            combined_output = torch.cat((values, policies), dim=-1)
            profile.nn_forward_submit_seconds += time.perf_counter() - forward_started
            submitted.append(
                (
                    central_chunk,
                    evaluation_count,
                    combined_output.detach(),
                )
            )

        completed: list[
            tuple[
                list[tuple[int, int, int, _RemoteEvalJob]],
                list[Any],
                list[Any],
            ]
        ] = []
        for central_chunk, evaluation_count, combined_output in submitted:
            output_started = time.perf_counter()
            output_rows_tensor = combined_output.cpu()
            profile.nn_output_wait_seconds += time.perf_counter() - output_started

            tolist_started = time.perf_counter()
            value_rows = output_rows_tensor[:, :1].tolist()
            policy_rows = output_rows_tensor[:, 1:].tolist()
            profile.nn_tolist_seconds += time.perf_counter() - tolist_started
            profile.nn_evaluations += evaluation_count
            profile.nn_batches += 1
            profile.batch_sizes.append(evaluation_count)
            completed.append((central_chunk, value_rows, policy_rows))
    profile.nn_seconds += time.perf_counter() - started

    response_pack_started = time.perf_counter()
    for central_chunk, value_rows, policy_rows in completed:
        row_offset = 0
        for worker_id, request_id, job_index, job in central_chunk:
            row_end = row_offset + job.batch_count
            responses[(worker_id, request_id)][job_index] = (
                value_rows[row_offset:row_end],
                policy_rows[row_offset:row_end],
            )
            row_offset = row_end
    profile.nn_response_pack_seconds += time.perf_counter() - response_pack_started


def _evaluate_remote_messages_mps_active(
    grouped: dict[str, list[tuple[int, int, int, _RemoteEvalJob]]],
    responses: dict[tuple[int, int], list[Any]],
    models: dict[str, torch.nn.Module],
    evaluator: _MpsModelAxisEvaluator,
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
) -> None:
    """固定MPS model-axisか、queued per-modelをwaveごとに選ぶ。"""
    active_model_keys = [
        model_key for model_key in evaluator.model_keys if model_key in grouped
    ]
    active_model_count = len(active_model_keys)
    evaluation_count = sum(
        entry[3].batch_count
        for entries in grouped.values()
        for entry in entries
    )
    mean_model_batch = evaluation_count / max(active_model_count, 1)
    max_model_batch = max(
        sum(entry[3].batch_count for entry in entries)
        for entries in grouped.values()
    )
    dense_row_efficiency = evaluation_count / (
        evaluator.model_count * max_model_batch
    )
    # stackは常に同じ全モデルとし、waveごとの重みindex_selectを避ける。
    # inactive modelのpadding費用を考慮して、active率に比例して閾値を下げる。
    model_axis_crossover = (
        _MPS_MODEL_AXIS_BATCH_CROSSOVER
        * active_model_count
        / evaluator.model_count
    )
    if (
        active_model_count >= 4
        and mean_model_batch < model_axis_crossover
        and dense_row_efficiency >= _MPS_MODEL_AXIS_MIN_ROW_EFFICIENCY
    ):
        profile.cuda_ensemble_waves += 1
        _evaluate_remote_messages_model_axis(
            grouped,
            responses,
            evaluator,
            device,
            batch_size,
            profile,
        )
    else:
        profile.cuda_per_model_waves += 1
        _evaluate_remote_messages_mps_queued(
            grouped,
            responses,
            models,
            device,
            batch_size,
            profile,
        )


def _evaluate_remote_messages(
    messages: list[tuple[int, int, list[_RemoteEvalJob]]],
    models: dict[str, torch.nn.Module],
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    response_queues: list[Any],
    model_axis_evaluator: _CudaEnsembleEvaluator | _MpsModelAxisEvaluator | None = None,
) -> None:
    """複数workerから届いた要求をmodel・decoder幅ごとに中央評価する。"""
    responses: dict[tuple[int, int], list[Any]] = {
        (worker_id, request_id): [None] * len(jobs)
        for worker_id, request_id, jobs in messages
    }
    grouped: dict[str, list[tuple[int, int, int, _RemoteEvalJob]]] = defaultdict(list)
    for worker_id, request_id, jobs in messages:
        for job_index, job in enumerate(jobs):
            grouped[job.model_key].append(
                (worker_id, request_id, job_index, job)
            )

    if isinstance(model_axis_evaluator, _MpsModelAxisEvaluator):
        _evaluate_remote_messages_mps_active(
            grouped,
            responses,
            models,
            model_axis_evaluator,
            device,
            batch_size,
            profile,
        )
        profile.ipc_messages += len(messages)
        response_put_started = time.perf_counter()
        for worker_id, request_id, _jobs in messages:
            response_queues[worker_id].put(
                (request_id, responses[(worker_id, request_id)])
            )
        profile.response_put_seconds += time.perf_counter() - response_put_started
        return

    evaluation_count = sum(
        entry[3].batch_count
        for entries in grouped.values()
        for entry in entries
    )
    mean_model_batch = evaluation_count / max(len(grouped), 1)
    if (
        model_axis_evaluator is not None
        and len(grouped) >= 2
        and mean_model_batch
        < _model_axis_batch_crossover(
            device,
            model_axis_evaluator,
            len(grouped),
        )
    ):
        profile.cuda_ensemble_waves += 1
        _evaluate_remote_messages_model_axis(
            grouped,
            responses,
            model_axis_evaluator,
            device,
            batch_size,
            profile,
        )
        profile.ipc_messages += len(messages)
        response_put_started = time.perf_counter()
        for worker_id, request_id, _jobs in messages:
            response_queues[worker_id].put(
                (request_id, responses[(worker_id, request_id)])
            )
        profile.response_put_seconds += time.perf_counter() - response_put_started
        return
    if model_axis_evaluator is not None:
        profile.cuda_per_model_waves += 1

    if device.type == "mps" and len(grouped) >= 2:
        profile.cuda_per_model_waves += 1
        _evaluate_remote_messages_mps_queued(
            grouped,
            responses,
            models,
            device,
            batch_size,
            profile,
        )
        profile.ipc_messages += len(messages)
        response_put_started = time.perf_counter()
        for worker_id, request_id, _jobs in messages:
            response_queues[worker_id].put(
                (request_id, responses[(worker_id, request_id)])
            )
        profile.response_put_seconds += time.perf_counter() - response_put_started
        return

    for model_key, entries in grouped.items():
        model = models.get(model_key)
        if model is None:
            raise RuntimeError(f"中央NN batcherにモデルがありません: {model_key}")

        # 同じbatchへ幅64と幅1を混ぜると全rowが64へpaddingされる。job数や
        # batch上限を変えず、近いdecoder幅が同じchunkへ入る順序にする。
        entries.sort(key=lambda entry: entry[3].decoder_words)
        central_chunks: list[list[tuple[int, int, int, _RemoteEvalJob]]] = []
        current: list[tuple[int, int, int, _RemoteEvalJob]] = []
        current_size = 0
        for entry in entries:
            job_size = entry[3].batch_count
            if current and current_size + job_size > batch_size:
                central_chunks.append(current)
                current = []
                current_size = 0
            current.append(entry)
            current_size += job_size
        if current:
            central_chunks.append(current)

        for central_chunk in central_chunks:
            merge_started = time.perf_counter()
            jobs = [entry[3] for entry in central_chunk]
            decoder_words = max(job.decoder_words for job in jobs)
            encoder = _merge_combined_sparse([job.encoder for job in jobs])
            decoder = _merge_combined_sparse(
                [_pad_remote_decoder(job, decoder_words) for job in jobs]
            )
            evaluation_count = sum(job.batch_count for job in jobs)
            profile.nn_decoder_source_tokens += sum(
                job.batch_count * job.decoder_words for job in jobs
            )
            profile.nn_decoder_padded_tokens += evaluation_count * decoder_words
            profile.nn_merge_seconds += time.perf_counter() - merge_started

            started = time.perf_counter()
            with torch.inference_mode():
                input_started = time.perf_counter()
                inputs = (
                    torch.from_numpy(encoder[0]).to(device),
                    torch.from_numpy(encoder[1]).to(device),
                    torch.from_numpy(encoder[2]).to(device),
                    torch.from_numpy(decoder[0]).to(device),
                    torch.from_numpy(decoder[1]).to(device),
                    torch.from_numpy(decoder[2]).to(device),
                )
                profile.nn_input_seconds += time.perf_counter() - input_started

                forward_started = time.perf_counter()
                values, policies = model(*inputs)
                profile.nn_forward_submit_seconds += (
                    time.perf_counter() - forward_started
                )

                output_started = time.perf_counter()
                value_rows_tensor = values.detach().cpu()
                policy_rows_tensor = policies.detach().cpu()
                profile.nn_output_wait_seconds += (
                    time.perf_counter() - output_started
                )

                tolist_started = time.perf_counter()
                value_rows = value_rows_tensor.tolist()
                policy_rows = policy_rows_tensor.tolist()
                profile.nn_tolist_seconds += time.perf_counter() - tolist_started
            profile.nn_seconds += time.perf_counter() - started
            profile.nn_evaluations += evaluation_count
            profile.nn_batches += 1
            profile.batch_sizes.append(evaluation_count)

            response_pack_started = time.perf_counter()
            row_offset = 0
            for worker_id, request_id, job_index, job in central_chunk:
                row_end = row_offset + job.batch_count
                responses[(worker_id, request_id)][job_index] = (
                    value_rows[row_offset:row_end],
                    policy_rows[row_offset:row_end],
                )
                row_offset = row_end
            profile.nn_response_pack_seconds += (
                time.perf_counter() - response_pack_started
            )

    profile.ipc_messages += len(messages)
    response_put_started = time.perf_counter()
    for worker_id, request_id, _jobs in messages:
        response_queues[worker_id].put(
            (request_id, responses[(worker_id, request_id)])
        )
    profile.response_put_seconds += time.perf_counter() - response_put_started


def _pad_empty_sparse_rows(
    vector: tuple[np.ndarray, np.ndarray, np.ndarray],
    current_rows: int,
    target_rows: int,
    words_per_row: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """model-axis batchの不足行をzero EmbeddingBagとして追加する。"""
    if current_rows > target_rows:
        raise ValueError("current_rowsがtarget_rowsを超えています。")
    indices, values, offsets = vector
    expected_offsets = current_rows * words_per_row
    if len(offsets) != expected_offsets:
        raise ValueError(
            f"sparse offset数が不正です: {len(offsets)} != {expected_offsets}"
        )
    missing_offsets = (target_rows - current_rows) * words_per_row
    if missing_offsets == 0:
        return vector
    return (
        indices,
        values,
        np.pad(
            offsets,
            (0, missing_offsets),
            constant_values=len(indices),
        ),
    )


def _evaluate_remote_messages_model_axis(
    grouped: dict[str, list[tuple[int, int, int, _RemoteEvalJob]]],
    responses: dict[tuple[int, int], list[Any]],
    evaluator: _CudaEnsembleEvaluator | _MpsModelAxisEvaluator,
    device: torch.device,
    batch_size: int,
    profile: BatchedProfile,
    *,
    model_keys: list[str] | None = None,
) -> None:
    """CPU worker群の小batchをmodel軸へ積み、1回で評価する。"""
    wave_model_keys = evaluator.model_keys if model_keys is None else model_keys
    chunked: dict[
        str,
        list[list[tuple[int, int, int, _RemoteEvalJob]]],
    ] = {}
    for model_key, entries in grouped.items():
        entries.sort(key=lambda entry: entry[3].decoder_words)
        chunks: list[list[tuple[int, int, int, _RemoteEvalJob]]] = []
        current: list[tuple[int, int, int, _RemoteEvalJob]] = []
        current_size = 0
        for entry in entries:
            entry_size = entry[3].batch_count
            if current and current_size + entry_size > batch_size:
                chunks.append(current)
                current = []
                current_size = 0
            current.append(entry)
            current_size += entry_size
        if current:
            chunks.append(current)
        chunked[model_key] = chunks

    wave_count = max(len(chunks) for chunks in chunked.values())
    for wave_index in range(wave_count):
        merge_started = time.perf_counter()
        active_chunks = {
            model_key: chunks[wave_index]
            for model_key, chunks in chunked.items()
            if wave_index < len(chunks)
        }
        model_counts = {
            model_key: sum(entry[3].batch_count for entry in entries)
            for model_key, entries in active_chunks.items()
        }
        model_batch = max(model_counts.values())
        required_decoder_words = max(
            entry[3].decoder_words
            for entries in active_chunks.values()
            for entry in entries
        )
        decoder_words = 1 << (required_decoder_words - 1).bit_length()
        profile.nn_decoder_source_tokens += sum(
            entry[3].batch_count * entry[3].decoder_words
            for entries in active_chunks.values()
            for entry in entries
        )
        profile.nn_decoder_padded_tokens += (
            len(wave_model_keys) * model_batch * decoder_words
        )

        combined_rows: list[
            tuple[
                tuple[np.ndarray, np.ndarray, np.ndarray],
                tuple[np.ndarray, np.ndarray, np.ndarray],
            ]
        ] = []
        for model_key in wave_model_keys:
            entries = active_chunks.get(model_key, [])
            row_count = model_counts.get(model_key, 0)
            encoder = _merge_combined_sparse(
                [entry[3].encoder for entry in entries]
            )
            decoder = _merge_combined_sparse(
                [_pad_remote_decoder(entry[3], decoder_words) for entry in entries]
            )
            encoder = _pad_empty_sparse_rows(
                encoder,
                row_count,
                model_batch,
                evaluator.num_words_encoder,
            )
            decoder = _pad_empty_sparse_rows(
                decoder,
                row_count,
                model_batch,
                decoder_words,
            )
            combined_rows.append((encoder, decoder))

        encoder_values = max(len(row[0][0]) for row in combined_rows)
        decoder_values = max(len(row[1][0]) for row in combined_rows)
        combined_rows = [
            (
                _pad_combined_sparse(encoder, encoder_values),
                _pad_combined_sparse(decoder, decoder_values),
            )
            for encoder, decoder in combined_rows
        ]
        profile.nn_merge_seconds += time.perf_counter() - merge_started

        started = time.perf_counter()
        input_started = time.perf_counter()
        stacked_encoder = tuple(
            np.stack([row[0][component] for row in combined_rows])
            for component in range(3)
        )
        stacked_decoder = tuple(
            np.stack([row[1][component] for row in combined_rows])
            for component in range(3)
        )
        if isinstance(evaluator, _MpsModelAxisEvaluator):
            inputs = _mps_sparse_inputs(
                stacked_encoder,
                stacked_decoder,
                device,
            )
        else:
            inputs = tuple(
                torch.from_numpy(array).to(device)
                for array in (*stacked_encoder, *stacked_decoder)
            )
        profile.nn_input_seconds += time.perf_counter() - input_started
        with torch.inference_mode():
            forward_started = time.perf_counter()
            if isinstance(evaluator, _MpsModelAxisEvaluator):
                values, policies = evaluator.evaluate(*inputs)
                combined_output = torch.cat((values, policies), dim=-1)
            else:
                values, policies = evaluator.evaluate(*inputs)
            profile.nn_forward_submit_seconds += time.perf_counter() - forward_started

            output_started = time.perf_counter()
            if isinstance(evaluator, _MpsModelAxisEvaluator):
                output_rows_tensor = combined_output.detach().cpu()
            else:
                value_rows_tensor = values.detach().cpu()
                policy_rows_tensor = policies.detach().cpu()
            profile.nn_output_wait_seconds += time.perf_counter() - output_started

            tolist_started = time.perf_counter()
            if isinstance(evaluator, _MpsModelAxisEvaluator):
                value_rows = output_rows_tensor[..., :1].tolist()
                policy_rows = output_rows_tensor[..., 1:].tolist()
            else:
                value_rows = value_rows_tensor.tolist()
                policy_rows = policy_rows_tensor.tolist()
            profile.nn_tolist_seconds += time.perf_counter() - tolist_started
        profile.nn_seconds += time.perf_counter() - started

        response_pack_started = time.perf_counter()
        output_model_indices = {
            model_key: index for index, model_key in enumerate(wave_model_keys)
        }
        for model_key, entries in active_chunks.items():
            model_index = output_model_indices[model_key]
            row_offset = 0
            for worker_id, request_id, job_index, job in entries:
                row_end = row_offset + job.batch_count
                responses[(worker_id, request_id)][job_index] = (
                    value_rows[model_index][row_offset:row_end],
                    policy_rows[model_index][row_offset:row_end],
                )
                row_offset = row_end
            profile.nn_evaluations += row_offset
            profile.nn_batches += 1
            profile.batch_sizes.append(row_offset)
        profile.nn_response_pack_seconds += (
            time.perf_counter() - response_pack_started
        )


def _collect_ready_messages(
    request_queue: Any,
    first_message: Any,
) -> list[Any]:
    """IPC queueへ到着済みの要求だけを待たずに回収する。"""
    messages = [first_message]
    while True:
        try:
            message = request_queue.get_nowait()
        except queue.Empty:
            break
        messages.append(message)
    return messages


def run_worker_batched_tournament(
    specs: list[Any],
    pairings: list[tuple[str, str]],
    num_games: int,
    *,
    games_per_pairing: list[int] | None = None,
    game_index_offset: int = 0,
    game_index_offsets: list[int] | None = None,
    alternate_sides: bool = True,
    device_name: str = "auto",
    batch_size: int = 256,
    lanes: int = 128,
    search_count: int = 10,
    max_selections: int = 2000,
    max_turns: int | None = None,
    seed: int = 0,
    cpu_workers: int = 2,
    model_axis_models: bool = True,
    training_json_dir: Path | None = None,
) -> BatchedTournamentOutput:
    """CPU libcg worker群と中央NN batcherで総当たりを実行する。"""
    if cpu_workers < 1:
        raise ValueError("cpu_workersは1以上で指定してください。")

    training_output_dir = (
        Path(training_json_dir).resolve()
        if training_json_dir is not None
        else None
    )
    if training_output_dir is not None:
        training_output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device_name)
    if games_per_pairing is None:
        games_per_pairing = [num_games] * len(pairings)
    if len(games_per_pairing) != len(pairings):
        raise ValueError("games_per_pairingはpairingsと同じ長さにしてください。")
    if any(count < 0 for count in games_per_pairing):
        raise ValueError("games_per_pairingの各試合数は0以上にしてください。")
    if game_index_offset < 0:
        raise ValueError("game_index_offsetは0以上にしてください。")
    if game_index_offsets is None:
        game_index_offsets = [game_index_offset] * len(pairings)
    if len(game_index_offsets) != len(pairings):
        raise ValueError("game_index_offsetsはpairingsと同じ長さにしてください。")
    if any(offset < 0 for offset in game_index_offsets):
        raise ValueError("game_index_offsetsの各値は0以上にしてください。")

    game_requests = [
        BatchedGameRequest(
            name0=name0,
            name1=name1,
            game_index=pairing_offset + game_index + 1,
            swap=(
                alternate_sides
                and (pairing_offset + game_index) % 2 == 1
            ),
        )
        for (name0, name1), pairing_games, pairing_offset in zip(
            pairings,
            games_per_pairing,
            game_index_offsets,
            strict=True,
        )
        for game_index in range(pairing_games)
    ]
    if not game_requests:
        return BatchedTournamentOutput([], BatchedProfile(), str(device))

    worker_count = min(cpu_workers, len(game_requests))
    request_chunks = [
        game_requests[index::worker_count] for index in range(worker_count)
    ]
    worker_lanes = max(1, math.ceil(lanes / worker_count))
    # worker jobを中央batchへ隙間なく詰められるよう、worker数で均等分割する。
    # 大きなjobは途中分割できず、実測で平均batchとgames/sを悪化させた。
    if device.type == "mps":
        # 30モデル条件では28件上限がjobを細分化しすぎるため、中央batchへ
        # worker数の約半分を中央batchへ同居させつつ、worker内jobを大きくする。
        remote_batch_size = max(1, batch_size // math.ceil(worker_count / 2))
    else:
        remote_batch_size = max(1, batch_size // worker_count)
    spec_records = [
        (
            str(spec.name),
            str(Path(spec.agent_path).resolve()),
            str(Path(spec.deck_path).resolve()),
        )
        for spec in specs
    ]

    context = multiprocessing.get_context("spawn")
    request_queue = context.Queue(maxsize=worker_count * 2)
    result_queue = context.Queue()
    response_queues = [context.Queue(maxsize=2) for _ in range(worker_count)]
    from batched_worker_bootstrap import worker_main

    request_records = [
        [
            (request.name0, request.name1, request.game_index, request.swap)
            for request in chunk
        ]
        for chunk in request_chunks
    ]
    processes = [
        context.Process(
            target=worker_main,
            args=(
                worker_id,
                spec_records,
                request_records[worker_id],
                worker_lanes,
                search_count,
                max_selections,
                max_turns,
                seed,
                remote_batch_size,
                str(training_output_dir) if training_output_dir is not None else None,
                request_queue,
                response_queues[worker_id],
                result_queue,
            ),
            name=f"libcg-worker-{worker_id}",
        )
        for worker_id in range(worker_count)
    ]
    for process in processes:
        process.start()

    canonical_spec = next(
        (
            spec
            for spec in specs
            if (Path(spec.agent_path).resolve().parent / "model.pth").exists()
        ),
        None,
    )
    if canonical_spec is None:
        for process in processes:
            process.terminate()
        raise ValueError("worker-batched backendにはrl_mcts agentが必要です。")
    runtime = _load_runtime(Path(canonical_spec.agent_path).resolve().parent)
    participants = _load_participants(specs, runtime, device)
    models = {
        model_key: model
        for participant in participants.values()
        for model_key, model in (
            (participant.model_key, participant.model),
            (participant.opponent_model_key, participant.opponent_model),
        )
        if model_key is not None and model is not None
    }
    model_axis_evaluator: _CudaEnsembleEvaluator | _MpsModelAxisEvaluator | None
    if model_axis_models and device.type == "cuda":
        model_axis_evaluator = _CudaEnsembleEvaluator(participants.values(), device)
    elif model_axis_models and device.type == "mps":
        # wave単位の一括H2D後は、矩形paddingを伴うmanual model-axisより
        # 通常model forwardをqueueへ連続投入する方がMPSで速い。
        model_axis_evaluator = None
    else:
        model_axis_evaluator = None
    if isinstance(model_axis_evaluator, _CudaEnsembleEvaluator):
        torch.cuda.synchronize(device)

    profile = BatchedProfile(cpu_workers=worker_count)
    worker_outputs: dict[int, BatchedTournamentOutput] = {}
    failure: str | None = None
    try:
        while len(worker_outputs) < worker_count and failure is None:
            while True:
                try:
                    status, worker_id, payload = result_queue.get_nowait()
                except queue.Empty:
                    break
                if status == "error":
                    failure = f"libcg worker {worker_id} failed:\n{payload}"
                    break
                worker_outputs[worker_id] = payload
            if failure is not None or len(worker_outputs) == worker_count:
                break

            try:
                first_message = request_queue.get(timeout=0.01)
            except queue.Empty:
                crashed = [
                    process
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if crashed:
                    failure = ", ".join(
                        f"{process.name}: exitcode={process.exitcode}"
                        for process in crashed
                    )
                continue

            collect_started = time.perf_counter()
            messages = _collect_ready_messages(
                request_queue,
                first_message,
            )
            profile.batch_collect_seconds += time.perf_counter() - collect_started
            _evaluate_remote_messages(
                messages,
                models,
                device,
                batch_size,
                profile,
                response_queues,
                model_axis_evaluator,
            )
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        if failure is not None:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=5)

    if failure is not None:
        raise RuntimeError(failure)
    if len(worker_outputs) != worker_count:
        raise RuntimeError(
            f"worker結果が不足しています: {len(worker_outputs)}/{worker_count}"
        )

    results: list[BatchedGameResult] = []
    for worker_id in range(worker_count):
        output = worker_outputs[worker_id]
        results.extend(output.results)
        worker_profile = output.profile
        for field_name in (
            "battle_start_seconds",
            "battle_step_seconds",
            "search_begin_seconds",
            "search_step_seconds",
            "search_step_c_api_seconds",
            "search_step_json_seconds",
            "search_step_dataclass_seconds",
            "feature_seconds",
            "search_finalize_seconds",
            "battle_steps",
            "search_steps",
            "remote_wait_seconds",
            "remote_numpy_pack_seconds",
        ):
            setattr(
                profile,
                field_name,
                getattr(profile, field_name) + getattr(worker_profile, field_name),
            )

    results.sort(key=lambda result: (result.name0, result.name1, result.game_index))
    return BatchedTournamentOutput(results, profile, str(device))
