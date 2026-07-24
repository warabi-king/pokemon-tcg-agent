"""MCTS探索木をdevice Tensorへ常駐させる総当たりbackend。

``batched_tournament`` はNN forward後のvalue/policyをCPUへ戻し、Pythonの
Node/Childへ反映していた。このbackendはvisit/total/value/prior/childをdevice
Tensorとして保持し、UCB選択とbackpropもPyTorch演算で行う。CPUへ戻すのは
libcg ``SearchStep`` に必要なcontext/node/action indexだけである。

盤面の正本とSearchStateはCPU版libcgが所有するため、状態遷移と特徴量生成は
引き続きCPUで行う。GPUへ渡す疎入力は、6 Tensorを個別転送せず、整数・浮動
小数の2本へpackしてからdevice上でviewへ分解する。
"""

from __future__ import annotations

from collections import defaultdict, deque
import ctypes
from dataclasses import dataclass
from pathlib import Path
import random
import time
from typing import Any

import torch

try:
    from batched_tournament import (
        MAX_ACTIONS,
        BatchedGameRequest,
        BatchedGameResult,
        BatchedProfile,
        BatchedTournamentOutput,
        _MatchSession,
        _Participant,
        _Runtime,
        _battle_observation,
        _combine_sparse,
        _enumerate_actions,
        _load_participants,
        _load_runtime,
        _pad_sparse_offsets,
        _random_action,
        _raw_result,
        _start_session,
        _to_result,
        select_device,
    )
except ModuleNotFoundError:  # ``python -m unittest tools...``用
    from .batched_tournament import (
        MAX_ACTIONS,
        BatchedGameRequest,
        BatchedGameResult,
        BatchedProfile,
        BatchedTournamentOutput,
        _MatchSession,
        _Participant,
        _Runtime,
        _battle_observation,
        _combine_sparse,
        _enumerate_actions,
        _load_participants,
        _load_runtime,
        _pad_sparse_offsets,
        _random_action,
        _raw_result,
        _start_session,
        _to_result,
        select_device,
    )


@dataclass
class GpuTreeProfile(BatchedProfile):
    gpu_tree_seconds: float = 0.0
    gpu_to_cpu_seconds: float = 0.0
    packed_input_bytes: int = 0


@dataclass
class _GpuContext:
    session: _MatchSession
    participant: _Participant
    your_index: int


@dataclass
class _PendingEvaluation:
    context_index: int
    node_index: int
    encoder: Any
    decoder: Any


_MPS_KERNELS: Any | None = None


def _get_mps_kernels() -> Any:
    """UCB traversal・NN結果反映・backpropを融合したMetal kernelを返す。"""
    global _MPS_KERNELS
    if _MPS_KERNELS is not None:
        return _MPS_KERNELS
    source = r"""
#include <metal_stdlib>
using namespace metal;

kernel void reset_tree(
    device long* launch [[buffer(0)]],
    device long* parent [[buffer(1)]],
    device long* current_player [[buffer(2)]],
    device long* result [[buffer(3)]],
    device long* action_count [[buffer(4)]],
    device long* child [[buffer(5)]],
    device float* visit [[buffer(6)]],
    device float* total [[buffer(7)]],
    device float* node_value [[buffer(8)]],
    device float* prior [[buffer(9)]],
    device long* root_player [[buffer(10)]],
    device const long* new_root_player [[buffer(11)]],
    constant long& context_count [[buffer(12)]],
    constant long& max_nodes [[buffer(13)]],
    constant long& max_actions [[buffer(14)]],
    uint context [[thread_position_in_grid]]) {
  if (context >= context_count) return;
  launch[context] = 0;
  root_player[context] = new_root_player[context];
  for (long node = 0; node < max_nodes; ++node) {
    long node_offset = ((long)context) * max_nodes + node;
    parent[node_offset] = -1;
    current_player[node_offset] = -1;
    result[node_offset] = -1;
    action_count[node_offset] = 0;
    visit[node_offset] = 0.0f;
    total[node_offset] = 0.0f;
    node_value[node_offset] = 0.0f;
    for (long action = 0; action < max_actions; ++action) {
      long edge_offset = node_offset * max_actions + action;
      child[edge_offset] = -1;
      prior[edge_offset] = 0.0f;
    }
  }
}

kernel void install_metadata(
    device long* launch [[buffer(0)]],
    device long* parent [[buffer(1)]],
    device long* current_player [[buffer(2)]],
    device long* result [[buffer(3)]],
    device long* action_count [[buffer(4)]],
    device const long* metadata [[buffer(5)]],
    constant long& item_count [[buffer(6)]],
    constant long& max_nodes [[buffer(7)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= item_count) return;
  launch[idx] = 0;
  long context = metadata[idx * 6];
  long node = metadata[idx * 6 + 1];
  long offset = context * max_nodes + node;
  parent[offset] = metadata[idx * 6 + 2];
  current_player[offset] = metadata[idx * 6 + 3];
  result[offset] = metadata[idx * 6 + 4];
  action_count[offset] = metadata[idx * 6 + 5];
}

kernel void link_children(
    device long* launch [[buffer(0)]],
    device long* child [[buffer(1)]],
    device const long* metadata [[buffer(2)]],
    constant long& item_count [[buffer(3)]],
    constant long& max_nodes [[buffer(4)]],
    constant long& max_actions [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= item_count) return;
  launch[idx] = 0;
  long context = metadata[idx * 4];
  long parent = metadata[idx * 4 + 1];
  long edge = metadata[idx * 4 + 2];
  long node = metadata[idx * 4 + 3];
  child[(context * max_nodes + parent) * max_actions + edge] = node;
}

kernel void finish_actions(
    device long* best_edge [[buffer(0)]],
    device const long* child [[buffer(1)]],
    device const float* visit [[buffer(2)]],
    device const float* prior [[buffer(3)]],
    constant long& context_count [[buffer(4)]],
    constant long& max_nodes [[buffer(5)]],
    constant long& max_actions [[buffer(6)]],
    uint context [[thread_position_in_grid]]) {
  if (context >= context_count) return;
  long root_offset = ((long)context) * max_nodes;
  long selected = 0;
  float best_visit = -1.0f;
  bool has_expanded = false;
  for (long action = 0; action < max_actions; ++action) {
    long candidate = child[root_offset * max_actions + action];
    if (candidate >= 0) {
      has_expanded = true;
      float candidate_visit = visit[((long)context) * max_nodes + candidate];
      if (candidate_visit > best_visit) {
        best_visit = candidate_visit;
        selected = action;
      }
    }
  }
  if (!has_expanded) {
    float best_prior = -INFINITY;
    for (long action = 0; action < max_actions; ++action) {
      float candidate_prior = prior[root_offset * max_actions + action];
      if (candidate_prior > best_prior) {
        best_prior = candidate_prior;
        selected = action;
      }
    }
  }
  best_edge[context] = selected;
}

kernel void backprop_values(
    device long* launch [[buffer(0)]],
    device float* total [[buffer(1)]],
    device float* visit [[buffer(2)]],
    device float* node_value [[buffer(3)]],
    device const long* parent [[buffer(4)]],
    device const long* contexts [[buffer(5)]],
    device const long* nodes [[buffer(6)]],
    device const float* values [[buffer(7)]],
    constant long& item_count [[buffer(8)]],
    constant long& max_nodes [[buffer(9)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= item_count) return;
  launch[idx] = 0;
  long context = contexts[idx];
  long node = nodes[idx];
  float value = values[idx];
  node_value[context * max_nodes + node] = value;
  for (long depth = 0; depth < max_nodes && node >= 0; ++depth) {
    long offset = context * max_nodes + node;
    total[offset] += value;
    visit[offset] += 1.0f;
    node = parent[offset];
  }
}

kernel void apply_evaluation(
    device long* launch [[buffer(0)]],
    device float* total [[buffer(1)]],
    device float* visit [[buffer(2)]],
    device float* node_value [[buffer(3)]],
    device float* prior [[buffer(4)]],
    device const long* parent [[buffer(5)]],
    device const long* current_player [[buffer(6)]],
    device const long* root_player [[buffer(7)]],
    device const long* action_count [[buffer(8)]],
    device const long* contexts [[buffer(9)]],
    device const long* nodes [[buffer(10)]],
    device const float* values [[buffer(11)]],
    device const float* policies [[buffer(12)]],
    constant long& item_count [[buffer(13)]],
    constant long& max_nodes [[buffer(14)]],
    constant long& max_actions [[buffer(15)]],
    constant long& policy_width [[buffer(16)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx >= item_count) return;
  launch[idx] = 0;
  long context = contexts[idx];
  long node = nodes[idx];
  long node_offset = context * max_nodes + node;
  float value = values[idx];
  if (current_player[node_offset] != root_player[context]) value = -value;
  node_value[node_offset] = value;

  long count = min(action_count[node_offset], policy_width);
  float maximum = -INFINITY;
  for (long action = 0; action < count; ++action) {
    maximum = max(maximum, policies[idx * policy_width + action] * 10.0f);
  }
  float sum = 0.0f;
  for (long action = 0; action < count; ++action) {
    float probability = exp(policies[idx * policy_width + action] * 10.0f - maximum);
    prior[(node_offset * max_actions) + action] = probability;
    sum += probability;
  }
  if (sum > 0.0f) {
    for (long action = 0; action < count; ++action) {
      prior[(node_offset * max_actions) + action] /= sum;
    }
  }

  long current = node;
  for (long depth = 0; depth < max_nodes && current >= 0; ++depth) {
    long offset = context * max_nodes + current;
    total[offset] += value;
    visit[offset] += 1.0f;
    current = parent[offset];
  }
}

kernel void select_expansions(
    device long* expansion_parent [[buffer(0)]],
    device long* expansion_edge [[buffer(1)]],
    device float* total [[buffer(2)]],
    device float* visit [[buffer(3)]],
    device const float* node_value [[buffer(4)]],
    device const float* prior [[buffer(5)]],
    device const long* parent [[buffer(6)]],
    device const long* current_player [[buffer(7)]],
    device const long* result [[buffer(8)]],
    device const long* action_count [[buffer(9)]],
    device const long* child [[buffer(10)]],
    device const long* root_player [[buffer(11)]],
    constant long& context_count [[buffer(12)]],
    constant long& max_nodes [[buffer(13)]],
    constant long& max_actions [[buffer(14)]],
    uint context [[thread_position_in_grid]]) {
  if (context >= context_count) return;
  expansion_parent[context] = -1;
  expansion_edge[context] = -1;
  long current = 0;
  for (long depth = 0; depth < max_nodes && current >= 0; ++depth) {
    long node_offset = ((long)context) * max_nodes + current;
    long count = action_count[node_offset];
    if (count <= 0) return;
    float parent_visit = max(visit[node_offset], 1.0f);
    float parent_average = total[node_offset] / parent_visit;
    float exploration = 0.4f * sqrt(parent_visit);
    bool flip = current_player[node_offset] != root_player[context];
    float best_score = -INFINITY;
    long best_edge = -1;
    long best_child = -1;
    for (long action = 0; action < count; ++action) {
      long edge_offset = node_offset * max_actions + action;
      long candidate = child[edge_offset];
      float child_visit = 0.0f;
      float q = parent_average;
      if (candidate >= 0) {
        long child_offset = ((long)context) * max_nodes + candidate;
        child_visit = visit[child_offset];
        q = total[child_offset] / max(child_visit, 1.0f);
      }
      if (flip) q = -q;
      float score = q + exploration * prior[edge_offset] / (1.0f + child_visit);
      if (score > best_score) {
        best_score = score;
        best_edge = action;
        best_child = candidate;
      }
    }
    if (best_edge < 0) return;
    if (best_child < 0) {
      expansion_parent[context] = current;
      expansion_edge[context] = best_edge;
      return;
    }
    long child_offset = ((long)context) * max_nodes + best_child;
    if (result[child_offset] >= 0) {
      float terminal_value = node_value[child_offset];
      long backprop_node = best_child;
      for (long back_depth = 0;
           back_depth < max_nodes && backprop_node >= 0;
           ++back_depth) {
        long offset = ((long)context) * max_nodes + backprop_node;
        total[offset] += terminal_value;
        visit[offset] += 1.0f;
        backprop_node = parent[offset];
      }
      return;
    }
    current = best_child;
  }
}

kernel void select_parallel_expansions(
    device long* launch [[buffer(0)]],
    device long* expansion_parent [[buffer(1)]],
    device long* expansion_edge [[buffer(2)]],
    device long* advanced_count [[buffer(3)]],
    device float* total [[buffer(4)]],
    device float* visit [[buffer(5)]],
    device const float* node_value [[buffer(6)]],
    device const float* prior [[buffer(7)]],
    device const long* parent [[buffer(8)]],
    device const long* current_player [[buffer(9)]],
    device const long* result [[buffer(10)]],
    device const long* action_count [[buffer(11)]],
    device const long* child [[buffer(12)]],
    device const long* root_player [[buffer(13)]],
    device const long* remaining [[buffer(14)]],
    constant long& context_count [[buffer(15)]],
    constant long& max_nodes [[buffer(16)]],
    constant long& max_actions [[buffer(17)]],
    constant long& max_requests [[buffer(18)]],
    uint context [[thread_position_in_grid]]) {
  if (context >= context_count) return;
  launch[context] = 0;
  bool reserved_root[64];
  for (long action = 0; action < 64; ++action) reserved_root[action] = false;
  for (long request = 0; request < max_requests; ++request) {
    long output_offset = ((long)context) * max_requests + request;
    expansion_parent[output_offset] = -1;
    expansion_edge[output_offset] = -1;
  }
  long advanced = 0;
  long requested = min(remaining[context], max_requests);
  for (long request = 0; request < requested; ++request) {
    long current = 0;
    long root_edge = -1;
    bool completed = false;
    for (long depth = 0; depth < max_nodes && current >= 0; ++depth) {
      long node_offset = ((long)context) * max_nodes + current;
      long count = action_count[node_offset];
      if (count <= 0) break;
      float parent_visit = max(visit[node_offset], 1.0f);
      float parent_average = total[node_offset] / parent_visit;
      float exploration = 0.4f * sqrt(parent_visit);
      bool flip = current_player[node_offset] != root_player[context];
      float best_score = -INFINITY;
      long best_edge = -1;
      long best_child = -1;
      for (long action = 0; action < count; ++action) {
        if (current == 0 && reserved_root[action]) continue;
        long edge_offset = node_offset * max_actions + action;
        long candidate = child[edge_offset];
        float child_visit = 0.0f;
        float q = parent_average;
        if (candidate >= 0) {
          long child_offset = ((long)context) * max_nodes + candidate;
          child_visit = visit[child_offset];
          q = total[child_offset] / max(child_visit, 1.0f);
        }
        if (flip) q = -q;
        float score = q + exploration * prior[edge_offset] / (1.0f + child_visit);
        if (score > best_score) {
          best_score = score;
          best_edge = action;
          best_child = candidate;
        }
      }
      if (best_edge < 0) break;
      if (current == 0) root_edge = best_edge;
      if (best_child < 0) {
        long output_offset = ((long)context) * max_requests + request;
        expansion_parent[output_offset] = current;
        expansion_edge[output_offset] = best_edge;
        if (root_edge >= 0) reserved_root[root_edge] = true;
        advanced += 1;
        completed = true;
        break;
      }
      long child_offset = ((long)context) * max_nodes + best_child;
      if (result[child_offset] >= 0) {
        float terminal_value = node_value[child_offset];
        long backprop_node = best_child;
        for (long back_depth = 0;
             back_depth < max_nodes && backprop_node >= 0;
             ++back_depth) {
          long offset = ((long)context) * max_nodes + backprop_node;
          total[offset] += terminal_value;
          visit[offset] += 1.0f;
          backprop_node = parent[offset];
        }
        advanced += 1;
        completed = true;
        break;
      }
      current = best_child;
    }
    if (!completed) break;
  }
  advanced_count[context] = advanced;
}
"""
    _MPS_KERNELS = torch.mps.compile_shader(source)
    return _MPS_KERNELS


class _GpuMctsGroup:
    """同じモデルを使う複数局面の探索木を1つのdevice Tensorで保持する。"""

    def __init__(
        self,
        contexts: list[_GpuContext],
        device: torch.device,
        max_nodes: int,
        capacity: int | None = None,
    ) -> None:
        self.device = device
        self.capacity = capacity or len(contexts)
        if len(contexts) > self.capacity:
            raise ValueError("GPU MCTS workspace容量が不足しています。")
        self.contexts = contexts
        self.context_count = len(contexts)
        self.max_nodes = max_nodes
        shape = (self.capacity, max_nodes)

        self.parent = torch.empty(shape, dtype=torch.int64, device=device)
        self.current_player = torch.empty(shape, dtype=torch.int64, device=device)
        self.result = torch.empty(shape, dtype=torch.int64, device=device)
        self.action_count = torch.empty(shape, dtype=torch.int64, device=device)
        self.visit = torch.empty(shape, dtype=torch.float32, device=device)
        self.total = torch.empty(shape, dtype=torch.float32, device=device)
        self.value = torch.empty(shape, dtype=torch.float32, device=device)
        self.child = torch.empty(
            (self.capacity, max_nodes, MAX_ACTIONS),
            dtype=torch.int64,
            device=device,
        )
        self.prior = torch.empty(
            (self.capacity, max_nodes, MAX_ACTIONS),
            dtype=torch.float32,
            device=device,
        )
        self.root_player = torch.empty(
            self.capacity, dtype=torch.int64, device=device
        )
        self.full_context_range = torch.arange(
            self.capacity, dtype=torch.int64, device=device
        )
        self.context_range = self.full_context_range[: self.context_count]
        self.action_range = torch.arange(
            MAX_ACTIONS, dtype=torch.int64, device=device
        )

        # libcgのopaque SearchStateと可変長actionだけはCPUに残す。
        self.states: list[list[Any | None]] = [
            [None] * max_nodes for _ in range(self.capacity)
        ]
        self.actions: list[list[list[list[int]] | None]] = [
            [None] * max_nodes for _ in range(self.capacity)
        ]
        self.next_node = [1] * self.capacity
        self.mps_kernels = _get_mps_kernels() if device.type == "mps" else None
        self.reset(contexts)

    def reset(self, contexts: list[_GpuContext]) -> None:
        """確保済みworkspaceを別の手番集合へ再利用する。"""
        if len(contexts) > self.capacity:
            raise ValueError("GPU MCTS workspace容量が不足しています。")
        self.contexts = contexts
        self.context_count = len(contexts)
        self.context_range = self.full_context_range[: self.context_count]
        for context_index in range(self.context_count):
            self.states[context_index] = [None] * self.max_nodes
            self.actions[context_index] = [None] * self.max_nodes
            self.next_node[context_index] = 1

        root_players = torch.tensor(
            [context.your_index for context in contexts],
            dtype=torch.int64,
            device=self.device,
        )
        if self.mps_kernels is not None:
            launch = torch.empty(
                self.context_count, dtype=torch.int64, device=self.device
            )
            self.mps_kernels.reset_tree(
                launch,
                self.parent,
                self.current_player,
                self.result,
                self.action_count,
                self.child,
                self.visit,
                self.total,
                self.value,
                self.prior,
                self.root_player,
                root_players,
                self.context_count,
                self.max_nodes,
                MAX_ACTIONS,
            )
        else:
            self.parent[: self.context_count].fill_(-1)
            self.current_player[: self.context_count].fill_(-1)
            self.result[: self.context_count].fill_(-1)
            self.action_count[: self.context_count].zero_()
            self.child[: self.context_count].fill_(-1)
            self.visit[: self.context_count].zero_()
            self.total[: self.context_count].zero_()
            self.value[: self.context_count].zero_()
            self.prior[: self.context_count].zero_()
            self.root_player[: self.context_count].copy_(root_players)

    def _backprop(
        self,
        context_indices: torch.Tensor,
        node_indices: torch.Tensor,
        values: torch.Tensor,
        enabled: torch.Tensor | None = None,
    ) -> None:
        """親鎖を固定回数たどり、同期なしでdevice上だけで加算する。"""
        if self.mps_kernels is not None and enabled is None:
            launch = torch.empty_like(node_indices)
            self.mps_kernels.backprop_values(
                launch,
                self.total,
                self.visit,
                self.value,
                self.parent,
                context_indices,
                node_indices,
                values.contiguous(),
                node_indices.numel(),
                self.max_nodes,
            )
            return
        if enabled is None:
            enabled = torch.ones_like(node_indices, dtype=torch.bool)
        current = node_indices
        original_enabled = enabled
        for _ in range(self.max_nodes):
            active = original_enabled & (current >= 0)
            safe = current.clamp(min=0)
            old_total = self.total[context_indices, safe]
            old_visit = self.visit[context_indices, safe]
            increment = active.to(torch.float32)
            self.total[context_indices, safe] = old_total + values * increment
            self.visit[context_indices, safe] = old_visit + increment
            next_parent = self.parent[context_indices, safe]
            current = torch.where(active, next_parent, -1)

    def install_nodes(
        self,
        runtime: _Runtime,
        records: list[tuple[int, int, int, Any]],
        profile: GpuTreeProfile,
    ) -> list[_PendingEvaluation]:
        """libcg SearchStateを木へ登録し、NNが必要なノードだけ返す。"""
        if not records:
            return []

        contexts: list[int] = []
        nodes: list[int] = []
        parents: list[int] = []
        current_players: list[int] = []
        results: list[int] = []
        action_counts: list[int] = []
        pending: list[_PendingEvaluation] = []
        resolved_contexts: list[int] = []
        resolved_nodes: list[int] = []
        resolved_values: list[float] = []

        for context_index, node_index, parent_index, search_state in records:
            observation = search_state.observation
            state = observation.current
            self.states[context_index][node_index] = search_state

            contexts.append(context_index)
            nodes.append(node_index)
            parents.append(parent_index)
            current_players.append(int(state.yourIndex))
            results.append(int(state.result))

            if state.result >= 0:
                actions: list[list[int]] = []
                if state.result == 2:
                    resolved_value = 0.0
                elif state.result == self.contexts[context_index].your_index:
                    resolved_value = 1.0
                else:
                    resolved_value = -1.0
                resolved_contexts.append(context_index)
                resolved_nodes.append(node_index)
                resolved_values.append(resolved_value)
            else:
                actions = _enumerate_actions(
                    len(observation.select.option),
                    observation.select.maxCount,
                )
                if not actions:
                    resolved_contexts.append(context_index)
                    resolved_nodes.append(node_index)
                    resolved_values.append(0.0)
                else:
                    started = time.perf_counter()
                    encoder = runtime.get_encoder_input(
                        observation, self.contexts[context_index].participant.deck
                    )
                    decoder = runtime.get_decoder_input(observation, actions)
                    profile.feature_seconds += time.perf_counter() - started
                    pending.append(
                        _PendingEvaluation(
                            context_index,
                            node_index,
                            encoder,
                            decoder,
                        )
                    )
            self.actions[context_index][node_index] = actions
            action_counts.append(len(actions))

        if self.mps_kernels is not None:
            metadata_values: list[int] = []
            for values in zip(
                contexts,
                nodes,
                parents,
                current_players,
                results,
                action_counts,
                strict=True,
            ):
                metadata_values.extend(values)
            metadata = torch.tensor(
                metadata_values, dtype=torch.int64, device=self.device
            )
            launch = torch.empty(
                len(records), dtype=torch.int64, device=self.device
            )
            self.mps_kernels.install_metadata(
                launch,
                self.parent,
                self.current_player,
                self.result,
                self.action_count,
                metadata,
                len(records),
                self.max_nodes,
            )
        else:
            context_tensor = torch.tensor(
                contexts, dtype=torch.int64, device=self.device
            )
            node_tensor = torch.tensor(
                nodes, dtype=torch.int64, device=self.device
            )
            self.parent[context_tensor, node_tensor] = torch.tensor(
                parents, dtype=torch.int64, device=self.device
            )
            self.current_player[context_tensor, node_tensor] = torch.tensor(
                current_players, dtype=torch.int64, device=self.device
            )
            self.result[context_tensor, node_tensor] = torch.tensor(
                results, dtype=torch.int64, device=self.device
            )
            self.action_count[context_tensor, node_tensor] = torch.tensor(
                action_counts, dtype=torch.int64, device=self.device
            )

        if resolved_contexts:
            packed_resolved = torch.tensor(
                resolved_contexts + resolved_nodes,
                dtype=torch.int64,
                device=self.device,
            )
            resolved_count = len(resolved_contexts)
            resolved_context_tensor = packed_resolved[:resolved_count]
            resolved_node_tensor = packed_resolved[resolved_count:]
            resolved_value_tensor = torch.tensor(
                resolved_values, dtype=torch.float32, device=self.device
            )
            if self.mps_kernels is None:
                self.value[resolved_context_tensor, resolved_node_tensor] = (
                    resolved_value_tensor
                )
            self._backprop(
                resolved_context_tensor,
                resolved_node_tensor,
                resolved_value_tensor,
            )
        return pending

    def apply_evaluation(
        self,
        pending: list[_PendingEvaluation],
        values: torch.Tensor,
        policies: torch.Tensor,
    ) -> None:
        item_count = len(pending)
        packed_indices = torch.tensor(
            [item.context_index for item in pending]
            + [item.node_index for item in pending],
            dtype=torch.int64,
            device=self.device,
        )
        context_indices = packed_indices[:item_count]
        node_indices = packed_indices[item_count:]
        if self.mps_kernels is not None:
            launch = torch.empty_like(node_indices)
            self.mps_kernels.apply_evaluation(
                launch,
                self.total,
                self.visit,
                self.value,
                self.prior,
                self.parent,
                self.current_player,
                self.root_player,
                self.action_count,
                context_indices,
                node_indices,
                values.contiguous(),
                policies.contiguous(),
                node_indices.numel(),
                self.max_nodes,
                MAX_ACTIONS,
                policies.shape[1],
            )
            return
        node_players = self.current_player[context_indices, node_indices]
        root_players = self.root_player[context_indices]
        propagated = values[:, 0]
        propagated = torch.where(
            node_players == root_players, propagated, -propagated
        )
        self.value[context_indices, node_indices] = propagated

        policy_width = policies.shape[1]
        counts = self.action_count[context_indices, node_indices]
        valid = (
            torch.arange(policy_width, device=self.device)[None, :]
            < counts[:, None]
        )
        logits = (policies * 10.0).masked_fill(~valid, -torch.inf)
        probabilities = torch.softmax(logits, dim=1)
        self.prior[context_indices, node_indices, :policy_width] = probabilities
        self._backprop(context_indices, node_indices, propagated)

    def select_expansions(
        self,
        profile: GpuTreeProfile,
    ) -> list[tuple[int, int, int]]:
        """全contextを同時にUCB traversalし、CPU遷移が必要なedgeだけ返す。"""
        started = time.perf_counter()
        if self.mps_kernels is not None:
            expansion_parent = torch.empty(
                self.context_count, dtype=torch.int64, device=self.device
            )
            expansion_edge = torch.empty_like(expansion_parent)
            self.mps_kernels.select_expansions(
                expansion_parent,
                expansion_edge,
                self.total,
                self.visit,
                self.value,
                self.prior,
                self.parent,
                self.current_player,
                self.result,
                self.action_count,
                self.child,
                self.root_player,
                self.context_count,
                self.max_nodes,
                MAX_ACTIONS,
            )
            packed = torch.stack(
                (self.context_range, expansion_parent, expansion_edge), dim=1
            )
            packed = packed[expansion_parent >= 0]
            transfer_started = time.perf_counter()
            rows = packed.detach().cpu().tolist()
            profile.gpu_to_cpu_seconds += time.perf_counter() - transfer_started
            profile.gpu_tree_seconds += time.perf_counter() - started
            return [
                (int(c), int(parent), int(edge))
                for c, parent, edge in rows
            ]
        current = torch.zeros(
            self.context_count, dtype=torch.int64, device=self.device
        )
        active = torch.ones(
            self.context_count, dtype=torch.bool, device=self.device
        )
        expansion_parent = torch.full_like(current, -1)
        expansion_edge = torch.full_like(current, -1)

        for _ in range(self.max_nodes):
            safe_current = current.clamp(min=0)
            counts = self.action_count[self.context_range, safe_current]
            can_choose = active & (counts > 0)
            children = self.child[self.context_range, safe_current]
            safe_children = children.clamp(min=0)
            child_total = self.total[
                self.context_range[:, None], safe_children
            ]
            child_visit = self.visit[
                self.context_range[:, None], safe_children
            ]
            parent_total = self.total[self.context_range, safe_current]
            parent_visit = self.visit[self.context_range, safe_current]
            parent_average = parent_total / parent_visit.clamp(min=1.0)
            child_average = child_total / child_visit.clamp(min=1.0)
            q_value = torch.where(
                children < 0,
                parent_average[:, None],
                child_average,
            )
            flip = (
                self.current_player[self.context_range, safe_current]
                != self.root_player[: self.context_count]
            )
            q_value = torch.where(flip[:, None], -q_value, q_value)
            exploration = 0.4 * torch.sqrt(parent_visit.clamp(min=1.0))
            ucb = q_value + (
                exploration[:, None]
                * self.prior[self.context_range, safe_current]
                / (1.0 + child_visit)
            )
            valid_action = self.action_range[None, :] < counts[:, None]
            ucb = ucb.masked_fill(~valid_action, -torch.inf)
            best_edge = torch.argmax(ucb, dim=1)
            best_child = children[self.context_range, best_edge]

            should_expand = can_choose & (best_child < 0)
            expansion_parent = torch.where(
                should_expand, safe_current, expansion_parent
            )
            expansion_edge = torch.where(
                should_expand, best_edge, expansion_edge
            )

            existing = can_choose & (best_child >= 0)
            safe_best_child = best_child.clamp(min=0)
            terminal = existing & (
                self.result[self.context_range, safe_best_child] >= 0
            )
            terminal_values = self.value[
                self.context_range, safe_best_child
            ]
            self._backprop(
                self.context_range,
                safe_best_child,
                terminal_values,
                enabled=terminal,
            )

            continue_search = existing & ~terminal
            current = torch.where(continue_search, best_child, -1)
            active = continue_search

        packed = torch.stack(
            (self.context_range, expansion_parent, expansion_edge), dim=1
        )
        packed = packed[expansion_parent >= 0]
        transfer_started = time.perf_counter()
        rows = packed.detach().cpu().tolist()
        profile.gpu_to_cpu_seconds += time.perf_counter() - transfer_started
        profile.gpu_tree_seconds += time.perf_counter() - started
        return [(int(c), int(parent), int(edge)) for c, parent, edge in rows]

    def select_parallel_expansions(
        self,
        remaining: list[int],
        max_requests: int,
        profile: GpuTreeProfile,
    ) -> tuple[list[tuple[int, int, int]], list[int]]:
        """1 contextから複数の異なるroot branchを1 kernelで選ぶ。"""
        if self.mps_kernels is None:
            raise RuntimeError("parallel Metal selectionはMPS専用です。")
        started = time.perf_counter()
        launch = torch.empty(
            self.context_count, dtype=torch.int64, device=self.device
        )
        expansion_parent = torch.empty(
            (self.context_count, max_requests),
            dtype=torch.int64,
            device=self.device,
        )
        expansion_edge = torch.empty_like(expansion_parent)
        advanced_count = torch.empty(
            self.context_count, dtype=torch.int64, device=self.device
        )
        remaining_tensor = torch.tensor(
            remaining, dtype=torch.int64, device=self.device
        )
        self.mps_kernels.select_parallel_expansions(
            launch,
            expansion_parent,
            expansion_edge,
            advanced_count,
            self.total,
            self.visit,
            self.value,
            self.prior,
            self.parent,
            self.current_player,
            self.result,
            self.action_count,
            self.child,
            self.root_player,
            remaining_tensor,
            self.context_count,
            self.max_nodes,
            MAX_ACTIONS,
            max_requests,
        )
        packed = torch.cat(
            (
                expansion_parent.flatten(),
                expansion_edge.flatten(),
                advanced_count,
            )
        )
        transfer_started = time.perf_counter()
        rows = packed.detach().cpu().tolist()
        profile.gpu_to_cpu_seconds += time.perf_counter() - transfer_started
        profile.gpu_tree_seconds += time.perf_counter() - started

        flat_count = self.context_count * max_requests
        parent_rows = rows[:flat_count]
        edge_rows = rows[flat_count : 2 * flat_count]
        advanced_rows = [int(value) for value in rows[2 * flat_count :]]
        expansions: list[tuple[int, int, int]] = []
        for context_index in range(self.context_count):
            for request_index in range(max_requests):
                offset = context_index * max_requests + request_index
                parent_index = int(parent_rows[offset])
                if parent_index >= 0:
                    expansions.append(
                        (
                            context_index,
                            parent_index,
                            int(edge_rows[offset]),
                        )
                    )
        return expansions, advanced_rows

    def link_children(
        self,
        records: list[tuple[int, int, int, int]],
    ) -> None:
        if not records:
            return
        if self.mps_kernels is not None:
            metadata_values: list[int] = []
            for record in records:
                metadata_values.extend(record)
            metadata = torch.tensor(
                metadata_values, dtype=torch.int64, device=self.device
            )
            launch = torch.empty(
                len(records), dtype=torch.int64, device=self.device
            )
            self.mps_kernels.link_children(
                launch,
                self.child,
                metadata,
                len(records),
                self.max_nodes,
                MAX_ACTIONS,
            )
            return
        contexts = torch.tensor(
            [record[0] for record in records],
            dtype=torch.int64,
            device=self.device,
        )
        parents = torch.tensor(
            [record[1] for record in records],
            dtype=torch.int64,
            device=self.device,
        )
        edges = torch.tensor(
            [record[2] for record in records],
            dtype=torch.int64,
            device=self.device,
        )
        nodes = torch.tensor(
            [record[3] for record in records],
            dtype=torch.int64,
            device=self.device,
        )
        self.child[contexts, parents, edges] = nodes

    def finish_actions(self, profile: GpuTreeProfile) -> dict[int, list[int]]:
        if self.mps_kernels is not None:
            best_edge = torch.empty(
                self.context_count, dtype=torch.int64, device=self.device
            )
            self.mps_kernels.finish_actions(
                best_edge,
                self.child,
                self.visit,
                self.prior,
                self.context_count,
                self.max_nodes,
                MAX_ACTIONS,
            )
            started = time.perf_counter()
            edge_rows = best_edge.detach().cpu().tolist()
            profile.gpu_to_cpu_seconds += time.perf_counter() - started
            return self._actions_from_edges(edge_rows)

        children = self.child[: self.context_count, 0]
        safe_children = children.clamp(min=0)
        visits = self.visit[self.context_range[:, None], safe_children]
        expanded = children >= 0
        visit_scores = visits.masked_fill(~expanded, -torch.inf)
        best_visited = torch.argmax(visit_scores, dim=1)
        any_expanded = torch.any(expanded, dim=1)
        best_prior = torch.argmax(self.prior[: self.context_count, 0], dim=1)
        best_edge = torch.where(any_expanded, best_visited, best_prior)

        started = time.perf_counter()
        edge_rows = best_edge.detach().cpu().tolist()
        profile.gpu_to_cpu_seconds += time.perf_counter() - started
        return self._actions_from_edges(edge_rows)

    def _actions_from_edges(self, edge_rows: list[int]) -> dict[int, list[int]]:
        selected: dict[int, list[int]] = {}
        for context_index, edge in enumerate(edge_rows):
            root_actions = self.actions[context_index][0]
            if not root_actions:
                observation = self.contexts[context_index].session.observation
                selected[self.contexts[context_index].session.battle_ptr] = (
                    _random_action(observation)
                )
            else:
                selected[self.contexts[context_index].session.battle_ptr] = (
                    root_actions[int(edge)]
                )
        return selected


def _evaluate_pending(
    group: _GpuMctsGroup,
    pending: list[_PendingEvaluation],
    batch_size: int,
    profile: GpuTreeProfile,
) -> None:
    if not pending:
        return
    model = group.contexts[0].participant.model
    if model is None:
        raise RuntimeError("GPU MCTSにNNモデルがありません。")

    for offset in range(0, len(pending), batch_size):
        chunk = pending[offset : offset + batch_size]
        required_decoder_words = max(len(item.decoder.offset) for item in chunk)
        decoder_words = 1 << (required_decoder_words - 1).bit_length()
        for item in chunk:
            _pad_sparse_offsets(item.decoder, decoder_words)
        encoder = _combine_sparse([item.encoder for item in chunk])
        decoder = _combine_sparse([item.decoder for item in chunk])

        # 4本の整数配列と2本のfloat配列をそれぞれ1本にpackすることで、
        # CPU→device転送を6回から2回へ減らす。
        int_lengths = (
            len(encoder[0]),
            len(encoder[2]),
            len(decoder[0]),
            len(decoder[2]),
        )
        float_lengths = (len(encoder[1]), len(decoder[1]))
        packed_int = encoder[0] + encoder[2] + decoder[0] + decoder[2]
        packed_float = encoder[1] + decoder[1]

        started = time.perf_counter()
        int_tensor = torch.tensor(
            packed_int, dtype=torch.int32, device=group.device
        )
        float_tensor = torch.tensor(
            packed_float, dtype=torch.float32, device=group.device
        )
        i0, i1, i2, i3 = int_lengths
        f0, f1 = float_lengths
        index_encoder = int_tensor[:i0]
        offset_encoder = int_tensor[i0 : i0 + i1]
        index_decoder = int_tensor[i0 + i1 : i0 + i1 + i2]
        offset_decoder = int_tensor[i0 + i1 + i2 : i0 + i1 + i2 + i3]
        value_encoder = float_tensor[:f0]
        value_decoder = float_tensor[f0 : f0 + f1]
        with torch.inference_mode():
            values, policies = model(
                index_encoder,
                value_encoder,
                offset_encoder,
                index_decoder,
                value_decoder,
                offset_decoder,
            )
            group.apply_evaluation(chunk, values, policies)
        profile.nn_seconds += time.perf_counter() - started
        profile.nn_evaluations += len(chunk)
        profile.nn_batches += 1
        profile.batch_sizes.append(len(chunk))
        profile.packed_input_bytes += int_tensor.numel() * int_tensor.element_size()
        profile.packed_input_bytes += float_tensor.numel() * float_tensor.element_size()


def _begin_search_state(
    runtime: _Runtime,
    context: _GpuContext,
    profile: GpuTreeProfile,
) -> Any:
    observation = runtime.to_observation_class(context.session.observation)
    state = observation.current
    active = state.players[1 - context.your_index].active
    deck = context.participant.deck
    started = time.perf_counter()
    search_state = runtime.search_begin(
        observation,
        your_deck=random.sample(
            deck,
            min(len(deck), state.players[context.your_index].deckCount),
        ),
        your_prize=random.sample(
            deck,
            min(len(deck), len(state.players[context.your_index].prize)),
        ),
        opponent_deck=[1072] * state.players[1 - context.your_index].deckCount,
        opponent_prize=[1] * len(state.players[1 - context.your_index].prize),
        opponent_hand=[1] * state.players[1 - context.your_index].handCount,
        opponent_active=[1072] if len(active) > 0 and active[0] is None else [],
    )
    profile.search_begin_seconds += time.perf_counter() - started
    return search_state


def _run_gpu_search_wave(
    runtime: _Runtime,
    contexts: list[_GpuContext],
    search_count: int,
    device: torch.device,
    batch_size: int,
    profile: GpuTreeProfile,
    workspaces: dict[str, _GpuMctsGroup],
    workspace_capacity: int,
) -> dict[int, list[int]]:
    if not contexts:
        return {}

    grouped: dict[str, list[tuple[_GpuContext, Any]]] = defaultdict(list)
    search_started = False
    try:
        for context in contexts:
            state = _begin_search_state(runtime, context, profile)
            search_started = True
            key = context.participant.model_key
            if key is None:
                raise RuntimeError("GPU MCTS contextにmodel keyがありません。")
            grouped[key].append((context, state))

        selected_actions: dict[int, list[int]] = {}
        for model_key, entries in grouped.items():
            group_contexts = [entry[0] for entry in entries]
            group = workspaces.get(model_key)
            if group is None:
                group = _GpuMctsGroup(
                    group_contexts,
                    device,
                    max_nodes=search_count + 1,
                    capacity=workspace_capacity,
                )
                workspaces[model_key] = group
            else:
                group.reset(group_contexts)
            root_records = [
                (context_index, 0, -1, entry[1])
                for context_index, entry in enumerate(entries)
            ]
            pending = group.install_nodes(runtime, root_records, profile)
            _evaluate_pending(group, pending, batch_size, profile)

            def advance_expansions(
                expansions: list[tuple[int, int, int]],
            ) -> None:
                new_nodes: list[tuple[int, int, int, int]] = []
                install_records: list[tuple[int, int, int, Any]] = []
                for context_index, parent_index, edge_index in expansions:
                    parent_state = group.states[context_index][parent_index]
                    parent_actions = group.actions[context_index][parent_index]
                    if parent_state is None or parent_actions is None:
                        raise RuntimeError("GPU MCTSのCPU state/actionがありません。")
                    action = parent_actions[edge_index]
                    started = time.perf_counter()
                    next_state = runtime.search_step(parent_state.searchId, action)
                    profile.search_step_seconds += time.perf_counter() - started
                    profile.search_steps += 1

                    node_index = group.next_node[context_index]
                    if node_index >= group.max_nodes:
                        raise RuntimeError("GPU MCTS node容量を超えました。")
                    group.next_node[context_index] += 1
                    new_nodes.append(
                        (context_index, parent_index, edge_index, node_index)
                    )
                    install_records.append(
                        (context_index, node_index, parent_index, next_state)
                    )
                group.link_children(new_nodes)
                pending = group.install_nodes(runtime, install_records, profile)
                _evaluate_pending(group, pending, batch_size, profile)

            if group.mps_kernels is not None:
                remaining = [search_count] * group.context_count
                while any(value > 0 for value in remaining):
                    expansions, advanced = group.select_parallel_expansions(
                        remaining,
                        search_count,
                        profile,
                    )
                    if not any(advanced):
                        break
                    remaining = [
                        max(0, value - completed)
                        for value, completed in zip(
                            remaining, advanced, strict=True
                        )
                    ]
                    advance_expansions(expansions)
            else:
                for _ in range(search_count):
                    expansions = group.select_expansions(profile)
                    advance_expansions(expansions)

            selected_actions.update(group.finish_actions(profile))
        return selected_actions
    finally:
        if search_started:
            started = time.perf_counter()
            runtime.search_end()
            profile.search_finalize_seconds += time.perf_counter() - started


def run_gpu_tree_tournament(
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
    """MCTS treeをdevice常駐させて総当たりを実行する。"""
    if batch_size < 1 or lanes < 1:
        raise ValueError("batch_sizeとlanesは1以上で指定してください。")
    if search_count < 1:
        raise ValueError("gpu-tree backendのsearch_countは1以上必要です。")

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
        raise ValueError("gpu-tree backendにはrl_mcts agentが必要です。")
    runtime = _load_runtime(Path(canonical_spec.agent_path).resolve().parent)
    participants = _load_participants(specs, runtime, device)
    profile = GpuTreeProfile()
    workspaces: dict[str, _GpuMctsGroup] = {}

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
            contexts: list[_GpuContext] = []
            for session in active:
                observation = session.observation
                select = observation.get("select")
                if select is None or not select["option"] or int(select["maxCount"]) == 0:
                    selected_actions[session.battle_ptr] = []
                    continue
                your_index = int(observation["current"]["yourIndex"])
                participant = session.players[your_index]
                if participant.random_policy:
                    selected_actions[session.battle_ptr] = _random_action(observation)
                else:
                    contexts.append(
                        _GpuContext(session, participant, your_index)
                    )

            selected_actions.update(
                _run_gpu_search_wave(
                    runtime,
                    contexts,
                    search_count,
                    device,
                    batch_size,
                    profile,
                    workspaces,
                    lanes,
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
    return BatchedTournamentOutput(results, profile, str(device))
