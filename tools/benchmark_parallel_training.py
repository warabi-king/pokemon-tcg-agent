"""独立したモデル学習ジョブを並列実行して所要時間を測る。

本番のモデルやcheckpointは更新しない。各ジョブは指定された前処理済みshardを
読み、``output_dir`` 配下へmetrics/logを書き出す。中央device方式では既定で
モデルファイルを作らず、その他の方式でも計測後に一時モデルだけ削除する。
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import random
import re
import runpy
import subprocess
import sys
import time
import traceback


@dataclass(frozen=True)
class TrainingJob:
    name: str
    shards: Path
    initial_model: Path
    seed: int
    samples: int = 0
    source_games: int = 0


def _manifest_count(path: Path, key: str) -> int:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return max(0, int(manifest.get(key, 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0


def discover_training_jobs(
    shards_root: Path,
    agents_root: Path,
    *,
    seed: int,
) -> list[TrainingJob]:
    """前処理済みshardからself/opponent学習ジョブを安定した順序で作る。"""
    shards_root = Path(shards_root).resolve()
    agents_root = Path(agents_root).resolve()
    if not shards_root.is_dir():
        raise FileNotFoundError(f"shards rootがありません: {shards_root}")
    if not agents_root.is_dir():
        raise FileNotFoundError(f"agents rootがありません: {agents_root}")

    own_manifests = sorted(shards_root.glob("*_own/manifest.json"))
    if not own_manifests:
        raise FileNotFoundError(f"own shard manifestがありません: {shards_root}")

    jobs: list[TrainingJob] = []
    for agent_index, manifest in enumerate(own_manifests):
        shard_name = manifest.parent.name
        agent_name = shard_name.removesuffix("_own")
        src = agents_root / agent_name / "src"
        self_model = src / "model.pth"
        opponent_model = src / "opponent_model.pth"
        if not self_model.is_file():
            raise FileNotFoundError(f"初期モデルがありません: {self_model}")
        if not opponent_model.is_file():
            opponent_model = self_model

        role_specs = (
            ("self", shards_root / f"{agent_name}_own", self_model),
            (
                "opponent",
                shards_root / f"{agent_name}_opponent",
                opponent_model,
            ),
        )
        for role_index, (role, shards, initial_model) in enumerate(role_specs):
            role_manifest = shards / "manifest.json"
            if not role_manifest.is_file():
                raise FileNotFoundError(f"shard manifestがありません: {shards}")
            if not any(shards.glob("shard_*.pkl")):
                raise FileNotFoundError(f"shardがありません: {shards}")
            jobs.append(
                TrainingJob(
                    name=f"{agent_name}_{role}",
                    shards=shards,
                    initial_model=initial_model,
                    seed=seed + agent_index * 2 + role_index,
                    samples=_manifest_count(role_manifest, "samples"),
                    source_games=_manifest_count(role_manifest, "episodes"),
                )
            )
    return jobs


def schedule_training_jobs(
    jobs: list[TrainingJob],
    order: str,
) -> list[TrainingJob]:
    """全体の終了待ちを短くする順序へ学習ジョブを並べる。"""
    if order == "discovery":
        return list(jobs)
    if order == "largest-first":
        return sorted(jobs, key=lambda job: (-job.samples, job.name))
    raise ValueError(f"未対応のschedule_orderです: {order}")


def assign_training_jobs(
    jobs: list[TrainingJob],
    workers: int,
) -> list[list[TrainingJob]]:
    """重いjobから、推定サンプル数が最も少ないCPU workerへ割り当てる。"""
    if workers < 1:
        raise ValueError("workersは1以上にしてください。")
    assignments: list[list[TrainingJob]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for job in sorted(jobs, key=lambda item: (-item.samples, item.name)):
        worker_index = min(range(workers), key=lambda index: (loads[index], index))
        assignments[worker_index].append(job)
        loads[worker_index] += job.samples
    return assignments


# SELFPLAY_COMPLETED_Q_TARGET_PATCH_V1
_POLICY_TARGET_MODES = frozenset({"hard", "completed_q"})


def _policy_target_mode() -> str:
    mode = os.environ.get("SELFPLAY_POLICY_TARGET", "hard")
    if mode not in _POLICY_TARGET_MODES:
        raise ValueError(
            f"SELFPLAY_POLICY_TARGETが不正です: {mode!r} "
            f"(有効値: {sorted(_POLICY_TARGET_MODES)})"
        )
    return mode


# SELFPLAY_VALUE_LOSS_PATCH_V1
_VALUE_LOSS_KINDS = frozenset({"huber", "mse"})


def _value_loss_config() -> tuple[str, float, float]:
    """valueヘッドの損失(種類, Huberのdelta, 重み)を返す。既定は従来と同じ。

    従来は ``HuberLoss(delta=0.2)`` を重み1.0で policy の cross entropy に足していた。
    Huberは |誤差|>delta で勾配が一定(=L1)になるので、教師が最終勝敗の±1しかない
    この設計では条件付き「中央値」に寄る。中央値は勝ち局面なら+1、負け局面なら-1に
    振り切れるため、「どのくらい有利か」という大きさの情報が落ちる。
    q tie-breakの採用で着手はrootの子のQ値(=valueヘッド)の大小比較で決まるように
    なったので、この大きさの情報はそのまま着手の質になる。MSEなら条件付き平均
    (=勝率の線形変換)に寄るため、比較したい量そのものを学習することになる。
    実測ではvalue lossは全体の約7%(0.095 / 1.33)しかない。
    """
    kind = os.environ.get("SELFPLAY_VALUE_LOSS", "huber")
    if kind not in _VALUE_LOSS_KINDS:
        raise ValueError(
            f"SELFPLAY_VALUE_LOSSが不正です: {kind!r} "
            f"(有効値: {sorted(_VALUE_LOSS_KINDS)})"
        )
    delta = float(os.environ.get("SELFPLAY_VALUE_HUBER_DELTA", "0.2"))
    if delta <= 0.0:
        raise ValueError(f"SELFPLAY_VALUE_HUBER_DELTAは正の値が必要です: {delta}")
    weight = float(os.environ.get("SELFPLAY_VALUE_LOSS_WEIGHT", "1.0"))
    if weight < 0.0:
        raise ValueError(f"SELFPLAY_VALUE_LOSS_WEIGHTは0以上にしてください: {weight}")
    return kind, delta, weight


def _build_value_loss(torch_module: object) -> tuple[object, float]:
    kind, delta, weight = _value_loss_config()
    if kind == "mse":
        return torch_module.nn.MSELoss(), weight
    return torch_module.nn.HuberLoss(delta=delta), weight


def _policy_target_zero_spread() -> str:
    """Q優位度が全合法手で同値だった行の扱い(既定はhard labelへ退避)。

    実測(2026-08-05, 実shard 107,997サンプル)で28.2%の行がこれに当たるため、
    softmaxをそのまま当てると学習データの28%が「方策を平坦にせよ」という
    勾配になる。既定では従来のhard labelのままにする。
    """
    mode = os.environ.get("SELFPLAY_POLICY_TARGET_ZERO_SPREAD", "hard")
    if mode not in {"hard", "uniform"}:
        raise ValueError(
            f"SELFPLAY_POLICY_TARGET_ZERO_SPREADが不正です: {mode!r} "
            "(有効値: ['hard', 'uniform'])"
        )
    return mode


def _value_target_lambda() -> float:
    """valueの教師を「最終勝敗」と「探索rootの評価値」で混ぜる比率。

    既定1.0は従来どおり最終勝敗そのまま。1試合の全局面へ一律に±1を付ける教師は
    分散が最大で、q tie-break採用後はQ値が着手を決めるためvalueヘッドの質が
    直接効く。SELFPLAY_SEARCH_VALUE_TARGET_PATCH_V1
    """
    lam = float(os.environ.get("SELFPLAY_VALUE_TARGET_LAMBDA", "1.0"))
    if not 0.0 <= lam <= 1.0:
        raise ValueError(
            f"SELFPLAY_VALUE_TARGET_LAMBDAは0.0〜1.0にしてください: {lam}"
        )
    return lam


# SELFPLAY_VALUE_TARGET_DISCOUNT_PATCH_V1
def _value_target_discount_final() -> float:
    """valueの教師に掛ける割引の最終値。1.0で従来どおり全局面に最終勝敗そのまま。

    従来の教師は試合中の全局面へ一律に±1を付けるため、終局から遠い局面にも
    「勝ち確定」と教えている(実測: 予測+0.999の局面の実際の平均結果は+0.583)。
    方策側と同じく 最終値^(残りターン数/総ターン数) を掛けると、序盤ほど0へ
    寄る教師になる。序盤の局面は組み合わせが少なく学習データに繰り返し現れる
    ので、頻度の偏りを抑える方向にも働く。
    """
    discount_final = float(
        os.environ.get("SELFPLAY_VALUE_TARGET_DISCOUNT_FINAL", "1.0")
    )
    if not 0.0 < discount_final <= 1.0:
        raise ValueError(
            "SELFPLAY_VALUE_TARGET_DISCOUNT_FINALは0より大きく1以下にしてください: "
            f"{discount_final}"
        )
    return discount_final


# SELFPLAY_OUTCOME_WEIGHTED_POLICY_PATCH_V1
_POLICY_LOSS_KINDS = frozenset({"cross_entropy", "outcome_weighted"})


def _policy_loss_kind() -> str:
    """方策損失の形。既定は従来どおり打った手への交差エントロピー。

    outcome_weighted では、打った手の対数確率に
    「その試合の最終結果 × 残りターン数に応じた割引」を掛ける。
    勝った試合の手は確率を上げ、負けた試合の手は下げる。基準線は使わない。
    """
    kind = os.environ.get("SELFPLAY_POLICY_LOSS", "cross_entropy")
    if kind not in _POLICY_LOSS_KINDS:
        raise ValueError(
            f"SELFPLAY_POLICY_LOSSが不正です: {kind!r} "
            f"(有効値: {sorted(_POLICY_LOSS_KINDS)})"
        )
    return kind


def _policy_discount_final() -> float:
    """割引の最終値。1.0で割引なし(位置による差を付けない)。

    重みは 割引の最終値^(終局までの残りターン数 / その試合の総ターン数)。
    指数は最終ターンで0、初手で1に近づくので、割引は最終ターンで1.0(割引なし)、
    初手でこの値まで下がる。試合ごとに総ターン数で正規化した割引なので、
    固定の割引率と違って試合の長さで扱いが変わらない。0.3を指定すると
    「初手は最終ターンの0.3倍まで割り引く」という意味になる。
    """
    discount_final = float(os.environ.get("SELFPLAY_POLICY_DISCOUNT_FINAL", "1.0"))
    if not 0.0 < discount_final <= 1.0:
        raise ValueError(
            "SELFPLAY_POLICY_DISCOUNT_FINALは0より大きく1以下にしてください: "
            f"{discount_final}"
        )
    return discount_final


def _policy_weight_opponent_jobs() -> bool:
    """相手モデルの学習にも結果重みを掛けるか。

    相手モデルは「相手が実際に打つ手」を当てるための予測器であり、強さを
    上げる対象ではない。結果で重み付けすると「勝つ手を予測する」方向へ
    歪むため、既定では相手モデルは従来の交差エントロピーのままにする。
    """
    return os.environ.get("SELFPLAY_POLICY_LOSS_OPPONENT", "0") == "1"


def _policy_target_temperature() -> float:
    temperature = float(
        os.environ.get("SELFPLAY_POLICY_TARGET_TEMPERATURE", "0.25")
    )
    if temperature <= 0.0:
        raise ValueError("SELFPLAY_POLICY_TARGET_TEMPERATUREは正の値が必要です。")
    return temperature


def _build_batch_arrays(
    batch: list[tuple],
    max_actions: int,
) -> tuple[object, ...]:
    """torchを読み込まないCPU workerで、1 minibatch分のNumPy配列を作る。"""
    import numpy as np

    encoder_index: list[int] = []
    encoder_value: list[float] = []
    encoder_offset: list[int] = []
    decoder_index: list[int] = []
    decoder_value: list[float] = []
    decoder_offset: list[int] = []
    label_value: list[float] = []
    chosen_indices: list[int] = []
    mask = np.zeros((len(batch), max_actions), dtype=np.bool_)
    # SELFPLAY_COMPLETED_Q_TARGET_PATCH_V1
    # サンプルは (…, chosen_index, value) の8要素に加えて、合法手ごとのQ優位度を
    # 9要素目として持つことがある(旧shardや探索なし局面では空)。
    mode = _policy_target_mode()
    temperature = _policy_target_temperature()
    zero_spread = _policy_target_zero_spread()
    value_lambda = _value_target_lambda()
    soft_target = (
        np.zeros((len(batch), max_actions), dtype=np.float32)
        if mode == "completed_q"
        else None
    )
    # SELFPLAY_OUTCOME_WEIGHTED_POLICY_PATCH_V1
    # 打った手ごとの重み = 最終結果 × 割引の最終値^(残りターン数 / 総ターン数)。
    # 常に組み立てて配列の並びを固定する(cross_entropy時は使われない)。
    discount_final = _policy_discount_final()
    value_discount_final = _value_target_discount_final()
    policy_weight = np.zeros(len(batch), dtype=np.float32)

    for row_index, sample in enumerate(batch):
        (
            enc_index,
            enc_value,
            enc_offset,
            dec_index,
            dec_value,
            dec_offset,
            chosen_index,
            value,
        ) = sample[:8]
        completed_q = sample[8] if len(sample) > 8 else ()
        search_value = sample[9] if len(sample) > 9 else None
        # 割引の基準は「その試合の最終結果」なので、value教師の混合より前に取る。
        outcome = float(value)
        remaining_fraction = float(sample[10]) if len(sample) > 10 else 0.0
        policy_weight[row_index] = outcome * (discount_final**remaining_fraction)
        # SELFPLAY_VALUE_TARGET_DISCOUNT_PATCH_V1: valueの教師も同じ形で割り引く。
        if value_discount_final < 1.0:
            value = outcome * (value_discount_final**remaining_fraction)
        if value_lambda < 1.0 and search_value is not None:
            value = value_lambda * value + (1.0 - value_lambda) * float(search_value)
        enc_count = len(encoder_index)
        encoder_index.extend(enc_index)
        encoder_value.extend(enc_value)
        encoder_offset.extend(offset + enc_count for offset in enc_offset)

        dec_count = len(decoder_index)
        decoder_index.extend(dec_index)
        decoder_value.extend(dec_value)
        decoder_offset.extend(offset + dec_count for offset in dec_offset)
        decoder_offset.extend(
            [len(decoder_index)] * (max_actions - len(dec_offset))
        )
        mask[row_index, : len(dec_offset)] = True
        label_value.append(value)
        chosen_indices.append(chosen_index)
        if soft_target is not None:
            n_candidates = len(dec_offset)
            flat = (
                len(completed_q) > 0
                and max(completed_q) == min(completed_q)
                and zero_spread == "hard"
            )
            if len(completed_q) == n_candidates and n_candidates > 0 and not flat:
                scaled = np.asarray(completed_q, dtype=np.float32) / temperature
                weights = np.exp(scaled - scaled.max())
                total = float(weights.sum())
                if total > 0.0:
                    soft_target[row_index, :n_candidates] = weights / total
                else:
                    soft_target[row_index, chosen_index] = 1.0
            else:
                # 探索が走らなかった局面や旧shardはhard labelへ退避する。
                soft_target[row_index, chosen_index] = 1.0

    arrays = [
        np.asarray(encoder_index, dtype=np.int32),
        np.asarray(encoder_value, dtype=np.float32),
        np.asarray(encoder_offset, dtype=np.int32),
        np.asarray(decoder_index, dtype=np.int32),
        np.asarray(decoder_value, dtype=np.float32),
        np.asarray(decoder_offset, dtype=np.int32),
        mask,
        np.asarray(label_value, dtype=np.float32).reshape(-1, 1),
        np.asarray(chosen_indices, dtype=np.int64),
        policy_weight,
    ]
    if soft_target is not None:
        arrays.append(soft_target)
    return tuple(arrays)


def _central_loader_worker(
    worker_id: int,
    jobs: list[TrainingJob],
    epochs: int,
    batch_size: int,
    max_actions: int,
    message_queue: object,
) -> None:
    """shard I/O・shuffle・NumPy batch組立だけを行う軽量CPU worker。"""
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    try:
        for job in jobs:
            message_queue.put(("job_start", worker_id, job))
            generator = random.Random(job.seed)
            shard_paths = sorted(job.shards.glob("shard_*.pkl"))
            generator.shuffle(shard_paths)
            for epoch in range(epochs):
                epoch_paths = list(shard_paths)
                generator.shuffle(epoch_paths)
                for path in epoch_paths:
                    with path.open("rb") as shard_file:
                        samples = pickle.load(shard_file)
                    generator.shuffle(samples)
                    stop = len(samples) - batch_size + 1
                    for start in range(0, stop, batch_size):
                        arrays = _build_batch_arrays(
                            samples[start : start + batch_size],
                            max_actions,
                        )
                        message_queue.put(
                            ("batch", worker_id, job.name, epoch, arrays)
                        )
                message_queue.put(
                    ("epoch_done", worker_id, job.name, epoch)
                )
            message_queue.put(("job_done", worker_id, job.name))
        message_queue.put(("worker_done", worker_id))
    except BaseException:
        message_queue.put(("worker_error", worker_id, traceback.format_exc()))


def _empty_epoch_stats() -> dict[str, object]:
    return {
        "batches": 0,
        "total_seen": 0,
        "loss": None,
        "loss_value": None,
        "loss_policy": None,
        "correct": None,
    }


def _accumulate_tensor(
    current: object | None,
    value: object,
) -> object:
    detached = value.detach()
    return detached if current is None else current + detached


def _training_state_path(
    output_model: Path,
    initial_model: Path,
) -> Path | None:
    """stateful版(train_imitation_stateful.py)と同じstateファイル位置を返す。

    SELFPLAY_TRAINING_STATE_ROOTが未設定なら None を返し、従来どおり
    optimizer状態を持ち越さない挙動のままにする(過去実験の再現用)。
    """
    state_root = os.environ.get("SELFPLAY_TRAINING_STATE_ROOT")
    if not state_root:
        return None
    text = f"{output_model} {initial_model}"
    match = re.search(r"cluster_\d+", text)
    if not match:
        return None
    role = (
        "opponent"
        if "opponent_model" in text or "opponent" in Path(output_model).stem
        else "self"
    )
    return Path(state_root) / match.group(0) / role / "training_state.pth"


def _load_training_state(
    torch,
    optimizer,
    *,
    base_lr: float,
    output_model: Path,
    initial_model: Path,
    device,
    log_file: Path,
) -> tuple[object, int, Path | None]:
    """AdamW/schedulerの状態を前回更新から引き継ぐ。

    AdamWのモーメントの引き継ぎと、学習率の減衰は独立している。既定の
    SELFPLAY_LR_GAMMA=1.0では学習率は減衰せず、モーメントの引き継ぎだけが働く。
    減衰は別セッションの実測で 3e-4→8.15e-5(73%減)まで動かしても
    −2.33pt(p=0.20)と改善が出なかったため、既定では無効にしている。
    """
    gamma = float(os.environ.get("SELFPLAY_LR_GAMMA", "1.0"))
    min_lr = float(os.environ.get("SELFPLAY_MIN_LR", "3e-5"))
    min_factor = min_lr / base_lr if base_lr > 0 else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(min_factor, gamma**step),
    )
    state_path = _training_state_path(output_model, initial_model)
    global_step = 0
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state.get("global_step", 0))
        message = (
            f"loaded training state: {state_path} global_step={global_step} "
            f"lr={optimizer.param_groups[0]['lr']:.8g}\n"
        )
    elif state_path is not None:
        message = f"new training state: {state_path}\n"
    else:
        message = "training state disabled (SELFPLAY_TRAINING_STATE_ROOT未設定)\n"
    with log_file.open("a", encoding="utf-8") as file:
        file.write(message)
    return scheduler, global_step, state_path


def _save_training_state(
    torch,
    state: dict,
    *,
    gamma: float | None = None,
    min_lr: float | None = None,
) -> None:
    state_path = state.get("state_path")
    if state_path is None:
        return
    payload = {
        "optimizer": state["optimizer"].state_dict(),
        "scheduler": state["scheduler"].state_dict(),
        "global_step": int(state["global_step"]),
        "base_lr": float(state["base_lr"]),
        "gamma": (
            gamma
            if gamma is not None
            else float(os.environ.get("SELFPLAY_LR_GAMMA", "1.0"))
        ),
        "min_lr": (
            min_lr
            if min_lr is not None
            else float(os.environ.get("SELFPLAY_MIN_LR", "3e-5"))
        ),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, state_path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_central_device_training(
    jobs: list[TrainingJob],
    *,
    train_script: Path,
    output_dir: Path,
    workers: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    keep_models: bool,
    device_wave_size: int,
) -> tuple[list[dict[str, object]], str]:
    """CPU前処理を並列化し、torch/MPSは中央processだけで実行する。"""
    runtime = runpy.run_path(
        str(train_script),
        run_name="_benchmark_training_runtime",
    )
    torch = runtime["torch"]
    functional = runtime["F"]
    create_model = runtime["create_model"]
    select_device = runtime["select_device"]
    max_actions = int(runtime["MAX_ACTIONS"])
    device = select_device(device_name)
    # SELFPLAY_VALUE_LOSS_PATCH_V1: 既定は従来どおり HuberLoss(delta=0.2) の重み1.0。
    loss_fn_value, value_loss_weight = _build_value_loss(torch)
    # SELFPLAY_OUTCOME_WEIGHTED_POLICY_PATCH_V1
    policy_loss_kind = _policy_loss_kind()
    weight_opponent_jobs = _policy_weight_opponent_jobs()
    if policy_loss_kind == "outcome_weighted":
        print(
            f"policy loss: outcome_weighted (discount_final={_policy_discount_final()}, "
            f"opponent_jobs={'weighted' if weight_opponent_jobs else 'cross_entropy'})",
            flush=True,
        )

    worker_count = min(workers, len(jobs))
    assignments = assign_training_jobs(jobs, worker_count)
    context = multiprocessing.get_context("spawn")
    message_queue = context.Queue(maxsize=max(2, worker_count * 2))
    processes = [
        context.Process(
            target=_central_loader_worker,
            args=(
                worker_id,
                worker_jobs,
                epochs,
                batch_size,
                max_actions,
                message_queue,
            ),
        )
        for worker_id, worker_jobs in enumerate(assignments)
    ]
    for process in processes:
        process.start()

    states: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    completed_workers = 0
    pending_messages: deque[tuple[object, ...]] = deque()

    def process_batch_wave(messages: list[tuple[object, ...]]) -> None:
        prepared: list[tuple[dict[str, object], list[object], object, ...]] = []
        for message in messages:
            _kind, _worker_id, job_name, _epoch, arrays = message
            state = states[str(job_name)]
            model = state["model"]
            optimizer = state["optimizer"]
            tensors = [torch.from_numpy(array).to(device) for array in arrays]
            optimizer.zero_grad(set_to_none=True)
            out_enc, out_dec = model(*tensors[:6])
            mask_tensor, label_value_tensor, chosen_index_tensor = tensors[6:9]
            policy_weight_tensor = tensors[9]
            soft_target_tensor = tensors[10] if len(tensors) > 10 else None
            loss_value = loss_fn_value(out_enc, label_value_tensor)
            masked_logits = out_dec.masked_fill(~mask_tensor, float("-inf"))
            if policy_loss_kind == "outcome_weighted" and (
                weight_opponent_jobs or not str(job_name).endswith("_opponent")
            ):
                # SELFPLAY_OUTCOME_WEIGHTED_POLICY_PATCH_V1
                # 打った手の対数確率に「最終結果 × 残りターン数に応じた割引」を掛ける。
                # 重みが負(負け試合)なら、その手の確率を下げる勾配になる。
                log_probabilities = torch.log_softmax(masked_logits, dim=1)
                chosen_log_probability = log_probabilities.gather(
                    1, chosen_index_tensor.unsqueeze(1)
                ).squeeze(1)
                loss_policy = -(
                    policy_weight_tensor * chosen_log_probability
                ).mean()
            elif soft_target_tensor is None:
                loss_policy = functional.cross_entropy(
                    masked_logits,
                    chosen_index_tensor,
                )
            else:
                # maskされた位置のlog_probは-infなので 0*-inf=nan を避けて0で埋める。
                log_probabilities = torch.log_softmax(masked_logits, dim=1)
                log_probabilities = log_probabilities.masked_fill(
                    ~mask_tensor, 0.0
                )
                loss_policy = -(
                    soft_target_tensor * log_probabilities
                ).sum(dim=1).mean()
            loss = value_loss_weight * loss_value + loss_policy
            prepared.append(
                (
                    state,
                    tensors,
                    loss,
                    loss_value,
                    loss_policy,
                    masked_logits,
                    chosen_index_tensor,
                )
            )

        # 異なるmodelのforwardを先に全てMPS command queueへ投入してから
        # backward/optimizerを実行する。ゲーム側の中央waveと同じ考え方。
        for (
            state,
            _tensors,
            loss,
            loss_value,
            loss_policy,
            masked_logits,
            chosen_index_tensor,
        ) in prepared:
            loss.backward()
            state["optimizer"].step()
            state["scheduler"].step()
            state["global_step"] = int(state["global_step"]) + 1
            stats = state["stats"]
            stats["batches"] += 1
            stats["total_seen"] += int(chosen_index_tensor.shape[0])
            stats["loss"] = _accumulate_tensor(stats["loss"], loss)
            stats["loss_value"] = _accumulate_tensor(
                stats["loss_value"],
                loss_value,
            )
            stats["loss_policy"] = _accumulate_tensor(
                stats["loss_policy"],
                loss_policy,
            )
            with torch.no_grad():
                correct = (
                    masked_logits.argmax(dim=1) == chosen_index_tensor
                ).sum()
            stats["correct"] = _accumulate_tensor(
                stats["correct"],
                correct,
            )

    try:
        while completed_workers < worker_count:
            message = (
                pending_messages.popleft()
                if pending_messages
                else message_queue.get()
            )
            kind = message[0]
            if kind == "job_start":
                _kind, worker_id, job = message
                output_model, metrics_file, log_file, _job_dir = _job_paths(
                    job,
                    output_dir,
                )
                fieldnames = [
                    "epoch",
                    "batches",
                    "loss",
                    "loss_value",
                    "loss_policy",
                    "train_accuracy",
                    "val_accuracy",
                    "learning_rate",
                    "global_step",
                    "elapsed_seconds",
                ]
                with metrics_file.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=fieldnames).writeheader()
                log_file.write_text(
                    f"device: {device}\nCPU loader worker: {worker_id}\n",
                    encoding="utf-8",
                )
                torch.manual_seed(job.seed)
                model = create_model().to(device)
                model.load_state_dict(
                    torch.load(job.initial_model, map_location=device)
                )
                model.train()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                )
                # SELFPLAY_TRAINING_STATE_PATCH_V1: この中央device経路は
                # --train-script のmain()を実行しないため、stateful版が持つ
                # AdamW state継続とlr schedulerが従来まったく効いていなかった
                # (更新ごとに新しいoptimizer・lr固定)。stateful版と同じ
                # 環境変数・同じstateファイル形式でここでも継続させる。
                scheduler, global_step, state_path = _load_training_state(
                    torch,
                    optimizer,
                    base_lr=learning_rate,
                    output_model=output_model,
                    initial_model=job.initial_model,
                    device=device,
                    log_file=log_file,
                )
                states[job.name] = {
                    "job": job,
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "global_step": global_step,
                    "state_path": state_path,
                    "base_lr": learning_rate,
                    "output_model": output_model,
                    "metrics_file": metrics_file,
                    "log_file": log_file,
                    "fieldnames": fieldnames,
                    "stats": _empty_epoch_stats(),
                    "trained_samples": 0,
                    "started": time.perf_counter(),
                }
            elif kind == "batch":
                wave = [message]
                wave_jobs = {str(message[2])}
                while len(wave) < device_wave_size:
                    try:
                        candidate = message_queue.get_nowait()
                    except queue.Empty:
                        break
                    if (
                        candidate[0] == "batch"
                        and str(candidate[2]) not in wave_jobs
                    ):
                        wave.append(candidate)
                        wave_jobs.add(str(candidate[2]))
                    else:
                        pending_messages.append(candidate)
                process_batch_wave(wave)
            elif kind == "epoch_done":
                _kind, _worker_id, job_name, epoch = message
                state = states[job_name]
                stats = state["stats"]
                batches = int(stats["batches"])
                if batches:
                    values = torch.stack(
                        (
                            stats["loss"],
                            stats["loss_value"],
                            stats["loss_policy"],
                            stats["correct"].to(dtype=torch.float32),
                        )
                    ).cpu().tolist()
                    loss, loss_value, loss_policy = (
                        value / batches for value in values[:3]
                    )
                    accuracy = values[3] / max(int(stats["total_seen"]), 1)
                else:
                    loss = loss_value = loss_policy = accuracy = 0.0
                elapsed = time.perf_counter() - float(state["started"])
                current_lr = float(
                    state["optimizer"].param_groups[0]["lr"]
                )
                row = {
                    "epoch": epoch,
                    "batches": batches,
                    "loss": loss,
                    "loss_value": loss_value,
                    "loss_policy": loss_policy,
                    "train_accuracy": accuracy,
                    "val_accuracy": 0.0,
                    "learning_rate": current_lr,
                    "global_step": int(state["global_step"]),
                    "elapsed_seconds": elapsed,
                }
                with state["metrics_file"].open(
                    "a", newline="", encoding="utf-8"
                ) as file:
                    csv.DictWriter(
                        file,
                        fieldnames=state["fieldnames"],
                    ).writerow(row)
                with state["log_file"].open("a", encoding="utf-8") as file:
                    file.write(
                        f"epoch={epoch} loss={loss:.4f} "
                        f"loss_value={loss_value:.4f} "
                        f"loss_policy={loss_policy:.4f} "
                        f"train_acc={accuracy:.3f} "
                        f"lr={current_lr:.8g} "
                        f"global_step={int(state['global_step'])} "
                        f"elapsed={elapsed:.1f}s\n"
                    )
                state["trained_samples"] += int(stats["total_seen"])
                state["stats"] = _empty_epoch_stats()
            elif kind == "job_done":
                _kind, _worker_id, job_name = message
                state = states.pop(job_name)
                job = state["job"]
                model = state["model"]
                model_bytes = sum(
                    parameter.numel() * parameter.element_size()
                    for parameter in model.parameters()
                )
                if keep_models:
                    torch.save(model.state_dict(), state["output_model"])
                # 重み保存後にoptimizer/schedulerを保存する。次の更新は
                # このstateから続き、lrは全更新を通して減衰し続ける。
                _save_training_state(torch, state)
                elapsed = time.perf_counter() - float(state["started"])
                results.append(
                    {
                        "job": job.name,
                        "elapsedSeconds": elapsed,
                        "modelBytes": model_bytes,
                        "metricsFile": str(state["metrics_file"]),
                        "logFile": str(state["log_file"]),
                        "workerPid": os.getpid(),
                        "samples": job.samples,
                        "trainedSamples": int(state["trained_samples"]),
                    }
                )
                print(
                    f"[{len(results)}/{len(jobs)}] {job.name}: {elapsed:.2f}s",
                    flush=True,
                )
                del model
                del state
            elif kind == "worker_done":
                completed_workers += 1
            elif kind == "worker_error":
                _kind, worker_id, error = message
                raise RuntimeError(f"CPU loader worker {worker_id}が失敗しました:\n{error}")
            else:
                raise RuntimeError(f"不明なCPU loader messageです: {kind}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        message_queue.close()
    return results, str(device)


def _thread_environment(threads_per_worker: int) -> dict[str, str]:
    """PyTorch/BLASがワーカー数以上にCPUを取り合わない環境を返す。"""
    value = str(threads_per_worker)
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "VECLIB_MAXIMUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def _initialize_worker(threads_per_worker: int) -> None:
    os.environ.update(_thread_environment(threads_per_worker))
    os.environ["PYTHONIOENCODING"] = "utf-8"


def _job_paths(
    job: TrainingJob,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    job_dir = output_dir / "jobs" / job.name
    job_dir.mkdir(parents=True, exist_ok=False)
    return (
        job_dir / "model.pth",
        job_dir / "metrics.csv",
        job_dir / "train.log",
        job_dir,
    )


def _training_arguments(
    job: TrainingJob,
    *,
    train_script: Path,
    output_model: Path,
    metrics_file: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> list[str]:
    return [
        str(train_script),
        "--shards",
        str(job.shards),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--device",
        device,
        "--output-model",
        str(output_model),
        "--metrics-file",
        str(metrics_file),
        "--initial-model",
        str(job.initial_model),
        "--seed",
        str(job.seed),
        "--val-shards",
        "0",
    ]


def _finish_job(
    job: TrainingJob,
    *,
    output_model: Path,
    metrics_file: Path,
    log_file: Path,
    elapsed: float,
    keep_models: bool,
    worker_pid: int | None,
    batch_size: int,
) -> dict[str, object]:
    if not output_model.is_file():
        raise RuntimeError(f"{job.name}の出力モデルがありません: {output_model}")
    model_bytes = output_model.stat().st_size
    if not keep_models:
        output_model.unlink()
    with metrics_file.open(newline="", encoding="utf-8") as metrics:
        trained_samples = sum(
            int(row["batches"]) * batch_size
            for row in csv.DictReader(metrics)
        )
    return {
        "job": job.name,
        "elapsedSeconds": elapsed,
        "modelBytes": model_bytes,
        "metricsFile": str(metrics_file),
        "logFile": str(log_file),
        "workerPid": worker_pid,
        "samples": job.samples,
        "trainedSamples": trained_samples,
    }


def _run_job_persistent(
    job: TrainingJob,
    *,
    train_script: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    keep_models: bool,
) -> dict[str, object]:
    """常駐process内で学習し、torch/MPS初期化をworkerごとに1回にする。"""
    output_model, metrics_file, log_file, _job_dir = _job_paths(
        job,
        output_dir,
    )
    arguments = _training_arguments(
        job,
        train_script=train_script,
        output_model=output_model,
        metrics_file=metrics_file,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    previous_argv = sys.argv
    previous_cwd = Path.cwd()
    started = time.perf_counter()
    try:
        sys.argv = arguments
        os.chdir(train_script.resolve().parents[2])
        with log_file.open("w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                runpy.run_path(str(train_script), run_name="__main__")
    except (Exception, SystemExit) as error:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            f"{job.name}が失敗しました: {type(error).__name__}: {error}\n{tail}"
        ) from error
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
    return _finish_job(
        job,
        output_model=output_model,
        metrics_file=metrics_file,
        log_file=log_file,
        elapsed=time.perf_counter() - started,
        keep_models=keep_models,
        worker_pid=os.getpid(),
        batch_size=batch_size,
    )


def _run_job_subprocess(
    job: TrainingJob,
    *,
    python: Path,
    train_script: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    keep_models: bool,
    threads_per_worker: int,
) -> dict[str, object]:
    output_model, metrics_file, log_file, _job_dir = _job_paths(
        job,
        output_dir,
    )
    command = [
        str(python),
        *_training_arguments(
            job,
            train_script=train_script,
            output_model=output_model,
            metrics_file=metrics_file,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        ),
    ]
    environment = os.environ.copy()
    environment.update(_thread_environment(threads_per_worker))
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=train_script.resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        tail = log_file.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(
            f"{job.name}がexit={completed.returncode}で失敗しました。\n{tail}"
        )
    return _finish_job(
        job,
        output_model=output_model,
        metrics_file=metrics_file,
        log_file=log_file,
        elapsed=elapsed,
        keep_models=keep_models,
        worker_pid=None,
        batch_size=batch_size,
    )


def _absolute_path_without_resolving_symlinks(path: Path) -> Path:
    """venvのpython symlinkを保った絶対パスを返す。"""
    return Path(os.path.abspath(os.path.expanduser(path)))


def _effective_threads_per_worker(workers: int, requested: int) -> int:
    if requested:
        return requested
    available = os.cpu_count() or 1
    return max(1, min(4, available // workers))


def _same_executable(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def benchmark_parallel_training(
    *,
    python: Path,
    train_script: Path,
    shards_root: Path,
    agents_root: Path,
    output_dir: Path,
    workers: int,
    job_limit: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    seed: int,
    keep_models: bool = False,
    persistent_workers: bool = True,
    threads_per_worker: int = 0,
    schedule_order: str = "largest-first",
    central_device_owner: bool = False,
    device_wave_size: int = 3,
) -> dict[str, object]:
    """同じshard/model初期値のジョブを最大``workers``本ずつ実行する。"""
    if workers < 1:
        raise ValueError("workersは1以上にしてください。")
    if job_limit < 0:
        raise ValueError("job_limitは0以上にしてください。")
    if epochs < 1:
        raise ValueError("epochsは1以上にしてください。")
    if batch_size < 1:
        raise ValueError("batch_sizeは1以上にしてください。")
    if threads_per_worker < 0:
        raise ValueError("threads_per_workerは0以上にしてください。")
    if device_wave_size < 1:
        raise ValueError("device_wave_sizeは1以上にしてください。")
    if schedule_order not in ("discovery", "largest-first"):
        raise ValueError(
            "schedule_orderはdiscovery/largest-firstにしてください。"
        )
    python = _absolute_path_without_resolving_symlinks(Path(python))
    train_script = Path(train_script).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Pythonがありません: {python}")
    if not train_script.is_file():
        raise FileNotFoundError(f"学習スクリプトがありません: {train_script}")

    jobs = discover_training_jobs(shards_root, agents_root, seed=seed)
    if job_limit:
        jobs = jobs[:job_limit]
    jobs = schedule_training_jobs(jobs, schedule_order)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    max_workers = min(workers, len(jobs))
    effective_threads = _effective_threads_per_worker(
        max_workers,
        threads_per_worker,
    )
    current_python = _absolute_path_without_resolving_symlinks(
        Path(sys.executable),
    )
    if central_device_owner and not _same_executable(python, current_python):
        raise ValueError(
            "central_device_ownerではrunnerと--pythonを同じ環境にしてください。"
        )
    use_persistent_workers = persistent_workers and _same_executable(
        python,
        current_python,
    )
    if central_device_owner:
        execution_mode = "central-device"
        effective_threads = 1
    else:
        execution_mode = "persistent" if use_persistent_workers else "subprocess"

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    resolved_device = device
    if central_device_owner:
        results, resolved_device = _run_central_device_training(
            jobs,
            train_script=train_script,
            output_dir=output_dir,
            workers=max_workers,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device_name=device,
            keep_models=keep_models,
            device_wave_size=min(device_wave_size, max_workers),
        )
    elif use_persistent_workers:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(effective_threads,),
        )
        job_function = _run_job_persistent
        shared_arguments = {
            "train_script": train_script,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "keep_models": keep_models,
        }
    elif not central_device_owner:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        job_function = _run_job_subprocess
        shared_arguments = {
            "python": python,
            "train_script": train_script,
            "output_dir": output_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "device": device,
            "keep_models": keep_models,
            "threads_per_worker": effective_threads,
        }
    if not central_device_owner:
        with executor:
            futures = {
                executor.submit(
                    job_function,
                    job,
                    **shared_arguments,
                ): job
                for job in jobs
            }
            for completed_count, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                result = future.result()
                results.append(result)
                print(
                    f"[{completed_count}/{len(jobs)}] {result['job']}: "
                    f"{result['elapsedSeconds']:.2f}s",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: str(item["job"]))
    summary = {
        "workers": workers,
        "jobs": len(jobs),
        "sourceGames": max((job.source_games for job in jobs), default=0),
        "manifestSamples": sum(job.samples for job in jobs),
        "trainedSamples": sum(
            int(result["trainedSamples"]) for result in results
        ),
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "device": resolved_device,
        "executionMode": execution_mode,
        "threadsPerWorker": effective_threads,
        "scheduleOrder": schedule_order,
        "deviceWaveSize": (
            min(device_wave_size, max_workers) if central_device_owner else None
        ),
        "elapsedSeconds": elapsed,
        "jobsPerSecond": len(jobs) / elapsed,
        "keepModels": keep_models,
        "shardsRoot": str(Path(shards_root).resolve()),
        "agentsRoot": str(Path(agents_root).resolve()),
        "jobResults": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"BENCHMARK_SUMMARY={summary_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--agents-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--job-limit",
        type=int,
        default=8,
        help="先頭から測るジョブ数。0は全32ジョブ",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ワーカー内でtorch/MPSを再利用する（既定: 有効）",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="ワーカーごとのCPUスレッド数。0は自動",
    )
    parser.add_argument(
        "--schedule-order",
        choices=("discovery", "largest-first"),
        default="largest-first",
        help="ジョブ投入順。largest-firstは重いshardを先に投入する",
    )
    parser.add_argument(
        "--central-device-owner",
        action="store_true",
        help=(
            "CPU workerは前処理だけを行い、torch/deviceは中央processだけで"
            "所有する"
        ),
    )
    parser.add_argument(
        "--device-wave-size",
        type=int,
        default=3,
        help="central-device-ownerで先にMPSへ投入する異なるmodel batch数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_parallel_training(
        python=args.python,
        train_script=args.train_script,
        shards_root=args.shards_root,
        agents_root=args.agents_root,
        output_dir=args.output_dir,
        workers=args.workers,
        job_limit=args.job_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        keep_models=args.keep_models,
        persistent_workers=args.persistent_workers,
        threads_per_worker=args.threads_per_worker,
        schedule_order=args.schedule_order,
        central_device_owner=args.central_device_owner,
        device_wave_size=args.device_wave_size,
    )


if __name__ == "__main__":
    main()
