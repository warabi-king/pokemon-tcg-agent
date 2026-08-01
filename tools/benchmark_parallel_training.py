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
        ) = sample
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

    return (
        np.asarray(encoder_index, dtype=np.int32),
        np.asarray(encoder_value, dtype=np.float32),
        np.asarray(encoder_offset, dtype=np.int32),
        np.asarray(decoder_index, dtype=np.int32),
        np.asarray(decoder_value, dtype=np.float32),
        np.asarray(decoder_offset, dtype=np.int32),
        mask,
        np.asarray(label_value, dtype=np.float32).reshape(-1, 1),
        np.asarray(chosen_indices, dtype=np.int64),
    )


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
    loss_fn_value = torch.nn.HuberLoss(delta=0.2)

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
            mask_tensor, label_value_tensor, chosen_index_tensor = tensors[6:]
            loss_value = loss_fn_value(out_enc, label_value_tensor)
            masked_logits = out_dec.masked_fill(~mask_tensor, float("-inf"))
            loss_policy = functional.cross_entropy(
                masked_logits,
                chosen_index_tensor,
            )
            loss = loss_value + loss_policy
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
                states[job.name] = {
                    "job": job,
                    "model": model,
                    "optimizer": torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rate,
                    ),
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
                row = {
                    "epoch": epoch,
                    "batches": batches,
                    "loss": loss,
                    "loss_value": loss_value,
                    "loss_policy": loss_policy,
                    "train_accuracy": accuracy,
                    "val_accuracy": 0.0,
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
                        f"train_acc={accuracy:.3f} elapsed={elapsed:.1f}s\n"
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
