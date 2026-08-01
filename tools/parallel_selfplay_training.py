"""計測処理を持たない、並列対戦とモデル更新の呼び出し関数。

1回の学習を呼ぶまでに生成する対戦数は ``workers * lanes_per_worker``、
``train_minibatch_size`` は学習処理内部のミニバッチサイズとして別々に扱う。
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping


def _pairings(
    agent_names: list[str],
    include_self: bool,
) -> list[tuple[str, str]]:
    if include_self:
        return list(itertools.combinations_with_replacement(agent_names, 2))
    return list(itertools.combinations(agent_names, 2))


def _balanced_games_per_pairing(
    pairings: list[tuple[str, str]],
    total_games: int,
) -> list[int]:
    """総試合数をカード間差1以内かつagent別試合数が均等になるよう配る。"""
    if not pairings:
        raise ValueError("対戦カードがありません。")
    if total_games < 1:
        raise ValueError("総試合数は1以上にしてください。")

    names = list(dict.fromkeys(name for pairing in pairings for name in pairing))
    name_to_index = {name: index for index, name in enumerate(names)}
    base_games, extra_games = divmod(total_games, len(pairings))
    counts = [base_games] * len(pairings)

    # self戦は同じagentが両側に立つため、学習データ上の登場数は2として扱う。
    appearances = [0] * len(names)
    standings = [0] * len(names)
    for (name0, name1), count in zip(pairings, counts, strict=True):
        index0 = name_to_index[name0]
        index1 = name_to_index[name1]
        if index0 == index1:
            appearances[index0] += count * 2
            standings[index0] += count
        else:
            appearances[index0] += count
            appearances[index1] += count
            standings[index0] += count
            standings[index1] += count

    available = set(range(len(pairings)))
    for _ in range(extra_games):
        def score(pairing_index: int) -> tuple[int, int, int, int, int]:
            candidate_appearances = appearances.copy()
            candidate_standings = standings.copy()
            name0, name1 = pairings[pairing_index]
            index0 = name_to_index[name0]
            index1 = name_to_index[name1]
            if index0 == index1:
                candidate_appearances[index0] += 2
                candidate_standings[index0] += 1
            else:
                candidate_appearances[index0] += 1
                candidate_appearances[index1] += 1
                candidate_standings[index0] += 1
                candidate_standings[index1] += 1
            return (
                max(candidate_appearances) - min(candidate_appearances),
                max(candidate_standings) - min(candidate_standings),
                sum(value * value for value in candidate_appearances),
                sum(value * value for value in candidate_standings),
                pairing_index,
            )

        selected = min(available, key=score)
        available.remove(selected)
        counts[selected] += 1
        name0, name1 = pairings[selected]
        index0 = name_to_index[name0]
        index1 = name_to_index[name1]
        if index0 == index1:
            appearances[index0] += 2
            standings[index0] += 1
        else:
            appearances[index0] += 1
            appearances[index1] += 1
            standings[index0] += 1
            standings[index1] += 1
    return counts


def _safe_training_name(value: str) -> str:
    visible = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    )[:40]
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{visible or 'agent'}_{digest}"


def _completed_pairing_games(
    episodes_dir: Path,
    pairings: list[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    """保存済みepisodeからカード別の完了数と最大game indexを返す。"""
    completed: list[int] = []
    max_indices: list[int] = []
    index_pattern = re.compile(r"_g(\d+)_s[01]\.(?:json|pkl)$")
    for name0, name1 in pairings:
        matchup = hashlib.sha1(f"{name0}\0{name1}".encode("utf-8")).hexdigest()[:8]
        prefix = (
            f"episode_{_safe_training_name(name0)}_vs_"
            f"{_safe_training_name(name1)}_{matchup}_g"
        )
        paths = [
            path
            for extension in ("json", "pkl")
            for path in episodes_dir.glob(f"{prefix}*_s*.{extension}")
        ]
        indices: set[int] = set()
        for path in paths:
            match = index_pattern.search(path.name)
            if match is not None:
                indices.add(int(match.group(1)))
        completed.append(len(indices))
        max_indices.append(max(indices, default=0))
    return completed, max_indices


def run_parallel_games(
    *,
    python: Path,
    runner: Path,
    match_root: Path,
    agents: Mapping[str, str | Path],
    episodes_dir: Path,
    workers: int,
    lanes_per_worker: int,
    include_self: bool,
    match_batch_size: int,
    search_count: int,
    device: str,
    seed: int,
    max_turns: int = 80,
    max_selections: int = 500,
    max_attempts: int = 5,
) -> int:
    """並列対戦を1更新単位だけ回し、生成された学習episode数を返す。"""
    if workers < 1:
        raise ValueError("workersは1以上で指定してください。")
    if lanes_per_worker < 1:
        raise ValueError("lanes_per_workerは1以上で指定してください。")
    if max_turns < 1:
        raise ValueError("max_turnsは1以上で指定してください。")
    if max_selections < 1:
        raise ValueError("max_selectionsは1以上で指定してください。")
    if max_attempts < 1:
        raise ValueError("max_attemptsは1以上で指定してください。")
    games_per_update = workers * lanes_per_worker

    pairings = _pairings(list(agents), include_self)
    pair_count = len(pairings)
    if pair_count < 1:
        raise ValueError("総当たりには必要な数のagentがありません。")
    target_counts = _balanced_games_per_pairing(
        pairings,
        games_per_update,
    )
    scheduled_matches = {name: 0 for name in agents}
    scheduled_player_sides = {name: 0 for name in agents}
    for (name0, name1), count in zip(pairings, target_counts, strict=True):
        scheduled_matches[name0] += count
        scheduled_player_sides[name0] += count
        if name0 == name1:
            scheduled_player_sides[name0] += count
        else:
            scheduled_matches[name1] += count
            scheduled_player_sides[name1] += count

    command_prefix = [str(python), str(runner)]
    for name, relative_main in agents.items():
        main_path = Path(relative_main)
        if not main_path.is_absolute():
            main_path = match_root / main_path
        command_prefix.extend(["--agent", f"{name}={main_path}"])
    command_prefix.extend(
        [
            "--backend",
            "worker-batched",
            "--device",
            device,
            "--workers",
            str(workers),
            "--lanes",
            str(games_per_update),
            "--batch-size",
            str(match_batch_size),
            "--search-count",
            str(search_count),
            "--max-turns",
            str(max_turns),
            "--max-selections",
            str(max_selections),
            "--quiet",
            "--training-json-dir",
            str(episodes_dir),
            "--training-format",
            "preencoded",
        ]
    )
    if not include_self:
        command_prefix.append("--no-self")

    print(
        f"parallel games: workers={workers}, lanes/worker={lanes_per_worker}, "
        f"games={games_per_update}, pairings={pair_count}, "
        f"self={'yes' if include_self else 'no'}, "
        f"limits={max_turns} turns/{max_selections} selections"
    )
    print(
        "balanced schedule: "
        f"matches/agent={min(scheduled_matches.values())}〜"
        f"{max(scheduled_matches.values())}, "
        f"player-sides/agent={min(scheduled_player_sides.values())}〜"
        f"{max(scheduled_player_sides.values())}"
    )
    completed_counts, max_indices = _completed_pairing_games(episodes_dir, pairings)
    if any(
        completed > target
        for completed, target in zip(completed_counts, target_counts, strict=True)
    ):
        raise RuntimeError(
            "出力先に今回の対戦割当を超える学習episodeがあります: "
            f"{episodes_dir}"
        )
    episode_count = sum(completed_counts)
    attempt = 0
    while episode_count < games_per_update:
        missing_counts = [
            target - completed
            for target, completed in zip(
                target_counts,
                completed_counts,
                strict=True,
            )
        ]
        command = command_prefix + [
            "--games-per-pairing-json",
            json.dumps(missing_counts, separators=(",", ":")),
            "--game-index-offsets-json",
            json.dumps(max_indices, separators=(",", ":")),
            "--seed",
            str(seed + attempt),
        ]
        if episode_count:
            command.append("--allow-existing-training-json")
        subprocess.run(
            command,
            check=True,
            cwd=match_root,
            stdout=subprocess.DEVNULL,
        )

        updated_completed, updated_max_indices = _completed_pairing_games(
            episodes_dir,
            pairings,
        )
        updated_count = sum(updated_completed)
        if updated_count <= episode_count:
            missing_labels = [
                f"{name0} vs {name1}: {target - completed}"
                for (name0, name1), target, completed in zip(
                    pairings,
                    target_counts,
                    updated_completed,
                    strict=True,
                )
                if completed < target
            ]
            raise RuntimeError(
                "追加対戦で完了した学習episodeが増えませんでした: "
                f"actual={updated_count}, missing={missing_labels}, "
                f"dir={episodes_dir}"
            )
        completed_counts = updated_completed
        max_indices = updated_max_indices
        episode_count = updated_count
        attempt += 1
        if episode_count < games_per_update:
            print(
                f"completed episodes: {episode_count}/{games_per_update}; "
                f"retry missing={games_per_update - episode_count}"
            )
            if attempt >= max_attempts:
                missing_labels = [
                    f"{name0} vs {name1}: {target - completed}"
                    for (name0, name1), target, completed in zip(
                        pairings,
                        target_counts,
                        completed_counts,
                        strict=True,
                    )
                    if completed < target
                ]
                raise RuntimeError(
                    f"対戦再試行が{max_attempts}回に達しました: "
                    f"missing={missing_labels}"
                )
    return episode_count


def read_parallel_training_metrics(
    training_output_dir: Path,
) -> list[dict[str, object]]:
    """並列学習が追記したepoch metricsを、書き込み途中に強く読み取る。"""
    rows: list[dict[str, object]] = []
    for metrics_file in sorted(
        Path(training_output_dir).glob("jobs/*/metrics.csv")
    ):
        try:
            with metrics_file.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    try:
                        rows.append(
                            {
                                "job": metrics_file.parent.name,
                                "epoch": int(row["epoch"]),
                                "batches": int(row["batches"]),
                                "loss": float(row["loss"]),
                                "loss_value": float(row["loss_value"]),
                                "loss_policy": float(row["loss_policy"]),
                                "train_accuracy": float(row["train_accuracy"]),
                                "val_accuracy": float(row["val_accuracy"]),
                                "elapsed_seconds": float(row["elapsed_seconds"]),
                            }
                        )
                    except (KeyError, TypeError, ValueError):
                        # CSV末尾が追記途中なら、次回のpollで完成行を読み直す。
                        continue
        except (FileNotFoundError, OSError):
            # job開始・終了とglob/openが競合しても監視は継続する。
            continue
    return rows


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _resolved_agent_sources(
    agents: Mapping[str, str | Path],
    match_root: Path,
) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for name, relative_main in agents.items():
        main_path = Path(relative_main)
        if not main_path.is_absolute():
            main_path = match_root / main_path
        source_dir = main_path.resolve().parent
        for filename in ("deck.csv", "model.pth"):
            required = source_dir / filename
            if not required.is_file():
                raise FileNotFoundError(f"agent学習入力がありません: {required}")
        sources[name] = source_dir
    return sources


def _agents_root(agent_sources: Mapping[str, Path]) -> Path:
    roots = {source.parent.parent.resolve() for source in agent_sources.values()}
    if len(roots) != 1:
        raise ValueError(
            "agentが同じモデルルートにないため並列学習できません: "
            f"{sorted(map(str, roots))}"
        )
    return next(iter(roots))


def _wait_for_parallel_training(
    command: list[str],
    *,
    cwd: Path,
    training_output_dir: Path,
    progress_callback: Callable[[list[dict[str, object]]], None] | None,
    poll_interval_seconds: float,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(command, cwd=cwd, env=environment)
    last_signature: tuple[tuple[object, ...], ...] | None = None
    try:
        while True:
            return_code = process.poll()
            if progress_callback is not None:
                metrics = read_parallel_training_metrics(training_output_dir)
                signature = tuple(
                    (
                        row["job"],
                        row["epoch"],
                        row["loss"],
                        row["loss_value"],
                        row["loss_policy"],
                        row["train_accuracy"],
                    )
                    for row in metrics
                )
                if signature != last_signature:
                    progress_callback(metrics)
                    last_signature = signature
            if return_code is not None:
                break
            time.sleep(poll_interval_seconds)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def train_models(
    *,
    python: Path,
    preprocessor: Path,
    train_script: Path,
    parallel_trainer: Path,
    train_root: Path,
    match_root: Path,
    agents: Mapping[str, str | Path],
    episodes_dir: Path,
    work_dir: Path,
    checkpoint_dir: Path | None = None,
    epochs: int,
    train_minibatch_size: int,
    learning_rate: float,
    shard_size: int,
    device: str,
    seed: int,
    training_workers: int,
    device_wave_size: int,
    progress_callback: Callable[[list[dict[str, object]]], None] | None = None,
    poll_interval_seconds: float = 0.5,
) -> dict[str, object]:
    """前処理後、ベンチマーク済み中央device方式で全モデルを並列学習する。"""
    if train_minibatch_size < 1:
        raise ValueError("train_minibatch_sizeは1以上で指定してください。")
    if training_workers < 1:
        raise ValueError("training_workersは1以上で指定してください。")
    if device_wave_size < 1:
        raise ValueError("device_wave_sizeは1以上で指定してください。")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_secondsは0より大きくしてください。")

    python = Path(python)
    preprocessor = Path(preprocessor).resolve()
    train_script = Path(train_script).resolve()
    parallel_trainer = Path(parallel_trainer).resolve()
    train_root = Path(train_root).resolve()
    match_root = Path(match_root).resolve()
    work_dir = Path(work_dir).resolve()
    episodes_dir = Path(episodes_dir).resolve()
    for required in (python, preprocessor, train_script, parallel_trainer):
        if not required.is_file():
            raise FileNotFoundError(f"並列学習に必要なファイルがありません: {required}")
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"学習work-dirが空ではありません: {work_dir}")

    agent_sources = _resolved_agent_sources(agents, match_root)
    agents_root = _agents_root(agent_sources)
    if checkpoint_dir is None:
        episode_name = work_dir.parent.name
        if not episode_name.startswith("episode"):
            raise ValueError(
                "work-dirから累計episode数を判定できないため"
                "checkpoint_dirを指定してください。"
            )
        checkpoint_dir = agents_root / f"model_{episode_name}"
    checkpoint_dir = Path(checkpoint_dir).resolve()
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"学習済みcheckpointの保存先が既に存在します: {checkpoint_dir}"
        )

    started = time.perf_counter()
    shards_dir = work_dir / "shards"
    preprocess_command = [
        str(python),
        str(preprocessor),
        "--episodes",
        str(episodes_dir),
        "--output-dir",
        str(shards_dir),
        "--shard-size",
        str(shard_size),
    ]
    for name, source_dir in agent_sources.items():
        preprocess_command.extend(
            ["--agent", f"{name}={source_dir / 'deck.csv'}"]
        )

    print(
        f"preprocess training episodes: episodes={episodes_dir}, "
        f"shard_size={shard_size}"
    )
    preprocess_started = time.perf_counter()
    subprocess.run(preprocess_command, check=True, cwd=train_root)
    preprocess_seconds = time.perf_counter() - preprocess_started

    training_output_dir = work_dir / "parallel_training"
    training_command = [
        str(python),
        str(parallel_trainer),
        "--python",
        str(python),
        "--train-script",
        str(train_script),
        "--shards-root",
        str(shards_dir),
        "--agents-root",
        str(agents_root),
        "--output-dir",
        str(training_output_dir),
        "--workers",
        str(training_workers),
        "--job-limit",
        "0",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(train_minibatch_size),
        "--lr",
        str(learning_rate),
        "--device",
        device,
        "--seed",
        str(seed),
        "--threads-per-worker",
        "0",
        "--schedule-order",
        "largest-first",
        "--device-wave-size",
        str(device_wave_size),
        "--central-device-owner",
        "--no-persistent-workers",
        "--keep-models",
    ]
    print(
        f"parallel train models: jobs={len(agent_sources) * 2}, "
        f"workers={training_workers}, wave={device_wave_size}, "
        f"minibatch={train_minibatch_size}, epochs={epochs}"
    )
    training_started = time.perf_counter()
    _wait_for_parallel_training(
        training_command,
        cwd=match_root,
        training_output_dir=training_output_dir,
        progress_callback=progress_callback,
        poll_interval_seconds=poll_interval_seconds,
    )
    training_seconds = time.perf_counter() - training_started

    parallel_summary_path = training_output_dir / "summary.json"
    parallel_summary = json.loads(
        parallel_summary_path.read_text(encoding="utf-8")
    )
    expected_jobs = len(agent_sources) * 2
    if int(parallel_summary.get("jobs", -1)) != expected_jobs:
        raise RuntimeError(
            "並列学習の完了job数が一致しません: "
            f"expected={expected_jobs}, actual={parallel_summary.get('jobs')}"
        )

    staging_dir = work_dir / ".checkpoint_staging"
    staging_dir.mkdir(parents=True, exist_ok=False)
    for name in agent_sources:
        for role, filename in (
            ("self", "model.pth"),
            ("opponent", "opponent_model.pth"),
        ):
            trained_model = (
                training_output_dir / "jobs" / f"{name}_{role}" / "model.pth"
            )
            if not trained_model.is_file():
                raise RuntimeError(
                    f"並列学習済みモデルがありません: {trained_model}"
                )
            destination = staging_dir / name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trained_model, destination)

    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir, checkpoint_dir)
    for name, source_dir in agent_sources.items():
        _copy_atomic(checkpoint_dir / name / "model.pth", source_dir / "model.pth")
        _copy_atomic(
            checkpoint_dir / name / "opponent_model.pth",
            source_dir / "opponent_model.pth",
        )

    summary: dict[str, object] = {
        "episodes": [str(episodes_dir)],
        "agents": list(agent_sources),
        "preprocessSeconds": preprocess_seconds,
        "trainingSeconds": training_seconds,
        "elapsedSeconds": time.perf_counter() - started,
        "epochs": epochs,
        "batchSize": train_minibatch_size,
        "learningRate": learning_rate,
        "device": str(parallel_summary.get("device", device)),
        "trainingWorkers": training_workers,
        "deviceWaveSize": device_wave_size,
        "executionMode": parallel_summary.get("executionMode"),
        "checkpointDir": str(checkpoint_dir),
        "parallelTrainingSummary": str(parallel_summary_path),
    }
    (work_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "episode checkpoint training done: "
        f"preprocess={preprocess_seconds:.1f}s "
        f"training={training_seconds:.1f}s "
        f"total={summary['elapsedSeconds']:.1f}s"
    )
    return summary
