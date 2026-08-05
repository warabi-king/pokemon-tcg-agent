"""固定デッキで自己対戦と凍結belief_puctを混ぜてRL学習する。

実行例:
    python agents/belief_puct/train/train_mixed_opponents.py \
        --deck agents/belief_puct/train/runs/phase2a-001/candidates/candidate_004/deck.csv \
        --initial-model agents/belief_puct/src/model.pth \
        --run-dir agents/belief_puct/train/runs/phase2b-001 \
        --iterations 10 --games-per-iteration 40 --frozen-opponent-fraction 0.5

各iterationの完了時に ``training_state_latest.pth`` を保存する。中断後は
``--resume <run-dir>/training_state_latest.pth`` を指定して再開できる。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl_mcts.mcts import LearnSample, mcts_agent  # noqa: E402
from rl_mcts.model import MyModel, create_model  # noqa: E402
from train import TrainStats, raise_for_deck_error, train_one_iteration  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> MyModel:
    """raw state_dictまたは本スクリプトの学習stateからモデルを復元する。"""

    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state") if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError(f"model state_dictを取得できません: {checkpoint_path}")
    model = create_model().to(device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            "checkpointと現在のcreate_model()の構造が一致しません: "
            f"{checkpoint_path}"
        ) from error
    return model


def read_fixed_deck(deck_path: Path) -> list[int]:
    """Phase 2aで選ばれた指定パスの60枚deck.csvを読み、枚数を検証する。"""

    deck = [int(line.strip()) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csvは60枚である必要があります: {deck_path} ({len(deck)}枚)")
    return deck


def sha256_file(path: Path) -> str:
    """入力deck・checkpointの同一性をrun設定へ残すSHA-256を計算する。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_terminal_values(samples: list[LearnSample], result: int, learner_index: int, lambda_value: float) -> None:
    """学習側プレイヤーの終局結果を、収集順のsampleへ逆伝播ラベルとして設定する。"""

    if result == 2:
        value = 0.0
    else:
        value = 1.0 if result == learner_index else -1.0
    for sample in reversed(samples):
        label = (value + sample.value) * 0.5
        value = value * lambda_value + sample.value * (1.0 - lambda_value)
        sample.value = label


def play_training_game(
    learner_model: MyModel,
    frozen_model: MyModel | None,
    deck: list[int],
    search_count: int,
    lambda_value: float,
    learner_index: int,
) -> tuple[list[LearnSample], str]:
    """1局行い、学習に使うsampleと対戦種別ごとの結果を返す。

    ``frozen_model`` がNoneなら両者が同一learnerで、そうでなければ指定playerだけが
    learnerとなる。自己対戦は両者のsampleを使い、凍結相手の手はsampleに含めないため、
    過去方策を教師へ混ぜない。
    """

    obs, start_data = battle_start(deck, deck)
    raise_for_deck_error(start_data)
    samples_by_player: list[list[LearnSample]] = [[], []]
    try:
        while obs["current"]["result"] < 0:
            current_index = int(obs["current"]["yourIndex"])
            active_model = learner_model if frozen_model is None or current_index == learner_index else frozen_model
            selected, sample = mcts_agent(obs, deck, active_model, search_count=search_count)
            if sample is not None and (frozen_model is None or current_index == learner_index):
                samples_by_player[current_index].append(sample)
            obs = battle_select(selected)
    finally:
        battle_finish()

    result = int(obs["current"]["result"])
    if frozen_model is None:
        # 自己対戦では両者とも学習中モデルなので、双方の局面を学習に使う。
        assign_terminal_values(samples_by_player[0], result, 0, lambda_value)
        assign_terminal_values(samples_by_player[1], result, 1, lambda_value)
        return samples_by_player[0] + samples_by_player[1], "play"

    learner_samples = samples_by_player[learner_index]
    assign_terminal_values(learner_samples, result, learner_index, lambda_value)
    if result == 2:
        return learner_samples, "draw"
    return learner_samples, "win" if result == learner_index else "loss"


def append_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    """iterationごとの学習・相手内訳・checkpointをCSVへ追記する。"""

    fieldnames = list(row)
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl(log_path: Path, event: dict[str, object]) -> None:
    """再開時にも追記できるイベントログをJSON Lines形式で記録する。"""

    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def save_training_state(
    state_path: Path,
    model: MyModel,
    optimizer: torch.optim.Optimizer,
    completed_iterations: int,
    config: dict[str, object],
) -> None:
    """モデル・optimizer・乱数状態を次回iteration開始位置として保存する。"""

    torch.save(
        {
            "format": "pokemon-tcg-agent/mixed-opponent-rl-state-v1",
            "completed_iterations": completed_iterations,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "config": config,
        },
        state_path,
    )


def restore_training_state(
    state_path: Path,
    model: MyModel,
    optimizer: torch.optim.Optimizer,
) -> int:
    """保存済みstateを復元し、次に実行するiteration番号を返す。"""

    payload = torch.load(state_path, map_location=torch.device("cpu"))
    if not isinstance(payload, dict) or payload.get("format") != "pokemon-tcg-agent/mixed-opponent-rl-state-v1":
        raise ValueError(f"mixed-opponent RLの学習stateではありません: {state_path}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["completed_iterations"])


def parse_args() -> argparse.Namespace:
    """Phase 2bのCLI引数を読む。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True, help="Phase 2aで選択した60枚deck.csv")
    parser.add_argument("--initial-model", type=Path, required=True, help="Phase 2aで固定した開始checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True, help="checkpoint・ログの保存先")
    parser.add_argument("--iterations", type=int, default=10, help="この実行で到達する総iteration数")
    parser.add_argument("--games-per-iteration", type=int, default=40, help="1iterationの総対戦数")
    parser.add_argument("--frozen-opponent-fraction", type=float, default=0.5, help="凍結belief相手の比率")
    parser.add_argument("--frozen-model", type=Path, action="append", default=[], help="追加の凍結belief checkpoint。複数指定可")
    parser.add_argument("--search-count", type=int, default=10, help="各手番のMCTS探索回数")
    parser.add_argument("--batch-size", type=int, default=128, help="学習batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--lambda-value", type=float, default=0.9, help="終局価値の逆向き更新率")
    parser.add_argument("--seed", type=int, default=20260805, help="新規runの乱数seed")
    parser.add_argument("--resume", type=Path, default=None, help="training_state_latest.pthから再開する")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """実行前に復元不能な設定ミスを検出する。"""

    if args.iterations <= 0 or args.games_per_iteration <= 0 or args.search_count <= 0:
        raise ValueError("iterations、games-per-iteration、search-countは1以上で指定してください")
    if not 0.0 <= args.frozen_opponent_fraction <= 1.0:
        raise ValueError("frozen-opponent-fractionは0以上1以下で指定してください")
    for path in [args.deck, args.initial_model, *args.frozen_model]:
        if not path.exists():
            raise FileNotFoundError(f"入力ファイルがありません: {path}")


def main() -> None:
    """固定deck・混合相手でPhase 2bのRL学習を実行する。"""

    args = parse_args()
    validate_args(args)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    events_path = run_dir / "events.jsonl"
    state_path = run_dir / "training_state_latest.pth"
    frozen_dir = run_dir / "frozen_opponents"
    frozen_dir.mkdir(exist_ok=True)

    copied_initial = frozen_dir / "initial_model.pth"
    if not copied_initial.exists():
        shutil.copy2(args.initial_model, copied_initial)
    frozen_paths = [copied_initial, *(path.resolve() for path in args.frozen_model)]
    config: dict[str, object] = {
        "deck": str(args.deck.resolve()),
        "deck_sha256": sha256_file(args.deck.resolve()),
        "initial_model": str(args.initial_model.resolve()),
        "initial_model_sha256": sha256_file(args.initial_model.resolve()),
        "frozen_models": [str(path) for path in frozen_paths],
        "frozen_model_sha256": {str(path): sha256_file(path) for path in frozen_paths},
        "games_per_iteration": args.games_per_iteration,
        "frozen_opponent_fraction": args.frozen_opponent_fraction,
        "search_count": args.search_count,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_value": args.lambda_value,
    }
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    deck = read_fixed_deck(args.deck.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner_model = load_model(args.initial_model.resolve(), device)
    optimizer = torch.optim.AdamW(learner_model.parameters(), lr=args.lr)
    if args.resume is None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        start_iteration = 0
    else:
        start_iteration = restore_training_state(args.resume.resolve(), learner_model, optimizer)
        if start_iteration >= args.iterations:
            print(f"すでに目標iterationに到達しています: {start_iteration}/{args.iterations}")
            return

    frozen_models = [load_model(path, device).eval() for path in frozen_paths]
    write_jsonl(events_path, {"event": "run_started", "resume": args.resume is not None, "config": config})
    frozen_games = round(args.games_per_iteration * args.frozen_opponent_fraction)

    for iteration in range(start_iteration, args.iterations):
        started = time.time()
        opponent_kinds = ["frozen"] * frozen_games + ["self"] * (args.games_per_iteration - frozen_games)
        random.shuffle(opponent_kinds)
        samples: list[LearnSample] = []
        outcomes = {"self_play": 0, "frozen_win": 0, "frozen_loss": 0, "frozen_draw": 0}
        learner_model.eval()
        with torch.inference_mode():
            for game_index, opponent_kind in enumerate(opponent_kinds):
                frozen_model = random.choice(frozen_models) if opponent_kind == "frozen" else None
                learner_index = game_index % 2 if frozen_model is not None else 0
                game_samples, outcome = play_training_game(
                    learner_model, frozen_model, deck, args.search_count, args.lambda_value, learner_index
                )
                samples.extend(game_samples)
                outcomes[f"{opponent_kind}_{outcome}"] += 1

        stats: TrainStats = train_one_iteration(learner_model, optimizer, samples, args.batch_size, device)
        checkpoint_path = checkpoints_dir / f"model_iteration_{iteration + 1:03d}.pth"
        torch.save(learner_model.state_dict(), checkpoint_path)
        save_training_state(state_path, learner_model, optimizer, iteration + 1, config)
        elapsed = time.time() - started
        row: dict[str, object] = {
            "iteration": iteration + 1,
            "samples": len(samples),
            "batches": stats.batches,
            "loss": stats.loss,
            "loss_value": stats.loss_value,
            "loss_policy": stats.loss_policy,
            "elapsed_seconds": elapsed,
            "checkpoint_path": str(checkpoint_path),
            **outcomes,
        }
        append_metrics(metrics_path, row)
        write_jsonl(events_path, {"event": "iteration_completed", **row})
        print(f"iteration={iteration + 1} samples={len(samples)} loss={stats.loss:.6f} checkpoint={checkpoint_path}")

    final_model_path = run_dir / "model_final.pth"
    torch.save(learner_model.state_dict(), final_model_path)
    write_jsonl(events_path, {"event": "run_completed", "final_model": str(final_model_path)})
    print(f"Final model saved: {final_model_path}")


if __name__ == "__main__":
    main()
