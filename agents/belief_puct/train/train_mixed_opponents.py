"""固定デッキで自己対戦・凍結belief・外部agentを混ぜてRL学習する。

実行例:
    python agents/belief_puct/train/train_mixed_opponents.py \
        --deck agents/belief_puct/train/runs/phase2a-001/candidates/candidate_004/deck.csv \
        --initial-model agents/belief_puct/src/model.pth \
        --run-dir agents/belief_puct/train/runs/phase2b-001 \
        --external-opponents results/phase2a/opponents.json \
        --iterations 10 --games-per-iteration 40 --external-opponent-fraction 0.5

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
from dataclasses import dataclass
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
TOOLS_ROOT = AGENT_ROOT.parents[1] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

import torch  # noqa: E402

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl_mcts.mcts import LearnSample, mcts_agent  # noqa: E402
from rl_mcts.model import MyModel, create_model  # noqa: E402
from train import TrainStats, raise_for_deck_error, train_one_iteration  # noqa: E402
from evaluate_fixed_league import IsolatedAgent, read_deck  # noqa: E402


@dataclass(frozen=True)
class ExternalOpponent:
    """外部agentを隔離して学習対戦へ参加させるための入力一式。"""

    name: str
    agent_src: Path
    deck_path: Path
    model_path: Path | None


def resolve_config_path(raw_path: str, config_path: Path) -> Path:
    """外部相手JSON内の相対pathをJSON配置ディレクトリから解決する。"""

    path = Path(raw_path)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_external_opponents(config_path: Path) -> list[ExternalOpponent]:
    """Phase 2a形式の相手JSONを読み、外部agentの必須入力を検証する。"""

    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"外部相手JSONを解釈できません: {config_path}: {error}") from error
    raw_opponents = content.get("opponents") if isinstance(content, dict) else None
    if not isinstance(raw_opponents, list) or not raw_opponents:
        raise ValueError("外部相手JSONには空でない opponents 配列が必要です")

    opponents: list[ExternalOpponent] = []
    seen_names: set[str] = set()
    for index, raw_opponent in enumerate(raw_opponents):
        if not isinstance(raw_opponent, dict):
            raise ValueError(f"opponents[{index}] はobjectである必要があります")
        try:
            name = str(raw_opponent["name"])
            agent_src = resolve_config_path(str(raw_opponent["agent_src"]), config_path)
            deck_path = resolve_config_path(str(raw_opponent["deck"]), config_path)
        except KeyError as error:
            raise ValueError(f"opponents[{index}] に必須項目 {error.args[0]} がありません") from error
        if not name or name in seen_names:
            raise ValueError(f"外部相手名は空または重複できません: {name!r}")
        raw_model = raw_opponent.get("model")
        model_path = resolve_config_path(str(raw_model), config_path) if raw_model else None
        for label, path in (("agent_src", agent_src), ("deck", deck_path), ("model", model_path)):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"opponents[{index}] の {label} がありません: {path}")
        if not (agent_src / "main.py").is_file():
            raise FileNotFoundError(f"opponents[{index}] の agent_src にmain.pyがありません: {agent_src}")
        read_deck(deck_path)
        opponents.append(ExternalOpponent(name, agent_src, deck_path, model_path))
        seen_names.add(name)
    return opponents


def load_model(checkpoint_path: Path, device: torch.device) -> MyModel:
    """raw state_dictまたは本スクリプトの学習stateからモデルを復元する。"""

    payload = torch.load(checkpoint_path, map_location=device)
    # 提出用model.pthは生のstate_dictであり、学習再開stateだけがmodel_stateを包む。
    state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
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
) -> tuple[list[LearnSample], str, int]:
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
        return samples_by_player[0] + samples_by_player[1], "play", result

    learner_samples = samples_by_player[learner_index]
    assign_terminal_values(learner_samples, result, learner_index, lambda_value)
    if result == 2:
        return learner_samples, "draw", result
    return learner_samples, "win" if result == learner_index else "loss", result


def play_external_training_game(
    learner_model: MyModel,
    opponent: IsolatedAgent,
    learner_deck: list[int],
    opponent_deck: list[int],
    search_count: int,
    lambda_value: float,
    learner_index: int,
) -> tuple[list[LearnSample], str, int]:
    """外部agentの各自deckと対戦し、学習側のsampleだけを収集する。"""

    deck0, deck1 = (learner_deck, opponent_deck) if learner_index == 0 else (opponent_deck, learner_deck)
    obs, start_data = battle_start(deck0, deck1)
    raise_for_deck_error(start_data)
    learner_samples: list[LearnSample] = []
    opponent.reset_match()
    try:
        while obs["current"]["result"] < 0:
            current_index = int(obs["current"]["yourIndex"])
            if current_index == learner_index:
                selected, sample = mcts_agent(obs, learner_deck, learner_model, search_count=search_count)
                if sample is not None:
                    learner_samples.append(sample)
            else:
                selected = opponent(obs)
            obs = battle_select(selected)
    finally:
        battle_finish()

    result = int(obs["current"]["result"])
    assign_terminal_values(learner_samples, result, learner_index, lambda_value)
    if result == 2:
        return learner_samples, "draw", result
    return learner_samples, "win" if result == learner_index else "loss", result


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
    parser.add_argument("--external-opponents", type=Path, default=None, help="Phase 2a形式の外部相手JSON")
    parser.add_argument("--external-opponent-fraction", type=float, default=0.0, help="外部agent相手の比率")
    parser.add_argument("--action-timeout-seconds", type=float, default=600.0, help="外部agentの1回の推論上限秒")
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
    if not 0.0 <= args.external_opponent_fraction <= 1.0:
        raise ValueError("external-opponent-fractionは0以上1以下で指定してください")
    if args.frozen_opponent_fraction + args.external_opponent_fraction > 1.0:
        raise ValueError("frozen-opponent-fractionとexternal-opponent-fractionの合計は1以下にしてください")
    if args.external_opponent_fraction > 0.0 and args.external_opponents is None:
        raise ValueError("external-opponent-fractionを指定するには--external-opponentsが必要です")
    if args.action_timeout_seconds <= 0:
        raise ValueError("action-timeout-secondsは正の値で指定してください")
    for path in [args.deck, args.initial_model, *args.frozen_model]:
        if not path.exists():
            raise FileNotFoundError(f"入力ファイルがありません: {path}")


def main() -> None:
    """固定deck・混合相手でPhase 2bのRL学習を実行する。"""

    args = parse_args()
    validate_args(args)
    external_opponents = load_external_opponents(args.external_opponents.resolve()) if args.external_opponents else []
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
        "external_opponent_fraction": args.external_opponent_fraction,
        "external_opponents": [
            {
                "name": opponent.name,
                "agent_src": str(opponent.agent_src),
                "deck": str(opponent.deck_path),
                "deck_sha256": sha256_file(opponent.deck_path),
                "model": str(opponent.model_path) if opponent.model_path else None,
                "model_sha256": sha256_file(opponent.model_path) if opponent.model_path else None,
            }
            for opponent in external_opponents
        ],
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
    external_workers = [
        IsolatedAgent(
            opponent.agent_src,
            read_deck(opponent.deck_path),
            f"mixed_opponent_{index}",
            args.action_timeout_seconds,
            {"determinizations": None, "search_count": None, "model_path": str(opponent.model_path) if opponent.model_path else None},
        )
        for index, opponent in enumerate(external_opponents)
    ]
    write_jsonl(events_path, {"event": "run_started", "resume": args.resume is not None, "config": config})
    frozen_games = round(args.games_per_iteration * args.frozen_opponent_fraction)
    external_games = round(args.games_per_iteration * args.external_opponent_fraction)

    try:
        for iteration in range(start_iteration, args.iterations):
            started = time.time()
            opponent_kinds = (
                ["frozen"] * frozen_games
                + ["external"] * external_games
                + ["self"] * (args.games_per_iteration - frozen_games - external_games)
            )
            random.shuffle(opponent_kinds)
            samples: list[LearnSample] = []
            outcomes = {
                "self_play": 0,
                "frozen_win": 0,
                "frozen_loss": 0,
                "frozen_draw": 0,
                "external_win": 0,
                "external_loss": 0,
                "external_draw": 0,
                **{
                    f"external_{opponent.name}_{outcome}": 0
                    for opponent in external_opponents
                    for outcome in ("win", "loss", "draw")
                },
            }
            learner_model.eval()
            with torch.inference_mode():
                for game_index, opponent_kind in enumerate(opponent_kinds):
                    game_started = time.time()
                    opponent_name = "self_play"
                    learner_index = 0
                    if opponent_kind == "external":
                        opponent_index = random.randrange(len(external_workers))
                        opponent_name = external_opponents[opponent_index].name
                        learner_index = game_index % 2
                        game_samples, outcome, game_result = play_external_training_game(
                            learner_model,
                            external_workers[opponent_index],
                            deck,
                            read_deck(external_opponents[opponent_index].deck_path),
                            args.search_count,
                            args.lambda_value,
                            learner_index,
                        )
                        outcomes[f"external_{external_opponents[opponent_index].name}_{outcome}"] += 1
                    else:
                        frozen_model = random.choice(frozen_models) if opponent_kind == "frozen" else None
                        if frozen_model is not None:
                            learner_index = game_index % 2
                            opponent_name = "frozen_belief"
                        game_samples, outcome, game_result = play_training_game(
                            learner_model, frozen_model, deck, args.search_count, args.lambda_value, learner_index
                        )
                    samples.extend(game_samples)
                    outcomes[f"{opponent_kind}_{outcome}"] += 1
                    print(
                        f"iteration={iteration + 1} game={game_index + 1}/{args.games_per_iteration} "
                        f"type={opponent_kind} opponent={opponent_name} "
                        f"learner_player={learner_index} "
                        f"result={'draw' if game_result == 2 else f'player_{game_result}_win'} "
                        f"samples={len(game_samples)} elapsed_seconds={time.time() - game_started:.1f}",
                        flush=True,
                    )

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
    finally:
        for worker in external_workers:
            worker.close()


if __name__ == "__main__":
    main()
