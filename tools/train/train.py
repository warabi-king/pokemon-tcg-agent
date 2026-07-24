"""rl_mctsの自己対戦学習スクリプト。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
from typing import Iterator

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from cg.api import to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl_mcts.deck import read_deck_csv  # noqa: E402
from rl_mcts.mcts import LearnInput, LearnSample, MAX_ACTIONS, mcts_agent  # noqa: E402
from rl_mcts.model import create_model  # noqa: E402


@dataclass
class TrainStats:
    """1iterationの学習損失集計。"""

    batches: int
    loss: float
    loss_value: float
    loss_policy: float


def progress(count: int, text: str) -> Iterator[int]:
    """stderrへ進捗率を出す簡易progress iterator。"""
    current = 0
    while True:
        percent = 100 * current // max(count, 1)
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if current >= count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1


def random_agent(obs_dict: dict) -> list[int]:
    """評価用のランダムagent。"""
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)


def raise_for_deck_error(start_data) -> None:
    """battle_startのデッキエラーを読みやすい例外にする。"""
    if start_data.errorPlayer < 0:
        return

    error = "Deck error."
    if start_data.errorType == 1:
        error = "The deck contains invalid card ID."
    elif start_data.errorType == 2:
        error = (
            "You can include up to four cards with the same name in the deck, "
            "excluding basic Energy cards."
        )
    elif start_data.errorType == 3:
        error = "There are no Basic Pokemon in the deck."
    elif start_data.errorType == 4:
        error = "You can include only one Ace Spec card in the deck."
    raise ValueError(error)


def load_deck_csv(path: Path) -> list[int]:
    """deck.csvを読み込んでカードIDリストを返す。"""
    if not path.exists():
        raise FileNotFoundError(f"Deck file not found: {path}")
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 cards, but got {len(deck)}: {path}")
    return deck


def evaluate(model, deck: list[int], games: int, search_count: int) -> tuple[int, int, int]:
    """ランダムagent相手に評価する。戻り値は win, lose, draw。"""
    results = [0, 0, 0]
    for i in progress(games, "Evaluating... "):
        obs, start_data = battle_start(deck, deck)
        raise_for_deck_error(start_data)
        your_index = i % 2
        while True:
            if obs["current"]["result"] >= 0:
                break

            if obs["current"]["yourIndex"] == your_index:
                selected, _ = mcts_agent(obs, deck, model, search_count=search_count)
            else:
                selected = random_agent(obs)
            obs = battle_select(selected)

        battle_finish()

        if obs["current"]["result"] == 2:
            results[2] += 1
        elif obs["current"]["result"] == your_index:
            results[0] += 1
        else:
            results[1] += 1
    return results[0], results[1], results[2]


def collect_self_play_samples(
    model,
    deck: list[int],
    games: int,
    search_count: int,
    lambda_value: float,
) -> list[LearnSample]:
    """自己対戦で学習サンプルを収集する。"""
    sample_list: list[LearnSample] = []

    for _ in progress(games, "Training Data Collecting... "):
        obs, start_data = battle_start(deck, deck)
        raise_for_deck_error(start_data)
        samples: list[list[LearnSample]] = [[], []]
        while True:
            if obs["current"]["result"] >= 0:
                break

            selected, sample = mcts_agent(obs, deck, model, search_count=search_count)
            if sample is not None:
                samples[obs["current"]["yourIndex"]].append(sample)
            obs = battle_select(selected)

        battle_finish()

        for player_index in range(2):
            if obs["current"]["result"] == 2:
                value = 0.0
            else:
                value = 1.0 if player_index == obs["current"]["result"] else -1.0

            for sample in reversed(samples[player_index]):
                label = (value + sample.value) * 0.5
                value = value * lambda_value + sample.value * (1.0 - lambda_value)
                sample.value = label
                sample_list.append(sample)

    return sample_list


def collect_cross_play_samples(
    model_a,
    model_b,
    deck_a: list[int],
    deck_b: list[int],
    games: int,
    search_count: int,
    lambda_value: float,
) -> tuple[list[LearnSample], list[LearnSample]]:
    """モデルAとモデルBを対戦させ、A側とB側の学習サンプルを返す。

    A側は player0、B側は player1 として対戦が組まれます。
    戻り値は (samples_for_A, samples_for_B)。
    """
    samples_a: list[LearnSample] = []
    samples_b: list[LearnSample] = []

    for _ in progress(games, "Cross-play Data Collecting... "):
        obs, start_data = battle_start(deck_a, deck_b)
        raise_for_deck_error(start_data)
        per_player_samples: list[list[LearnSample]] = [[], []]
        while True:
            if obs["current"]["result"] >= 0:
                break

            if obs["current"]["yourIndex"] == 0:
                selected, sample = mcts_agent(obs, deck_a, model_a, search_count=search_count)
            else:
                selected, sample = mcts_agent(obs, deck_b, model_b, search_count=search_count)

            if sample is not None:
                per_player_samples[obs["current"]["yourIndex"]].append(sample)

            obs = battle_select(selected)

        battle_finish()

        for player_index in range(2):
            if obs["current"]["result"] == 2:
                value = 0.0
            else:
                value = 1.0 if player_index == obs["current"]["result"] else -1.0

            for sample in reversed(per_player_samples[player_index]):
                label = (value + sample.value) * 0.5
                value = value * lambda_value + sample.value * (1.0 - lambda_value)
                sample.value = label
                if player_index == 0:
                    samples_a.append(sample)
                else:
                    samples_b.append(sample)

    return samples_a, samples_b


def train_one_iteration(
    model,
    optimizer,
    samples: list[LearnSample],
    batch_size: int,
    device: torch.device,
) -> TrainStats:
    """収集済みサンプルで1iterationぶん学習し、平均lossを返す。"""
    if len(samples) < batch_size:
        print(f"Training skipped: samples={len(samples)}, batch_size={batch_size}")
        return TrainStats(batches=0, loss=0.0, loss_value=0.0, loss_policy=0.0)

    model.train()
    random.shuffle(samples)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    batch_count = len(samples) // batch_size
    loss_total = 0.0
    loss_value_total = 0.0
    loss_policy_total = 0.0

    for i in range(batch_count):
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask: list[float] = []
        label_enc: list[float] = []
        label_dec: list[float] = []
        start = batch_size * i

        for sample in samples[start : start + batch_size]:
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            mask.extend([1.0] * len(sample.policy))
            for _ in range(MAX_ACTIONS - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(
            batch_size,
            -1,
        )
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(
            batch_size,
            -1,
        )

        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        )

        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = (loss_dec * mask_tensor).sum() / float(batch_size)
        loss = loss_enc + loss_dec
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        loss_value_total += float(loss_enc.item())
        loss_policy_total += float(loss_dec.item())

    return TrainStats(
        batches=batch_count,
        loss=loss_total / batch_count,
        loss_value=loss_value_total / batch_count,
        loss_policy=loss_policy_total / batch_count,
    )


def append_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    """iterationメトリクスをCSVへ追記する。"""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "eval_games",
        "eval_win",
        "eval_lose",
        "eval_draw",
        "eval_win_rate",
        "games",
        "samples",
        "batches",
        "loss",
        "loss_value",
        "loss_policy",
        "elapsed_seconds",
        "checkpoint_path",
        "model_path",
    ]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5, help="学習iteration数")
    parser.add_argument("--eval-games", type=int, default=50, help="各iterationの評価試合数")
    parser.add_argument("--games", type=int, default=100, help="各iterationの対戦（self/cross）試合数")
    parser.add_argument("--batch-size", type=int, default=128, help="学習batch size")
    parser.add_argument("--search-count", type=int, default=10, help="MCTS探索回数")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--lambda-value", type=float, default=0.9, help="終局価値の逆向き更新率")
    parser.add_argument("--plot", action="store_true", help="学習後にPNGグラフを生成する")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=AGENT_ROOT / "train" / "checkpoints",
        help="iterationごとのcheckpoint保存先",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=AGENT_ROOT / "train" / "logs",
        help="CSVログとグラフの保存先",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="CSVログファイル。省略時は {log-dir}/train_metrics.csv。",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=SRC_ROOT / "model.pth",
        help="提出用に採用する最終モデル保存先",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="2モデルを同一プロセスで交互対戦させて両方更新するモード",
    )
    parser.add_argument(
        "--deck-a",
        type=Path,
        default=None,
        help="(dual) model A に使う deck.csv のパス。省略時は現在の agent の deck.csv。",
    )
    parser.add_argument(
        "--deck-b",
        type=Path,
        default=None,
        help="(dual) model B に使う deck.csv のパス。省略時は現在の agent の deck.csv。",
    )
    parser.add_argument(
        "--opponent-model",
        type=Path,
        default=None,
        help="(dual) 相手モデルの初期重みを読み込むパス。省略時はランダム初期化。",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="モデルA（単体実行時は唯一のモデル）の初期重みを読み込むパス。省略時はランダム初期化。",
    )
    parser.add_argument(
        "--opponent-output",
        type=Path,
        default=SRC_ROOT / "model_b.pth",
        help="(dual) 相手モデルの最終保存先",
    )
    return parser.parse_args()


def main() -> None:
    """学習処理の入口。"""
    args = parse_args()
    if args.dual:
        # Dual training: maintain two models (A,B), collect cross-play samples,
        # and update both models within the same process.
        metrics_path_a = args.metrics_file or args.log_dir / "a" / "train_metrics.csv"
        metrics_path_b = args.log_dir / "b" / "train_metrics.csv"
        default_deck = read_deck_csv()
        deck_a = default_deck if args.deck_a is None else load_deck_csv(args.deck_a)
        deck_b = default_deck if args.deck_b is None else load_deck_csv(args.deck_b)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_a = create_model().to(device)
        model_b = create_model().to(device)
        # load initial weights if provided
        if args.init_model and args.init_model.exists():
            state = torch.load(args.init_model, map_location=device)
            model_a.load_state_dict(state)
        if args.opponent_model and args.opponent_model.exists():
            state = torch.load(args.opponent_model, map_location=device)
            model_b.load_state_dict(state)

        opt_a = torch.optim.AdamW(model_a.parameters(), lr=args.lr)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=args.lr)

        # per-model checkpoint/log dirs
        ckpt_a = args.checkpoint_dir / "a"
        ckpt_b = args.checkpoint_dir / "b"
        log_a = args.log_dir / "a"
        log_b = args.log_dir / "b"
        ckpt_a.mkdir(parents=True, exist_ok=True)
        ckpt_b.mkdir(parents=True, exist_ok=True)
        log_a.mkdir(parents=True, exist_ok=True)
        log_b.mkdir(parents=True, exist_ok=True)
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        args.opponent_output.parent.mkdir(parents=True, exist_ok=True)

        for iteration in range(args.iterations):
            started = time.time()
            # save checkpoints
            cp_a = ckpt_a / f"model_{iteration}.pth"
            cp_b = ckpt_b / f"model_{iteration}.pth"
            torch.save(model_a.state_dict(), cp_a)
            torch.save(model_b.state_dict(), cp_b)
            print(f"Checkpoint saved: {cp_a}, {cp_b}")

            # evaluation vs random
            model_a.eval()
            model_b.eval()
            with torch.inference_mode():
                if args.eval_games > 0:
                    wa, la, da = evaluate(model_a, deck_a, args.eval_games, args.search_count)
                    wb, lb, db = evaluate(model_b, deck_b, args.eval_games, args.search_count)
                    decided_a = wa + la
                    decided_b = wb + lb
                    win_rate_a = 100.0 * wa / decided_a if decided_a else 0.0
                    win_rate_b = 100.0 * wb / decided_b if decided_b else 0.0
                    print(f"Eval A winrate {win_rate_a:.1f}% (W/L/D={wa}/{la}/{da})")
                    print(f"Eval B winrate {win_rate_b:.1f}% (W/L/D={wb}/{lb}/{db})")
                else:
                    wa = la = da = wb = lb = db = 0

                # collect cross-play samples
                samples_a, samples_b = collect_cross_play_samples(
                    model_a,
                    model_b,
                    deck_a,
                    deck_b,
                    args.games,
                    args.search_count,
                    args.lambda_value,
                )

            print(f"Training Start. samples A={len(samples_a)} B={len(samples_b)}")
            stats_a = train_one_iteration(model_a, opt_a, samples_a, args.batch_size, device)
            stats_b = train_one_iteration(model_b, opt_b, samples_b, args.batch_size, device)
            elapsed = time.time() - started
            print(
                f"Training Finish. A batches={stats_a.batches} loss={stats_a.loss:.6f} "
                f"B batches={stats_b.batches} loss={stats_b.loss:.6f} elapsed={elapsed:.1f}s"
            )

            append_metrics(
                metrics_path_a,
                {
                    "iteration": iteration,
                    "eval_games": args.eval_games,
                    "eval_win": wa,
                    "eval_lose": la,
                    "eval_draw": da,
                    "eval_win_rate": 100.0 * wa / (wa + la) if (wa + la) else 0.0,
                    "games": args.games,
                    "samples": len(samples_a),
                    "batches": stats_a.batches,
                    "loss": stats_a.loss,
                    "loss_value": stats_a.loss_value,
                    "loss_policy": stats_a.loss_policy,
                    "elapsed_seconds": elapsed,
                    "checkpoint_path": cp_a,
                    "model_path": args.output_model,
                },
            )

            append_metrics(
                metrics_path_b,
                {
                    "iteration": iteration,
                    "eval_games": args.eval_games,
                    "eval_win": wb,
                    "eval_lose": lb,
                    "eval_draw": db,
                    "eval_win_rate": 100.0 * wb / (wb + lb) if (wb + lb) else 0.0,
                    "games": args.games,
                    "samples": len(samples_b),
                    "batches": stats_b.batches,
                    "loss": stats_b.loss,
                    "loss_value": stats_b.loss_value,
                    "loss_policy": stats_b.loss_policy,
                    "elapsed_seconds": elapsed,
                    "checkpoint_path": cp_b,
                    "model_path": args.opponent_output,
                },
            )

        # save final models
        torch.save(model_a.state_dict(), args.output_model)
        torch.save(model_b.state_dict(), args.opponent_output)
        print(f"Final models saved: {args.output_model}, {args.opponent_output}")
        return
    metrics_path = args.metrics_file or args.log_dir / "train_metrics.csv"
    deck = read_deck_csv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    if args.init_model and args.init_model.exists():
        state = torch.load(args.init_model, map_location=device)
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.iterations):
        started = time.time()
        checkpoint_path = args.checkpoint_dir / f"model_{iteration}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        win = lose = draw = 0
        model.eval()
        with torch.inference_mode():
            if args.eval_games > 0:
                win, lose, draw = evaluate(model, deck, args.eval_games, args.search_count)
                decided = win + lose
                win_rate = 100.0 * win / decided if decided else 0.0
                print(f"Evaluation win rate {win_rate:.1f}% (W/L/D={win}/{lose}/{draw})")
            else:
                win_rate = 0.0

            samples = collect_self_play_samples(
                model,
                deck,
                args.games,
                args.search_count,
                args.lambda_value,
            )

        print(f"Training Start. samples={len(samples)}")
        train_stats = train_one_iteration(model, optimizer, samples, args.batch_size, device)
        elapsed = time.time() - started
        print(
            "Training Finish. "
            f"batches={train_stats.batches} "
            f"loss={train_stats.loss:.6f} "
            f"value={train_stats.loss_value:.6f} "
            f"policy={train_stats.loss_policy:.6f} "
            f"elapsed={elapsed:.1f}s"
        )

        append_metrics(
            metrics_path,
            {
                "iteration": iteration,
                "eval_games": args.eval_games,
                "eval_win": win,
                "eval_lose": lose,
                "eval_draw": draw,
                "eval_win_rate": win_rate,
                "games": args.games,
                "samples": len(samples),
                "batches": train_stats.batches,
                "loss": train_stats.loss,
                "loss_value": train_stats.loss_value,
                "loss_policy": train_stats.loss_policy,
                "elapsed_seconds": elapsed,
                "checkpoint_path": checkpoint_path,
                "model_path": args.output_model,
            },
        )
        print(f"Metrics appended: {metrics_path}")

    torch.save(model.state_dict(), args.output_model)
    print(f"Final model saved: {args.output_model}")

    if args.plot:
        from plot_metrics import plot_metrics

        plot_metrics(metrics_path, args.log_dir)


if __name__ == "__main__":
    main()
