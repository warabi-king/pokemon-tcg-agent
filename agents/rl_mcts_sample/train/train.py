"""rl_mcts_sampleの自己対戦学習スクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
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


def train_one_iteration(
    model,
    optimizer,
    samples: list[LearnSample],
    batch_size: int,
    device: torch.device,
) -> int:
    """収集済みサンプルで1iterationぶん学習する。戻り値は更新batch数。"""
    if len(samples) < batch_size:
        print(f"Training skipped: samples={len(samples)}, batch_size={batch_size}")
        return 0

    model.train()
    random.shuffle(samples)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    batch_count = len(samples) // batch_size

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

    return batch_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5, help="学習iteration数")
    parser.add_argument("--eval-games", type=int, default=50, help="各iterationの評価試合数")
    parser.add_argument("--self-play-games", type=int, default=100, help="各iterationの自己対戦数")
    parser.add_argument("--batch-size", type=int, default=128, help="学習batch size")
    parser.add_argument("--search-count", type=int, default=10, help="MCTS探索回数")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate")
    parser.add_argument("--lambda-value", type=float, default=0.9, help="終局価値の逆向き更新率")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=AGENT_ROOT / "train" / "checkpoints",
        help="iterationごとのcheckpoint保存先",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=SRC_ROOT / "model.pth",
        help="提出用に採用する最終モデル保存先",
    )
    return parser.parse_args()


def main() -> None:
    """学習処理の入口。"""
    args = parse_args()
    deck = read_deck_csv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    for iteration in range(args.iterations):
        checkpoint_path = args.checkpoint_dir / f"model_{iteration}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        model.eval()
        with torch.inference_mode():
            if args.eval_games > 0:
                win, lose, draw = evaluate(model, deck, args.eval_games, args.search_count)
                decided = win + lose
                win_rate = 100 * win // decided if decided else 0
                print(f"Evaluation win rate {win_rate}% (W/L/D={win}/{lose}/{draw})")

            samples = collect_self_play_samples(
                model,
                deck,
                args.self_play_games,
                args.search_count,
                args.lambda_value,
            )

        print(f"Training Start. samples={len(samples)}")
        batch_count = train_one_iteration(model, optimizer, samples, args.batch_size, device)
        print(f"Training Finish. batches={batch_count}")

    torch.save(model.state_dict(), args.output_model)
    print(f"Final model saved: {args.output_model}")


if __name__ == "__main__":
    main()
