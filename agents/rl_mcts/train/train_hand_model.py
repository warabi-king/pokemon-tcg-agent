"""Train the opponent hand-retention model from Kaggle daily episode ZIPs."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
REPO_ROOT = AGENT_ROOT.parents[1]
sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

from cg.api import SelectContext  # noqa: E402
from rl_mcts.model import card_count  # noqa: E402
from rl_mcts.opponent_hand import (  # noqa: E402
    HAND_SCALAR_SIZE,
    HandScoreModel,
)


DEFAULT_DATASET = REPO_ROOT / "tools" / "deck_generator" / "daily_dataset"
DEFAULT_OUTPUT = SRC_ROOT / "rl_mcts" / "hand_model.pth"


@dataclass
class HandSample:
    deck_counts: Counter[int]
    public_counts: Counter[int]
    remaining_counts: Counter[int]
    hand_counts: Counter[int]
    scalars: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-episodes", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def episode_decks(episode: dict) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    for step in episode.get("steps", []):
        for side, state in enumerate(step[:2]):
            action = state.get("action")
            if (
                decks[side] is None
                and isinstance(action, list)
                and len(action) == 60
                and all(isinstance(card_id, int) and card_id > 0 for card_id in action)
            ):
                decks[side] = list(action)
        if all(deck is not None for deck in decks):
            break
    return decks


def scalar_features(current: dict, target_index: int) -> tuple[float, ...]:
    player = current["players"][target_index]
    return (
        min(float(current["turn"]) / 20.0, 2.0),
        min(float(player["handCount"]) / 15.0, 2.0),
        float(player["deckCount"]) / 60.0,
        len(player["prize"]) / 6.0,
        min(len(player["discard"]) / 30.0, 2.0),
        len(player["bench"]) / 5.0,
        float(current["supporterPlayed"]),
        float(current["energyAttached"]),
        float(current["stadiumPlayed"]),
        float(current["retreated"]),
    )


def raw_pokemon_ids(pokemon: dict | None) -> list[int]:
    if pokemon is None:
        return []
    cards = [int(pokemon["id"])]
    for area in ("preEvolution", "tools", "energyCards"):
        cards.extend(int(card["id"]) for card in pokemon.get(area, []) if card is not None)
    return cards


def raw_public_counter(current: dict, target_index: int) -> Counter[int]:
    player = current["players"][target_index]
    cards = [int(card["id"]) for card in player["discard"] if card is not None]
    for pokemon in player["active"]:
        cards.extend(raw_pokemon_ids(pokemon))
    for pokemon in player["bench"]:
        cards.extend(raw_pokemon_ids(pokemon))
    cards.extend(int(card["id"]) for card in player["prize"] if card is not None)
    cards.extend(
        int(card["id"])
        for card in current["stadium"]
        if int(card["playerIndex"]) == target_index
    )
    return Counter(cards)


def selected_snapshots(episode: dict) -> list[tuple[int, dict]]:
    """Keep first and last MAIN prompt per player-turn to reduce correlated repeats."""

    by_turn: dict[tuple[int, int], tuple[dict, dict]] = {}
    seen_inputs: set[str] = set()
    for step in episode.get("steps", []):
        for side, state in enumerate(step[:2]):
            raw = state.get("observation")
            if not isinstance(raw, dict):
                continue
            current = raw.get("current")
            select = raw.get("select")
            token = raw.get("search_begin_input")
            if (
                not isinstance(current, dict)
                or not isinstance(select, dict)
                or current.get("yourIndex") != side
                or select.get("context") != int(SelectContext.MAIN)
                or not token
                or token in seen_inputs
            ):
                continue
            hand = current.get("players", [{}, {}])[side].get("hand")
            if not isinstance(hand, list) or not hand:
                continue
            seen_inputs.add(token)
            key = (side, int(current.get("turn", 0)))
            if key not in by_turn:
                by_turn[key] = (raw, raw)
            else:
                by_turn[key] = (by_turn[key][0], raw)

    result: list[tuple[int, dict]] = []
    for (side, _), (first, last) in by_turn.items():
        result.append((side, first))
        if last.get("search_begin_input") != first.get("search_begin_input"):
            result.append((side, last))
    return result


def make_sample(raw_obs: dict, target_index: int, deck: list[int]) -> HandSample | None:
    current = raw_obs["current"]
    player = current["players"][target_index]
    hand = player.get("hand")
    if not isinstance(hand, list) or not hand:
        return None

    deck_counts = Counter(deck)
    public_counts = raw_public_counter(current, target_index)
    remaining_counts = deck_counts - public_counts
    hand_counts = Counter(int(card["id"]) for card in hand)
    if hand_counts - remaining_counts:
        return None
    scalars = scalar_features(current, target_index)
    if len(scalars) != HAND_SCALAR_SIZE:
        raise AssertionError("hand scalar feature size mismatch")
    return HandSample(
        deck_counts=deck_counts,
        public_counts=public_counts,
        remaining_counts=remaining_counts,
        hand_counts=hand_counts,
        scalars=scalars,
    )


def collect_samples(args: argparse.Namespace) -> tuple[list[HandSample], list[HandSample], dict]:
    train: list[HandSample] = []
    validation: list[HandSample] = []
    episodes_seen = 0
    episodes_failed = 0
    snapshots_skipped = 0
    zip_files = sorted(args.dataset_dir.glob("*/*.zip"))
    zip_files_processed = 0
    per_zip_limit = (
        math.ceil(args.max_episodes / len(zip_files))
        if args.max_episodes and zip_files
        else None
    )
    per_zip_sample_limit = (
        math.ceil(args.max_samples / len(zip_files))
        if args.max_samples and zip_files
        else None
    )
    stop = False

    for zip_path in zip_files:
        zip_episodes = 0
        samples_before_zip = len(train) + len(validation)
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if args.max_episodes and episodes_seen >= args.max_episodes:
                    stop = True
                    break
                if args.max_samples and len(train) + len(validation) >= args.max_samples:
                    stop = True
                    break
                if per_zip_limit is not None and zip_episodes >= per_zip_limit:
                    break
                if (
                    per_zip_sample_limit is not None
                    and len(train) + len(validation) - samples_before_zip
                    >= per_zip_sample_limit
                ):
                    break
                try:
                    with archive.open(name) as source:
                        episode = json.load(source)
                    decks = episode_decks(episode)
                    if not all(decks):
                        episodes_failed += 1
                        episodes_seen += 1
                        zip_episodes += 1
                        continue
                    destination = validation if episodes_seen % 10 == 0 else train
                    for side, raw_obs in selected_snapshots(episode):
                        sample = make_sample(raw_obs, side, decks[side])
                        if sample is None:
                            snapshots_skipped += 1
                            continue
                        destination.append(sample)
                        if args.max_samples and len(train) + len(validation) >= args.max_samples:
                            break
                        if (
                            per_zip_sample_limit is not None
                            and len(train) + len(validation) - samples_before_zip
                            >= per_zip_sample_limit
                        ):
                            break
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    episodes_failed += 1
                episodes_seen += 1
                zip_episodes += 1
                if args.progress_interval and episodes_seen % args.progress_interval == 0:
                    print(
                        f"episodes={episodes_seen} train={len(train)} validation={len(validation)}",
                        flush=True,
                    )
        if zip_episodes:
            zip_files_processed += 1
        if stop:
            break

    summary = {
        "episodes_seen": episodes_seen,
        "episodes_failed": episodes_failed,
        "snapshots_skipped": snapshots_skipped,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "zip_files_available": len(zip_files),
        "zip_files_processed": zip_files_processed,
    }
    return train, validation, summary


def batch_tensors(
    samples: list[HandSample], vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(samples)
    features = torch.zeros(
        batch_size, 2 * vocab_size + HAND_SCALAR_SIZE, dtype=torch.float32
    )
    remaining = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
    target = torch.zeros(batch_size, vocab_size, dtype=torch.float32)
    for row, sample in enumerate(samples):
        for card_id, count in sample.deck_counts.items():
            if 0 <= card_id < vocab_size:
                features[row, card_id] = min(count / 4.0, 3.0)
        for card_id, count in sample.public_counts.items():
            if 0 <= card_id < vocab_size:
                features[row, vocab_size + card_id] = min(count / 4.0, 3.0)
        features[row, 2 * vocab_size :] = torch.tensor(sample.scalars)
        for card_id, count in sample.remaining_counts.items():
            if 0 <= card_id < vocab_size:
                remaining[row, card_id] = float(count)
        hand_size = sum(sample.hand_counts.values())
        for card_id, count in sample.hand_counts.items():
            if 0 <= card_id < vocab_size:
                target[row, card_id] = count / hand_size
    return features.to(device), remaining.to(device), target.to(device)


def hand_loss(
    model: HandScoreModel,
    features: torch.Tensor,
    remaining: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    base = torch.where(remaining > 0, remaining.log(), torch.full_like(remaining, -1e9))
    log_probability = torch.log_softmax(model(features) + base, dim=1)
    return -(target * log_probability).sum(dim=1).mean()


def greedy_overlap(scores: torch.Tensor, sample: HandSample) -> float:
    available = sample.remaining_counts.copy()
    predicted: Counter[int] = Counter()
    for _ in range(sum(sample.hand_counts.values())):
        if not available:
            break
        card_id = max(
            available,
            key=lambda value: math_log_count(available[value]) + float(scores[value]),
        )
        predicted[card_id] += 1
        available[card_id] -= 1
        if available[card_id] <= 0:
            del available[card_id]
    return sum((predicted & sample.hand_counts).values()) / sum(sample.hand_counts.values())


def math_log_count(value: int) -> float:
    return math.log(max(value, 1))


def evaluate(
    model: HandScoreModel,
    samples: list[HandSample],
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    model.eval()
    loss_total = 0.0
    model_overlap = 0.0
    base_overlap = 0.0
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            features, remaining, target = batch_tensors(batch, vocab_size, device)
            score_batch = model(features)
            loss_total += float(hand_loss(model, features, remaining, target).item()) * len(batch)
            for scores, sample in zip(score_batch.cpu(), batch):
                model_overlap += greedy_overlap(scores, sample)
                base_overlap += greedy_overlap(torch.zeros_like(scores), sample)
    count = len(samples)
    return loss_total / count, model_overlap / count, base_overlap / count


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--device gpu was requested, but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "gpu" else "cpu")

    train_samples, validation_samples, summary = collect_samples(args)
    if not train_samples:
        raise RuntimeError("no hand training samples were collected")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    vocab_size = card_count()
    model = HandScoreModel(vocab_size, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_samples)
        loss_total = 0.0
        count = 0
        for start in range(0, len(train_samples), args.batch_size):
            batch = train_samples[start : start + args.batch_size]
            features, remaining, target = batch_tensors(batch, vocab_size, device)
            optimizer.zero_grad()
            loss = hand_loss(model, features, remaining, target)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item()) * len(batch)
            count += len(batch)
        validation_loss, model_overlap, base_overlap = evaluate(
            model,
            validation_samples,
            args.batch_size,
            vocab_size,
            device,
        )
        print(
            f"epoch={epoch} train_loss={loss_total / count:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"overlap={model_overlap:.4f} base_overlap={base_overlap:.4f}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "vocab_size": vocab_size,
            "hidden_size": args.hidden_size,
            "training_summary": summary,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
