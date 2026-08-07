"""対戦ZIPの実観測から、公開カード→相手デッキMLPを学習する。

例:
    python agents/rl_mcts/train/train_opponent_deck.py train --date 2026-07-01
    python agents/rl_mcts/train/train_opponent_deck.py predict --observed 646,648 --turn 5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
import zlib
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


SCRIPT_PATH = Path(__file__).resolve()
AGENT_ROOT = SCRIPT_PATH.parents[1]
ROOT = SCRIPT_PATH.parents[3]
SRC_ROOT = AGENT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_mcts.opponent_deck import (  # noqa: E402
    DEFAULT_TURN_SCALE,
    DECK_SIZE,
    OpponentDeckMLP,
    PublicCardTracker,
    complete_deck,
    make_features,
)


DEFAULT_DATASET_DIR = ROOT / "tools" / "deck_generator" / "daily_dataset"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DEFAULT_OUTPUT = SRC_ROOT / "opponent_deck_mlp.pth"


@dataclass(frozen=True)
class CardMeta:
    card_id: int
    name: str
    kind: str
    rule: str

    @property
    def is_basic_energy(self) -> bool:
        return self.kind == "Basic Energy"

    @property
    def is_basic_pokemon(self) -> bool:
        return self.kind == "Basic"

    @property
    def is_ace_spec(self) -> bool:
        return "ACE SPEC" in f"{self.name} {self.rule}".upper()


@dataclass(frozen=True)
class BattleSample:
    observed: array
    deck: array
    turn: int
    episode_key: str


class Reservoir:
    """全期間を偏りなく上限件数へ落とすreservoir sampler。"""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.seen = 0
        self.items: list[BattleSample] = []

    def add(self, sample: BattleSample) -> None:
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(sample)
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            self.items[index] = sample


class TurnDeckDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        samples: list[BattleSample],
        vocab_size: int,
        count_scales: torch.Tensor,
        turn_scale: float,
    ) -> None:
        self.samples = samples
        self.vocab_size = vocab_size
        self.count_scales = count_scales
        self.turn_scale = turn_scale

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        features = make_features(
            sample.observed,
            sample.turn,
            self.vocab_size,
            self.count_scales,
            self.turn_scale,
        )
        target = torch.zeros(self.vocab_size, dtype=torch.float32)
        for card_id in sample.deck:
            if card_id < self.vocab_size:
                target[card_id] += 1.0
        return features, target / self.count_scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="対戦ZIPを走査してMLPを学習する")
    train.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    train.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--date", action="append", help="対象日。複数回指定可")
    train.add_argument("--limit-zips", type=int)
    train.add_argument("--max-episodes", type=int)
    train.add_argument("--max-samples", type=int, default=200_000)
    train.add_argument("--valid-ratio", type=float, default=0.1)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--positive-weight", type=float, default=8.0)
    train.add_argument("--sum-loss-weight", type=float, default=0.05)
    train.add_argument("--turn-scale", type=float, default=DEFAULT_TURN_SCALE)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    train.add_argument("--progress-every", type=int, default=500)

    predict = commands.add_parser("predict", help="checkpointを手動の公開カードで確認する")
    predict.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT)
    predict.add_argument("--observed", action="append", default=[])
    predict.add_argument("--turn", type=int, default=1)
    predict.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_card_meta(path: Path) -> dict[int, CardMeta]:
    result: dict[int, CardMeta] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        kind_column = next(
            (name for name in (reader.fieldnames or []) if "Stage" in name and "Type" in name),
            None,
        )
        if kind_column is None:
            raise ValueError(f"カード種別列が見つかりません: {path}")
        for row in reader:
            card_id = int(row["Card ID"])
            result[card_id] = CardMeta(
                card_id=card_id,
                name=row["Card Name"],
                kind=row[kind_column],
                rule=row.get("Rule", ""),
            )
    return result


def find_zip_files(dataset_dir: Path, dates: list[str] | None, limit: int | None) -> list[Path]:
    paths = sorted(path for path in dataset_dir.glob("*/*.zip") if path.is_file())
    if dates:
        wanted = set(dates)
        paths = [path for path in paths if path.parent.name in wanted]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"対戦ZIPが見つかりません: {dataset_dir}")
    return paths


def collect_decks(episode: dict[str, Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    for step in episode.get("steps", []):
        for player_index in (0, 1):
            if player_index >= len(step):
                continue
            action = step[player_index].get("action")
            if decks[player_index] is None and isinstance(action, list) and len(action) == DECK_SIZE:
                decks[player_index] = [int(card_id) for card_id in action]
        if all(deck is not None for deck in decks):
            break
    return decks


def iter_episode_samples(episode: dict[str, Any], episode_key: str) -> Iterator[BattleSample]:
    """各observerについて、そのターン最後の実観測を1件ずつ返す。"""
    decks = collect_decks(episode)
    if any(deck is None for deck in decks):
        return

    trackers = [PublicCardTracker(), PublicCardTracker()]
    pending: list[tuple[int, tuple[int, ...]] | None] = [None, None]
    for step in episode.get("steps", []):
        for observer in (0, 1):
            if observer >= len(step):
                continue
            observation = step[observer].get("observation")
            current = observation.get("current") if isinstance(observation, dict) else None
            if not isinstance(current, dict) or current.get("yourIndex") != observer:
                continue
            public = trackers[observer].update(observation)
            previous = pending[observer]
            if previous is not None and previous[0] != public.turn:
                target = decks[1 - observer]
                assert target is not None
                yield BattleSample(array("H", previous[1]), array("H", target), previous[0], episode_key)
            pending[observer] = (public.turn, public.cards)

    for observer, previous in enumerate(pending):
        if previous is None:
            continue
        target = decks[1 - observer]
        assert target is not None
        yield BattleSample(array("H", previous[1]), array("H", target), previous[0], episode_key)


def is_validation_episode(episode_key: str, valid_ratio: float) -> bool:
    threshold = int(valid_ratio * 10_000)
    return zlib.crc32(episode_key.encode("utf-8")) % 10_000 < threshold


def scan_battle_samples(args: argparse.Namespace) -> tuple[list[BattleSample], list[BattleSample], dict[str, int]]:
    if not 0 < args.valid_ratio < 1:
        raise ValueError("--valid-ratioは0より大きく1より小さくしてください")
    if args.max_samples < 2:
        raise ValueError("--max-samplesは2以上にしてください")

    valid_capacity = max(1, int(args.max_samples * args.valid_ratio))
    train_capacity = args.max_samples - valid_capacity
    train_reservoir = Reservoir(train_capacity, args.seed)
    valid_reservoir = Reservoir(valid_capacity, args.seed + 1)
    zip_files = find_zip_files(args.dataset_dir, args.date, args.limit_zips)
    episodes = 0
    parse_errors = 0

    stop = False
    for zip_path in zip_files:
        with zipfile.ZipFile(zip_path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir() and info.filename.endswith(".json")]
            for info in members:
                if args.max_episodes is not None and episodes >= args.max_episodes:
                    stop = True
                    break
                episode_key = f"{zip_path.parent.name}/{info.filename}"
                try:
                    episode = json.loads(archive.read(info))
                    reservoir = (
                        valid_reservoir
                        if is_validation_episode(episode_key, args.valid_ratio)
                        else train_reservoir
                    )
                    for sample in iter_episode_samples(episode, episode_key):
                        reservoir.add(sample)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as exc:
                    parse_errors += 1
                    if parse_errors <= 5:
                        print(f"warning: {episode_key}: {exc}", file=sys.stderr)
                episodes += 1
                if args.progress_every > 0 and episodes % args.progress_every == 0:
                    print(
                        f"episodes={episodes} train_seen={train_reservoir.seen} "
                        f"valid_seen={valid_reservoir.seen}",
                        flush=True,
                    )
        if stop:
            break

    if not train_reservoir.items:
        raise ValueError("学習サンプルを生成できませんでした")
    if not valid_reservoir.items:
        valid_reservoir.items.append(train_reservoir.items[-1])
    stats = {
        "zip_files": len(zip_files),
        "episodes": episodes,
        "parse_errors": parse_errors,
        "train_samples_seen": train_reservoir.seen,
        "valid_samples_seen": valid_reservoir.seen,
    }
    return train_reservoir.items, valid_reservoir.items, stats


def count_scales_for_samples(
    samples: Iterable[BattleSample],
    vocab_size: int,
    card_meta: dict[int, CardMeta],
) -> torch.Tensor:
    scales = torch.full((vocab_size,), 4.0, dtype=torch.float32)
    max_counts: Counter[int] = Counter()
    for sample in samples:
        counts = Counter(sample.deck)
        for card_id, count in counts.items():
            max_counts[card_id] = max(max_counts[card_id], count)
    for card_id, meta in card_meta.items():
        if card_id >= vocab_size:
            continue
        if meta.is_ace_spec:
            scales[card_id] = 1.0
        elif meta.is_basic_energy:
            scales[card_id] = max(1.0, float(max_counts[card_id]))
    return scales


def deck_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    count_scales: torch.Tensor,
    positive_weight: float,
    sum_loss_weight: float,
) -> torch.Tensor:
    element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = 1.0 + (target > 0).float() * positive_weight
    count_loss = (element_loss * weights).mean(dim=1)
    predicted_sum = (torch.relu(prediction) * count_scales).sum(dim=1)
    target_sum = (target * count_scales).sum(dim=1)
    sum_loss = F.smooth_l1_loss(predicted_sum, target_sum, reduction="none")
    return (count_loss + sum_loss_weight * sum_loss).mean()


def resolve_device(option: str) -> torch.device:
    if option == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAを利用できません")
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_completed_deck_overlap(
    model: OpponentDeckMLP,
    samples: list[BattleSample],
    vocab_size: int,
    count_scales: torch.Tensor,
    turn_scale: float,
    known_card_ids: list[int],
    card_meta: dict[int, CardMeta],
    device: torch.device,
) -> float:
    """丸め後の60枚について、正解と一致したカードコピーの割合を返す。"""
    basic_energy_ids = {
        card_id for card_id in known_card_ids if card_meta.get(card_id) and card_meta[card_id].is_basic_energy
    }
    ace_spec_ids = {
        card_id for card_id in known_card_ids if card_meta.get(card_id) and card_meta[card_id].is_ace_spec
    }
    card_names = {
        card_id: card_meta[card_id].name for card_id in known_card_ids if card_id in card_meta
    }
    overlap = 0
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            features = make_features(
                sample.observed,
                sample.turn,
                vocab_size,
                count_scales,
                turn_scale,
            ).to(device)
            predicted_counts = torch.relu(model(features.unsqueeze(0))[0]) * count_scales.to(device)
            deck = complete_deck(
                sample.observed,
                predicted_counts,
                known_card_ids,
                basic_energy_ids,
                ace_spec_ids,
                card_names,
            )
            overlap += sum((Counter(deck) & Counter(sample.deck)).values())
    return overlap / (len(samples) * DECK_SIZE)


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    card_meta = load_card_meta(args.card_data)
    train_samples, valid_samples, source_stats = scan_battle_samples(args)
    all_samples = train_samples + valid_samples
    largest_sample_id = max(
        (max(sample.deck, default=0) for sample in all_samples),
        default=0,
    )
    vocab_size = max(max(card_meta, default=0), largest_sample_id) + 1
    count_scales = count_scales_for_samples(train_samples, vocab_size, card_meta)
    train_dataset = TurnDeckDataset(train_samples, vocab_size, count_scales, args.turn_scale)
    valid_dataset = TurnDeckDataset(valid_samples, vocab_size, count_scales, args.turn_scale)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    device = resolve_device(args.device)
    scales_device = count_scales.to(device)
    model = OpponentDeckMLP(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for features, target in train_loader:
            prediction = model(features.to(device))
            loss = deck_loss(
                prediction,
                target.to(device),
                scales_device,
                args.positive_weight,
                args.sum_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())

        model.eval()
        valid_total = 0.0
        count_mae_total = 0.0
        with torch.inference_mode():
            for features, target in valid_loader:
                target_device = target.to(device)
                prediction = model(features.to(device))
                valid_total += float(
                    deck_loss(
                        prediction,
                        target_device,
                        scales_device,
                        args.positive_weight,
                        args.sum_loss_weight,
                    ).item()
                )
                count_mae_total += float(
                    (torch.relu(prediction) * scales_device - target_device * scales_device)
                    .abs()
                    .mean()
                    .item()
                )
        train_avg = train_total / max(len(train_loader), 1)
        valid_avg = valid_total / max(len(valid_loader), 1)
        count_mae = count_mae_total / max(len(valid_loader), 1)
        print(
            f"epoch={epoch:03d} train_loss={train_avg:.6f} "
            f"valid_loss={valid_avg:.6f} count_mae={count_mae:.6f}",
            flush=True,
        )
        if valid_avg < best_loss:
            best_loss = valid_avg
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    known_card_ids = sorted({int(card_id) for sample in all_samples for card_id in sample.deck})
    completed_deck_overlap = evaluate_completed_deck_overlap(
        model,
        valid_samples,
        vocab_size,
        count_scales,
        args.turn_scale,
        known_card_ids,
        card_meta,
        device,
    )
    print(f"valid_completed_deck_overlap={completed_deck_overlap:.6f}")
    config = {
        "vocab_size": vocab_size,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "turn_scale": args.turn_scale,
        "count_scales": count_scales.tolist(),
        "known_card_ids": known_card_ids,
        "basic_energy_ids": [card_id for card_id in known_card_ids if card_meta.get(card_id) and card_meta[card_id].is_basic_energy],
        "basic_pokemon_ids": [card_id for card_id in known_card_ids if card_meta.get(card_id) and card_meta[card_id].is_basic_pokemon],
        "ace_spec_ids": [card_id for card_id in known_card_ids if card_meta.get(card_id) and card_meta[card_id].is_ace_spec],
        "card_names": {str(card_id): card_meta[card_id].name for card_id in known_card_ids if card_id in card_meta},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": config,
            "metrics": {
                "best_valid_loss": best_loss,
                "valid_completed_deck_overlap": completed_deck_overlap,
                "train_samples": len(train_samples),
                "valid_samples": len(valid_samples),
                "dates": args.date or "all",
                "max_episodes": args.max_episodes,
                "max_samples": args.max_samples,
                **source_stats,
            },
        },
        args.output,
    )
    print(f"saved={args.output} best_valid_loss={best_loss:.6f}")
    return 0


def parse_observed(chunks: list[str]) -> list[int]:
    values: list[int] = []
    for chunk in chunks:
        values.extend(int(value) for value in chunk.replace(",", " ").split())
    return values


def predict(args: argparse.Namespace) -> int:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocab_size = int(config["vocab_size"])
    count_scales = torch.tensor(config["count_scales"], dtype=torch.float32)
    model = OpponentDeckMLP(
        vocab_size,
        int(config["hidden_size"]),
        int(config["layers"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    observed = parse_observed(args.observed)
    features = make_features(observed, args.turn, vocab_size, count_scales, float(config["turn_scale"]))
    with torch.inference_mode():
        prediction = torch.relu(model(features.unsqueeze(0))[0]) * count_scales
    deck = complete_deck(
        observed,
        prediction,
        [int(card_id) for card_id in config["known_card_ids"]],
        {int(card_id) for card_id in config.get("basic_energy_ids", [])},
        {int(card_id) for card_id in config.get("ace_spec_ids", [])},
        {int(card_id): name for card_id, name in config.get("card_names", {}).items()},
    )
    result = {
        "turn": args.turn,
        "observed": observed,
        "deck": deck,
        "deck_counts": dict(sorted(Counter(deck).items())),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n".join(str(card_id) for card_id in deck))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "train":
        return train(args)
    if args.command == "predict":
        return predict(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
