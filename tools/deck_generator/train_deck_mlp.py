from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INDEX = SCRIPT_DIR / "generated" / "deck_candidates.jsonl"
DEFAULT_WIN_INDEX = SCRIPT_DIR / "generated" / "deck_candidates_by_wins.jsonl"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DEFAULT_MODEL = SCRIPT_DIR / "generated" / "deck_mlp.pt"
DECK_SIZE = 60
DEFAULT_CARD_SCALE = 4.0


@dataclass(frozen=True)
class DeckRecord:
    deck: list[int]
    win_rate: float = 0.5
    games: int = 1
    wins: int = 0
    cluster_wins: int = 0
    cluster_weight: float = 1.0


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
    def is_ace_spec(self) -> bool:
        return "ACE SPEC" in f"{self.name} {self.rule}".upper()


class DeckMLP(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, layers: int, dropout: float):
        super().__init__()
        modules: list[torch.nn.Module] = []
        size = input_size
        for _ in range(layers):
            modules.append(torch.nn.Linear(size, hidden_size))
            modules.append(torch.nn.ReLU())
            if dropout > 0:
                modules.append(torch.nn.Dropout(dropout))
            size = hidden_size
        modules.append(torch.nn.Linear(size, output_size))
        self.net = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PartialDeckDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        deck_counts: torch.Tensor,
        count_scales: torch.Tensor,
        deck_weights: torch.Tensor,
        samples_per_deck: int,
        min_observed: int,
        max_observed: int,
        seed: int,
    ) -> None:
        self.deck_counts = deck_counts
        self.count_scales = count_scales
        self.deck_weights = deck_weights
        self.samples_per_deck = samples_per_deck
        self.min_observed = min_observed
        self.max_observed = max_observed
        self.seed = seed

    def __len__(self) -> int:
        return self.deck_counts.size(0) * self.samples_per_deck

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_index = index // self.samples_per_deck
        rng = random.Random(self.seed + index)
        target = self.deck_counts[deck_index]
        card_positions: list[int] = []
        for card_id, count in enumerate(target.tolist()):
            card_positions.extend([card_id] * int(round(count)))

        observed_total = rng.randint(self.min_observed, min(self.max_observed, len(card_positions)))
        observed_cards = rng.sample(card_positions, observed_total)
        observed = torch.zeros_like(target)
        for card_id in observed_cards:
            observed[card_id] += 1.0

        observed_feature = torch.cat(
            [
                observed / self.count_scales,
                torch.tensor([observed_total / DECK_SIZE], dtype=torch.float32),
            ]
        )
        return observed_feature, target / self.count_scales, self.deck_weights[deck_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or query an MLP deck completion model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train MLP from deck candidate JSONL.")
    train.add_argument("--index", type=Path, default=DEFAULT_WIN_INDEX)
    train.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden-size", type=int, default=512)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--positive-weight", type=float, default=8.0)
    train.add_argument("--sum-loss-weight", type=float, default=0.05)
    train.add_argument("--samples-per-deck", type=int, default=4)
    train.add_argument("--min-observed", type=int, default=1)
    train.add_argument("--max-observed", type=int, default=48)
    train.add_argument("--max-decks", type=int, help="Use at most this many decks from the index.")
    train.add_argument(
        "--win-rate-weight",
        type=float,
        default=0.0,
        help="Increase loss weight for high win-rate decks with log1p(win_rate). 0 disables weighting.",
    )
    train.add_argument(
        "--deck-sampling",
        choices=["cluster-weight", "cluster-wins"],
        default="cluster-weight",
        help=(
            "How to sample decks. "
            "cluster-weight uses cluster_weight; cluster-wins also multiplies by 1 + log1p(cluster_wins)."
        ),
    )
    train.add_argument("--valid-ratio", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    predict = subparsers.add_parser("predict", help="Predict a completed 60-card deck from observed cards.")
    predict.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--observed", action="append", default=[], help="Comma-separated observed card IDs.")
    predict.add_argument("--observed-file", type=Path, help="File containing one observed card ID per line.")
    predict.add_argument("--json", action="store_true")
    predict.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    return parser.parse_args()


def find_kind_column(fieldnames: list[str]) -> str:
    for name in fieldnames:
        if "Stage" in name and "Type" in name:
            return name
    raise ValueError("could not find card kind column in card data CSV")


def load_card_meta(path: Path) -> dict[int, CardMeta]:
    if not path.exists():
        return {}
    result: dict[int, CardMeta] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        kind_column = find_kind_column(reader.fieldnames or [])
        for row in reader:
            card_id = int(row["Card ID"])
            result[card_id] = CardMeta(
                card_id=card_id,
                name=row["Card Name"],
                kind=row[kind_column],
                rule=row.get("Rule", ""),
            )
    return result


def read_deck_candidates(path: Path) -> list[DeckRecord]:
    records: list[DeckRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            deck = raw["deck"]
            if len(deck) != DECK_SIZE:
                raise ValueError(f"deck length must be 60: {raw.get('episode_file')}")
            records.append(
                DeckRecord(
                    deck=[int(card_id) for card_id in deck],
                    win_rate=float(raw.get("win_rate", 0.5)),
                    games=int(raw.get("games", 1)),
                    wins=int(raw.get("wins", 0)),
                    cluster_wins=int(raw.get("cluster_wins", raw.get("wins", 0))),
                    cluster_weight=float(raw.get("cluster_weight", 1.0)),
                )
            )
    if not records:
        raise ValueError(f"no decks found in {path}")
    return records


def sample_records(
    records: list[DeckRecord],
    max_decks: int | None,
    seed: int,
    sampling: str,
) -> list[DeckRecord]:
    if max_decks is None or max_decks >= len(records):
        return records
    if max_decks <= 0:
        raise ValueError("--max-decks must be positive")

    weights = [deck_sampling_weight(record, sampling) for record in records]
    indices = weighted_sample_without_replacement(weights, max_decks, seed)
    return [records[index] for index in indices]


def deck_sampling_weight(record: DeckRecord, sampling: str) -> float:
    weight = max(record.cluster_weight, 1e-12)
    if sampling == "cluster-wins":
        weight *= 1.0 + math.log1p(max(record.cluster_wins, 0))
    return weight


def weighted_sample_without_replacement(weights: list[float], sample_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    keyed_indices: list[tuple[float, int]] = []
    for index, weight in enumerate(weights):
        safe_weight = max(float(weight), 1e-12)
        u = max(rng.random(), 1e-12)
        keyed_indices.append((math.log(u) / safe_weight, index))
    keyed_indices.sort(reverse=True)
    return [index for _, index in keyed_indices[:sample_size]]


def build_card_id_set(decks: list[list[int]]) -> list[int]:
    card_ids: set[int] = set()
    for deck in decks:
        card_ids.update(deck)
    return sorted(card_ids)


def decks_to_tensor(decks: list[list[int]], vocab_size: int) -> torch.Tensor:
    rows = torch.zeros((len(decks), vocab_size), dtype=torch.float32)
    for row_index, deck in enumerate(decks):
        for card_id in deck:
            rows[row_index, card_id] += 1.0
    return rows


def build_count_scales(
    card_meta: dict[int, CardMeta],
    deck_counts: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    scales = torch.full((vocab_size,), DEFAULT_CARD_SCALE, dtype=torch.float32)
    max_counts = deck_counts.max(dim=0).values if deck_counts.numel() else torch.zeros(vocab_size, dtype=torch.float32)
    for card_id in range(vocab_size):
        meta = card_meta.get(card_id)
        if meta and meta.is_ace_spec:
            scales[card_id] = 1.0
        elif meta and meta.is_basic_energy:
            scales[card_id] = max(1.0, float(max_counts[card_id].item()))
    return scales


def count_scales_from_config(config: dict[str, Any], vocab_size: int) -> torch.Tensor:
    raw_scales = config.get("count_scales")
    if raw_scales:
        scales = torch.tensor(raw_scales, dtype=torch.float32)
        if scales.numel() < vocab_size:
            padding = torch.full((vocab_size - scales.numel(),), DEFAULT_CARD_SCALE, dtype=torch.float32)
            scales = torch.cat([scales, padding])
        return scales[:vocab_size].clamp_min(1.0)
    legacy_scale = float(config.get("count_scale", DEFAULT_CARD_SCALE))
    return torch.full((vocab_size,), legacy_scale, dtype=torch.float32)


def weights_to_tensor(
    records: list[DeckRecord],
    win_rate_weight: float,
) -> torch.Tensor:
    weights = torch.ones(len(records), dtype=torch.float32)
    for index, record in enumerate(records):
        weight = 1.0
        if win_rate_weight > 0:
            weight *= 1.0 + win_rate_weight * math.log1p(max(0.0, min(1.0, record.win_rate)))
        weights[index] = weight
    return weights


def split_records(
    records: list[DeckRecord],
    valid_ratio: float,
    seed: int,
) -> tuple[list[DeckRecord], list[DeckRecord]]:
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    valid_size = max(1, int(len(indices) * valid_ratio))
    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]
    if not train_indices:
        train_indices, valid_indices = indices, indices
    return [records[index] for index in train_indices], [records[index] for index in valid_indices]


def dataset_sampling_weights(
    records: list[DeckRecord],
    samples_per_deck: int,
    sampling: str,
) -> torch.Tensor:
    weights: list[float] = []
    for record in records:
        weights.extend([deck_sampling_weight(record, sampling)] * samples_per_deck)
    return torch.tensor(weights, dtype=torch.double)


def resolve_device(device: str) -> torch.device:
    if device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("--device gpu was specified, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def deck_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    count_scales: torch.Tensor,
    sample_weight: torch.Tensor,
    positive_weight: float,
    sum_loss_weight: float,
) -> torch.Tensor:
    element_loss = F.smooth_l1_loss(pred, target, reduction="none")
    weights = torch.ones_like(target)
    weights = weights + (target > 0).float() * positive_weight
    count_loss_by_sample = (element_loss * weights).mean(dim=1)
    pred_sum = (torch.relu(pred) * count_scales).sum(dim=1)
    target_sum = (target * count_scales).sum(dim=1)
    sum_loss_by_sample = F.smooth_l1_loss(pred_sum, target_sum, reduction="none")
    loss_by_sample = count_loss_by_sample + sum_loss_by_sample * sum_loss_weight
    return (loss_by_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def train_model(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    card_meta = load_card_meta(args.card_data)
    all_records = read_deck_candidates(args.index)
    records = sample_records(all_records, args.max_decks, args.seed, args.deck_sampling)
    train_records, valid_records = split_records(records, args.valid_ratio, args.seed)
    decks = [record.deck for record in records]
    known_card_ids = build_card_id_set(decks)
    vocab_size = max(max(known_card_ids), max(card_meta, default=0)) + 1
    all_counts = decks_to_tensor(decks, vocab_size)
    count_scales = build_count_scales(card_meta, all_counts, vocab_size)
    train_counts = decks_to_tensor([record.deck for record in train_records], vocab_size)
    valid_counts = decks_to_tensor([record.deck for record in valid_records], vocab_size)
    train_weights = weights_to_tensor(train_records, args.win_rate_weight)
    valid_weights = weights_to_tensor(valid_records, args.win_rate_weight)

    train_dataset = PartialDeckDataset(
        train_counts,
        count_scales,
        train_weights,
        samples_per_deck=args.samples_per_deck,
        min_observed=args.min_observed,
        max_observed=args.max_observed,
        seed=args.seed,
    )
    valid_dataset = PartialDeckDataset(
        valid_counts,
        count_scales,
        valid_weights,
        samples_per_deck=max(1, args.samples_per_deck // 2),
        min_observed=args.min_observed,
        max_observed=args.max_observed,
        seed=args.seed + 1_000_000,
    )
    train_sampler = WeightedRandomSampler(
        weights=dataset_sampling_weights(train_records, args.samples_per_deck, args.deck_sampling),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    device = resolve_device(args.device)
    count_scales_device = count_scales.to(device)
    model = DeckMLP(
        input_size=vocab_size + 1,
        output_size=vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for x, y, sample_weight in train_loader:
            x = x.to(device)
            y = y.to(device)
            sample_weight = sample_weight.to(device)
            pred = model(x)
            loss = deck_loss(pred, y, count_scales_device, sample_weight, args.positive_weight, args.sum_loss_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        with torch.no_grad():
            for x, y, sample_weight in valid_loader:
                x = x.to(device)
                y = y.to(device)
                sample_weight = sample_weight.to(device)
                pred = model(x)
                loss = deck_loss(pred, y, count_scales_device, sample_weight, args.positive_weight, args.sum_loss_weight)
                valid_loss += float(loss.item())
                valid_batches += 1

        train_avg = train_loss / max(train_batches, 1)
        valid_avg = valid_loss / max(valid_batches, 1)
        print(f"epoch {epoch:03d} train_loss={train_avg:.6f} valid_loss={valid_avg:.6f}")
        if valid_avg < best_valid:
            best_valid = valid_avg
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "count_scales": [float(value) for value in count_scales.tolist()],
                "positive_weight": args.positive_weight,
                "sum_loss_weight": args.sum_loss_weight,
                "win_rate_weight": args.win_rate_weight,
                "deck_sampling": args.deck_sampling,
                "device_option": args.device,
                "known_card_ids": known_card_ids,
                "card_meta": {
                    str(card_id): {
                        "name": meta.name,
                        "kind": meta.kind,
                        "rule": meta.rule,
                    }
                    for card_id, meta in card_meta.items()
                },
            },
            "metrics": {
                "best_valid_loss": best_valid,
                "source_decks": len(all_records),
                "used_decks": len(records),
                "train_decks": int(train_counts.size(0)),
                "valid_decks": int(valid_counts.size(0)),
                "mean_win_rate": sum(record.win_rate for record in records) / len(records),
                "mean_cluster_weight": sum(record.cluster_weight for record in records) / len(records),
                "mean_cluster_wins": sum(record.cluster_wins for record in records) / len(records),
                "mean_loss_weight": float(train_weights.mean().item()),
                "mean_deck_sampling_weight": sum(deck_sampling_weight(record, args.deck_sampling) for record in records)
                / len(records),
                "min_count_scale": float(count_scales.min().item()),
                "max_count_scale": float(count_scales.max().item()),
            },
        },
        args.output,
    )
    print(f"saved model to {args.output}")
    return 0


def parse_observed(args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    for chunk in args.observed:
        for item in chunk.replace(",", " ").split():
            if item:
                values.append(int(item))
    if args.observed_file:
        for line in args.observed_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                values.append(int(text))
    return values


def meta_from_checkpoint(config: dict[str, Any]) -> dict[int, CardMeta]:
    result: dict[int, CardMeta] = {}
    for card_id_text, raw in config.get("card_meta", {}).items():
        card_id = int(card_id_text)
        result[card_id] = CardMeta(
            card_id=card_id,
            name=raw["name"],
            kind=raw["kind"],
            rule=raw.get("rule", ""),
        )
    return result


def allowed_to_add(card_id: int, deck: list[int], card_meta: dict[int, CardMeta]) -> bool:
    meta = card_meta.get(card_id)
    if meta and meta.is_ace_spec:
        return not any(card_meta.get(existing) and card_meta[existing].is_ace_spec for existing in deck)
    if meta and meta.is_basic_energy:
        return True
    name = meta.name if meta else str(card_id)
    return sum(1 for existing in deck if (card_meta.get(existing).name if card_meta.get(existing) else str(existing)) == name) < 4


def complete_from_prediction(
    observed_cards: list[int],
    predicted_counts: torch.Tensor,
    known_card_ids: list[int],
    card_meta: dict[int, CardMeta],
) -> list[int]:
    deck = list(observed_cards[:DECK_SIZE])
    observed_counts = Counter(deck)
    scores = predicted_counts.detach().cpu().tolist()

    for card_id, observed_count in observed_counts.items():
        if card_id < len(scores):
            scores[card_id] = max(scores[card_id], float(observed_count))

    while len(deck) < DECK_SIZE:
        best_card = None
        best_value = -1e9
        for card_id in known_card_ids:
            if card_id >= len(scores) or not allowed_to_add(card_id, deck, card_meta):
                continue
            value = scores[card_id] - deck.count(card_id) * 0.35
            if value > best_value:
                best_value = value
                best_card = card_id
        if best_card is None:
            best_card = 3
        deck.append(best_card)
    return deck


def predict(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    vocab_size = int(config["vocab_size"])
    count_scales = count_scales_from_config(config, vocab_size).to(device)
    model = DeckMLP(
        input_size=vocab_size + 1,
        output_size=vocab_size,
        hidden_size=int(config["hidden_size"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    observed_cards = parse_observed(args)
    observed = torch.zeros(vocab_size, dtype=torch.float32)
    for card_id in observed_cards:
        if 0 <= card_id < vocab_size:
            observed[card_id] += 1.0
    x = torch.cat([observed.to(device) / count_scales, torch.tensor([len(observed_cards) / DECK_SIZE], device=device)])
    with torch.no_grad():
        predicted = torch.clamp(model(x.unsqueeze(0))[0] * count_scales, min=0.0)

    card_meta = meta_from_checkpoint(config)
    known_card_ids = [int(card_id) for card_id in config["known_card_ids"]]
    deck = complete_from_prediction(observed_cards, predicted, known_card_ids, card_meta)
    output = {
        "observed": observed_cards,
        "deck": deck,
        "deck_counts": {str(card_id): count for card_id, count in sorted(Counter(deck).items())},
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("\n".join(str(card_id) for card_id in deck))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "train":
        return train_model(args)
    if args.command == "predict":
        return predict(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
