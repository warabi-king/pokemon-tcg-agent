"""Mask deck candidates and train a compatibility-aware MLP deck completer.

The model is trained only from deck_candidates_by_wins.jsonl.  It predicts
cards missing from a partial deck, while a learned cluster posterior suppresses
cards that belong to clearly unrelated deck families.  Mixing cards among
nearby deck variants remains possible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INDEX = SCRIPT_DIR / "generated" / "deck_candidates_by_wins.jsonl"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DEFAULT_MODEL = SCRIPT_DIR / "generated" / "deck_mlp_2.pt"
DECK_SIZE = 60


@dataclass(frozen=True)
class DeckRecord:
    deck: tuple[int, ...]
    cluster_id: int
    wins: int
    games: int
    cluster_weight: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.5


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


class DeckCompletionMLP2(torch.nn.Module):
    """Shared MLP with missing-card presence, count, and cluster heads."""

    def __init__(
        self,
        card_count: int,
        cluster_count: int,
        hidden_size: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        modules: list[torch.nn.Module] = []
        size = card_count + 1
        for _ in range(layers):
            modules.append(torch.nn.Linear(size, hidden_size))
            modules.append(torch.nn.ReLU())
            if dropout > 0:
                modules.append(torch.nn.Dropout(dropout))
            size = hidden_size
        self.body = torch.nn.Sequential(*modules)
        self.presence_head = torch.nn.Linear(size, card_count)
        self.count_head = torch.nn.Linear(size, card_count)
        self.cluster_head = torch.nn.Linear(size, cluster_count)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return (
            self.presence_head(hidden),
            self.count_head(hidden),
            self.cluster_head(hidden),
        )


class MaskedDeckDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Generate deterministic, epoch-dependent copy masks from complete decks."""

    def __init__(
        self,
        records: list[DeckRecord],
        deck_counts: torch.Tensor,
        count_scales: torch.Tensor,
        cluster_to_index: dict[int, int],
        samples_per_deck: int,
        min_observed: int,
        max_observed: int,
        seed: int,
    ) -> None:
        self.records = records
        self.deck_counts = deck_counts
        self.count_scales = count_scales
        self.cluster_to_index = cluster_to_index
        self.samples_per_deck = samples_per_deck
        self.min_observed = min_observed
        self.max_observed = max_observed
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_deck

    def generate(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_index = index // self.samples_per_deck
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        full_counts = self.deck_counts[deck_index]
        positions: list[int] = []
        for card_index, count in enumerate(full_counts.tolist()):
            positions.extend([card_index] * int(count))

        maximum = min(self.max_observed, len(positions) - 1)
        minimum = min(self.min_observed, maximum)
        observed_total = rng.randint(minimum, maximum)
        observed = torch.zeros_like(full_counts)
        for card_index in rng.sample(positions, observed_total):
            observed[card_index] += 1.0
        missing = full_counts - observed
        features = torch.cat(
            (
                observed / self.count_scales,
                torch.tensor([observed_total / DECK_SIZE], dtype=torch.float32),
            )
        )
        cluster_index = torch.tensor(
            self.cluster_to_index[self.records[deck_index].cluster_id],
            dtype=torch.long,
        )
        return features, missing, cluster_index, observed

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, missing, cluster_index, _ = self.generate(index)
        return features, missing, cluster_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train from masked candidate decks.")
    train.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    train.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--hidden-size", type=int, default=256)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--samples-per-deck", type=int, default=16)
    train.add_argument("--min-observed", type=int, default=1)
    train.add_argument("--max-observed", type=int, default=59)
    train.add_argument("--max-decks", type=int)
    train.add_argument("--valid-ratio", type=float, default=0.1)
    train.add_argument("--positive-weight", type=float, default=6.0)
    train.add_argument("--incompatible-weight", type=float, default=3.0)
    train.add_argument("--count-loss-weight", type=float, default=1.0)
    train.add_argument("--cluster-loss-weight", type=float, default=0.5)
    train.add_argument("--sum-loss-weight", type=float, default=0.02)
    train.add_argument("--prior-strength", type=float, default=2.0)
    train.add_argument("--compatibility-strength", type=float, default=1.0)
    train.add_argument("--cooccurrence-strength", type=float, default=1.0)
    train.add_argument("--win-weight", type=float, default=0.15)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=("cpu", "gpu"), default="cpu")

    predict = commands.add_parser("predict", help="Complete a partial deck.")
    predict.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    predict.add_argument("--observed", action="append", default=[])
    predict.add_argument("--observed-file", type=Path)
    predict.add_argument("--json", action="store_true")
    predict.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    return parser.parse_args()


def find_kind_column(fieldnames: list[str]) -> str:
    for name in fieldnames:
        if "Stage" in name and "Type" in name:
            return name
    raise ValueError("could not find card kind column")


def load_card_meta(path: Path) -> dict[int, CardMeta]:
    result: dict[int, CardMeta] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
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


def read_records(path: Path) -> list[DeckRecord]:
    records: list[DeckRecord] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            deck = tuple(int(card_id) for card_id in raw["deck"])
            if len(deck) != DECK_SIZE:
                raise ValueError(f"line {line_number}: deck must contain 60 cards")
            records.append(
                DeckRecord(
                    deck=deck,
                    cluster_id=int(raw.get("cluster_id", 0)),
                    wins=int(raw.get("wins", 0)),
                    games=int(raw.get("games", 1)),
                    cluster_weight=float(raw.get("cluster_weight", 1.0)),
                )
            )
    if not records:
        raise ValueError(f"no decks found in {path}")
    return records


def record_sampling_weight(record: DeckRecord, win_weight: float) -> float:
    return max(record.cluster_weight, 1e-12) * (
        1.0 + win_weight * math.log1p(max(record.wins, 0))
    )


def weighted_sample_records(
    records: list[DeckRecord],
    maximum: int | None,
    win_weight: float,
    seed: int,
) -> list[DeckRecord]:
    if maximum is None or maximum >= len(records):
        return records
    if maximum <= 0:
        raise ValueError("--max-decks must be positive")
    rng = random.Random(seed)
    keyed: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        weight = record_sampling_weight(record, win_weight)
        keyed.append((math.log(max(rng.random(), 1e-12)) / weight, index))
    keyed.sort(reverse=True)
    return [records[index] for _, index in keyed[:maximum]]


def split_records_by_cluster(
    records: list[DeckRecord],
    valid_ratio: float,
    seed: int,
) -> tuple[list[DeckRecord], list[DeckRecord]]:
    if not 0 < valid_ratio < 1:
        raise ValueError("--valid-ratio must be between 0 and 1")
    groups: dict[int, list[DeckRecord]] = defaultdict(list)
    for record in records:
        groups[record.cluster_id].append(record)
    train: list[DeckRecord] = []
    valid: list[DeckRecord] = []
    for cluster_id, group in sorted(groups.items()):
        shuffled = list(group)
        random.Random(seed + cluster_id * 10_007).shuffle(shuffled)
        valid_size = max(1, int(round(len(shuffled) * valid_ratio))) if len(shuffled) > 1 else 0
        valid.extend(shuffled[:valid_size])
        train.extend(shuffled[valid_size:])
    if not valid:
        valid.append(train[-1])
    return train, valid


def build_vocab(records: Iterable[DeckRecord]) -> list[int]:
    return sorted({card_id for record in records for card_id in record.deck})


def records_to_counts(records: list[DeckRecord], card_to_index: dict[int, int]) -> torch.Tensor:
    result = torch.zeros((len(records), len(card_to_index)), dtype=torch.float32)
    for row, record in enumerate(records):
        for card_id in record.deck:
            result[row, card_to_index[card_id]] += 1.0
    return result


def build_count_scales(
    records: list[DeckRecord],
    known_card_ids: list[int],
    card_meta: dict[int, CardMeta],
) -> torch.Tensor:
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    counts = records_to_counts(records, card_to_index)
    maximum = counts.max(dim=0).values
    scales = torch.full((len(known_card_ids),), 4.0, dtype=torch.float32)
    for index, card_id in enumerate(known_card_ids):
        meta = card_meta.get(card_id)
        if meta and meta.is_ace_spec:
            scales[index] = 1.0
        elif meta and meta.is_basic_energy:
            scales[index] = max(1.0, float(maximum[index]))
    return scales


def build_cluster_card_prior(
    records: list[DeckRecord],
    known_card_ids: list[int],
    cluster_ids: list[int],
    prior_strength: float,
) -> torch.Tensor:
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    cluster_to_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    present = torch.zeros((len(cluster_ids), len(known_card_ids)), dtype=torch.float64)
    cluster_total = torch.zeros(len(cluster_ids), dtype=torch.float64)
    global_present = torch.zeros(len(known_card_ids), dtype=torch.float64)
    global_total = 0.0
    for record in records:
        weight = max(record.cluster_weight, 1e-12)
        cluster_index = cluster_to_index[record.cluster_id]
        indices = [card_to_index[card_id] for card_id in set(record.deck)]
        present[cluster_index, indices] += weight
        cluster_total[cluster_index] += weight
        global_present[indices] += weight
        global_total += weight
    global_prior = global_present / max(global_total, 1e-12)
    prior = (
        present + prior_strength * global_prior.unsqueeze(0)
    ) / (cluster_total.unsqueeze(1) + prior_strength)
    return prior.float().clamp(1e-5, 1.0)


def build_card_conditional_prior(
    records: list[DeckRecord],
    known_card_ids: list[int],
    prior_strength: float,
) -> torch.Tensor:
    """Estimate P(candidate card | observed card) from complete decks."""
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    card_count = len(known_card_ids)
    pair = torch.zeros((card_count, card_count), dtype=torch.float64)
    observed_total = torch.zeros(card_count, dtype=torch.float64)
    global_present = torch.zeros(card_count, dtype=torch.float64)
    global_total = 0.0
    for record in records:
        weight = max(record.cluster_weight, 1e-12)
        indices = [card_to_index[card_id] for card_id in set(record.deck)]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        pair[index_tensor.unsqueeze(1), index_tensor.unsqueeze(0)] += weight
        observed_total[index_tensor] += weight
        global_present[index_tensor] += weight
        global_total += weight
    global_prior = global_present / max(global_total, 1e-12)
    conditional = (
        pair + prior_strength * global_prior.unsqueeze(0)
    ) / (observed_total.unsqueeze(1) + prior_strength)
    return conditional.float().clamp(1e-5, 1.0)


def observed_card_compatibility(
    observed_counts: torch.Tensor,
    card_conditional_prior: torch.Tensor,
) -> torch.Tensor:
    """Combine evidence from all observed card types with a geometric mean."""
    batch_result: list[torch.Tensor] = []
    for row in observed_counts:
        indices = torch.nonzero(row > 0, as_tuple=False).flatten()
        if indices.numel() == 0:
            batch_result.append(torch.ones(card_conditional_prior.size(1), device=row.device))
            continue
        priors = card_conditional_prior[indices].clamp_min(1e-5)
        batch_result.append(torch.exp(torch.log(priors).mean(dim=0)))
    return torch.stack(batch_result)


def combine_presence_with_cluster_prior(
    presence_logits: torch.Tensor,
    cluster_logits: torch.Tensor,
    cluster_card_prior: torch.Tensor,
    compatibility_strength: float,
    observed_counts: torch.Tensor | None = None,
    card_conditional_prior: torch.Tensor | None = None,
    cooccurrence_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cluster_probability = torch.softmax(cluster_logits, dim=-1)
    compatibility = cluster_probability @ cluster_card_prior
    if observed_counts is not None and card_conditional_prior is not None:
        cooccurrence = observed_card_compatibility(observed_counts, card_conditional_prior)
        compatibility = compatibility * cooccurrence.pow(cooccurrence_strength)
    presence = torch.sigmoid(presence_logits) * compatibility.pow(compatibility_strength)
    return presence, cluster_probability, compatibility


def completion_loss(
    presence_logits: torch.Tensor,
    normalized_counts: torch.Tensor,
    cluster_logits: torch.Tensor,
    missing_counts: torch.Tensor,
    cluster_targets: torch.Tensor,
    count_scales: torch.Tensor,
    cluster_card_prior: torch.Tensor,
    observed_counts: torch.Tensor,
    card_conditional_prior: torch.Tensor,
    positive_weight: float,
    incompatible_weight: float,
    count_loss_weight: float,
    cluster_loss_weight: float,
    sum_loss_weight: float,
    cooccurrence_strength: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    presence_target = (missing_counts > 0).float()
    true_cluster_prior = cluster_card_prior[cluster_targets]
    cooccurrence = observed_card_compatibility(observed_counts, card_conditional_prior)
    true_compatibility = true_cluster_prior * cooccurrence.pow(cooccurrence_strength)
    negative_weight = 1.0 + incompatible_weight * (1.0 - true_compatibility)
    element_weight = presence_target * positive_weight + (1.0 - presence_target) * negative_weight
    presence_element = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_target,
        reduction="none",
    )
    presence_loss = (presence_element * element_weight).mean()

    predicted_normalized = F.softplus(normalized_counts)
    target_normalized = missing_counts / count_scales
    positive_mask = presence_target
    count_loss = (
        F.smooth_l1_loss(predicted_normalized, target_normalized, reduction="none")
        * positive_mask
    ).sum() / positive_mask.sum().clamp_min(1.0)

    cluster_loss = F.cross_entropy(cluster_logits, cluster_targets)
    expected_counts = torch.sigmoid(presence_logits) * predicted_normalized * count_scales
    sum_loss = F.smooth_l1_loss(
        expected_counts.sum(dim=1),
        missing_counts.sum(dim=1),
    )
    total = (
        presence_loss
        + count_loss_weight * count_loss
        + cluster_loss_weight * cluster_loss
        + sum_loss_weight * sum_loss
    )
    return total, {
        "presence": float(presence_loss.detach()),
        "count": float(count_loss.detach()),
        "cluster": float(cluster_loss.detach()),
        "sum": float(sum_loss.detach()),
    }


def unique_percentile(records: list[DeckRecord], percentile: float) -> int:
    values = sorted(len(set(record.deck)) for record in records)
    index = min(len(values) - 1, max(0, math.ceil(len(values) * percentile) - 1))
    return values[index]


def complete_deck(
    observed_cards: Sequence[int],
    predicted_missing: torch.Tensor,
    presence_scores: torch.Tensor,
    known_card_ids: list[int],
    card_meta: dict[int, CardMeta],
    min_unique_cards: int,
    max_unique_cards: int,
) -> list[int]:
    if len(observed_cards) > DECK_SIZE:
        raise ValueError("observed deck contains more than 60 cards")
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    unknown = sorted({int(card_id) for card_id in observed_cards if int(card_id) not in card_to_index})
    if unknown:
        raise ValueError(f"cards were not present in training data: {unknown}")

    deck = [int(card_id) for card_id in observed_cards]
    observed_ids = set(deck)
    predicted_unique = int((presence_scores >= 0.5).sum()) + len(observed_ids)
    unique_limit = min(
        max(max_unique_cards, len(observed_ids)),
        max(min_unique_cards, predicted_unique, len(observed_ids)),
    )
    ranked_indices = sorted(
        range(len(known_card_ids)),
        key=lambda index: (
            float(presence_scores[index]),
            float(predicted_missing[index]),
            -known_card_ids[index],
        ),
        reverse=True,
    )
    selected = {card_to_index[card_id] for card_id in observed_ids}
    for index in ranked_indices:
        if len(selected) >= unique_limit:
            break
        selected.add(index)

    basic_energy_indices = {
        index
        for index, card_id in enumerate(known_card_ids)
        if card_meta.get(card_id) and card_meta[card_id].is_basic_energy
    }
    energy = next((index for index in ranked_indices if index in basic_energy_indices), None)
    if energy is not None and energy not in selected:
        removable = next(
            (
                index
                for index in reversed(ranked_indices)
                if index in selected and known_card_ids[index] not in observed_ids
            ),
            None,
        )
        if removable is not None and len(selected) >= unique_limit:
            selected.remove(removable)
        selected.add(energy)

    counts = Counter(deck)
    added = Counter()
    name_counts = Counter(
        card_meta.get(card_id).name if card_meta.get(card_id) else str(card_id)
        for card_id in deck
    )
    ace_count = sum(
        count
        for card_id, count in counts.items()
        if card_meta.get(card_id) and card_meta[card_id].is_ace_spec
    )
    candidates = [index for index in ranked_indices if index in selected]
    while len(deck) < DECK_SIZE:
        best_index: int | None = None
        best_score = float("-inf")
        for index in candidates:
            card_id = known_card_ids[index]
            meta = card_meta.get(card_id)
            name = meta.name if meta else str(card_id)
            if meta and meta.is_ace_spec and ace_count >= 1:
                continue
            if index not in basic_energy_indices and name_counts[name] >= 4:
                continue
            residual = float(predicted_missing[index]) - added[card_id]
            if residual > best_score:
                best_score = residual
                best_index = index
        if best_index is None:
            if energy is None:
                raise ValueError("could not complete a legal 60-card deck")
            best_index = energy
        card_id = known_card_ids[best_index]
        meta = card_meta.get(card_id)
        deck.append(card_id)
        counts[card_id] += 1
        added[card_id] += 1
        name_counts[meta.name if meta else str(card_id)] += 1
        if meta and meta.is_ace_spec:
            ace_count += 1
    return deck


def resolve_device(option: str) -> torch.device:
    if option == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_model(
    model: DeckCompletionMLP2,
    dataset: MaskedDeckDataset,
    records: list[DeckRecord],
    known_card_ids: list[int],
    count_scales: torch.Tensor,
    cluster_card_prior: torch.Tensor,
    card_conditional_prior: torch.Tensor,
    compatibility_strength: float,
    cooccurrence_strength: float,
    card_meta: dict[int, CardMeta],
    min_unique_cards: int,
    max_unique_cards: int,
    device: torch.device,
) -> dict[str, float]:
    full_overlap = 0
    hidden_overlap = 0
    hidden_total = 0
    exact = 0
    foreign = 0
    predicted_hidden_total = 0
    unique_total = 0
    singleton_total = 0
    model.eval()
    scales_device = count_scales.to(device)
    prior_device = cluster_card_prior.to(device)
    conditional_device = card_conditional_prior.to(device)
    with torch.inference_mode():
        for index in range(len(dataset)):
            features, _, _, observed_counts = dataset.generate(index)
            presence_logits, normalized_counts, cluster_logits = model(features.unsqueeze(0).to(device))
            presence, _, _ = combine_presence_with_cluster_prior(
                presence_logits,
                cluster_logits,
                prior_device,
                compatibility_strength,
                observed_counts.unsqueeze(0).to(device),
                conditional_device,
                cooccurrence_strength,
            )
            cooccurrence = observed_card_compatibility(
                observed_counts.unsqueeze(0).to(device),
                conditional_device,
            )[0].cpu()
            predicted_missing = F.softplus(normalized_counts[0]) * scales_device
            observed_cards = [
                known_card_ids[card_index]
                for card_index, count in enumerate(observed_counts.tolist())
                for _ in range(int(count))
            ]
            deck = complete_deck(
                observed_cards,
                predicted_missing.cpu(),
                presence[0].cpu(),
                known_card_ids,
                card_meta,
                min_unique_cards,
                max_unique_cards,
            )
            record = records[index // dataset.samples_per_deck]
            truth = Counter(record.deck)
            predicted = Counter(deck)
            observed = Counter(observed_cards)
            true_hidden = truth - observed
            predicted_hidden = predicted - observed
            full_overlap += sum((truth & predicted).values())
            hidden_overlap += sum((true_hidden & predicted_hidden).values())
            hidden_total += sum(true_hidden.values())
            exact += truth == predicted
            unique_total += len(predicted)
            singleton_total += sum(count == 1 for count in predicted.values())
            for card_id, count in predicted_hidden.items():
                predicted_hidden_total += count
                if cooccurrence[known_card_ids.index(card_id)] < 0.05:
                    foreign += count
    sample_count = len(dataset)
    return {
        "full_copy_overlap": full_overlap / (sample_count * DECK_SIZE),
        "hidden_copy_overlap": hidden_overlap / max(hidden_total, 1),
        "exact_deck_rate": exact / sample_count,
        "foreign_card_rate": foreign / max(predicted_hidden_total, 1),
        "mean_unique_cards": unique_total / sample_count,
        "mean_singleton_cards": singleton_total / sample_count,
    }


def train(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not 1 <= args.min_observed < args.max_observed < DECK_SIZE:
        raise ValueError("observed range must satisfy 1 <= min < max < 60")
    all_records = read_records(args.index)
    records = weighted_sample_records(all_records, args.max_decks, args.win_weight, args.seed)
    train_records, valid_records = split_records_by_cluster(records, args.valid_ratio, args.seed)
    known_card_ids = build_vocab(records)
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    cluster_ids = sorted({record.cluster_id for record in records})
    cluster_to_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    card_meta = load_card_meta(args.card_data)
    count_scales = build_count_scales(train_records, known_card_ids, card_meta)
    cluster_card_prior = build_cluster_card_prior(
        train_records,
        known_card_ids,
        cluster_ids,
        args.prior_strength,
    )
    card_conditional_prior = build_card_conditional_prior(
        train_records,
        known_card_ids,
        args.prior_strength,
    )
    train_counts = records_to_counts(train_records, card_to_index)
    valid_counts = records_to_counts(valid_records, card_to_index)
    train_dataset = MaskedDeckDataset(
        train_records,
        train_counts,
        count_scales,
        cluster_to_index,
        args.samples_per_deck,
        args.min_observed,
        args.max_observed,
        args.seed,
    )
    valid_dataset = MaskedDeckDataset(
        valid_records,
        valid_counts,
        count_scales,
        cluster_to_index,
        max(1, args.samples_per_deck // 2),
        args.min_observed,
        args.max_observed,
        args.seed + 10_000_000,
    )
    sample_weights = [
        record_sampling_weight(record, args.win_weight)
        for record in train_records
        for _ in range(args.samples_per_deck)
    ]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    device = resolve_device(args.device)
    scales_device = count_scales.to(device)
    prior_device = cluster_card_prior.to(device)
    conditional_device = card_conditional_prior.to(device)
    model = DeckCompletionMLP2(
        len(known_card_ids),
        len(cluster_ids),
        args.hidden_size,
        args.layers,
        args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        train_total = 0.0
        for features, missing, cluster_targets in train_loader:
            features_device = features.to(device)
            observed_counts = features_device[:, :-1] * scales_device
            presence, counts, clusters = model(features_device)
            loss, _ = completion_loss(
                presence,
                counts,
                clusters,
                missing.to(device),
                cluster_targets.to(device),
                scales_device,
                prior_device,
                observed_counts,
                conditional_device,
                args.positive_weight,
                args.incompatible_weight,
                args.count_loss_weight,
                args.cluster_loss_weight,
                args.sum_loss_weight,
                args.cooccurrence_strength,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())

        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for features, missing, cluster_targets in valid_loader:
                features_device = features.to(device)
                observed_counts = features_device[:, :-1] * scales_device
                presence, counts, clusters = model(features_device)
                loss, _ = completion_loss(
                    presence,
                    counts,
                    clusters,
                    missing.to(device),
                    cluster_targets.to(device),
                    scales_device,
                    prior_device,
                    observed_counts,
                    conditional_device,
                    args.positive_weight,
                    args.incompatible_weight,
                    args.count_loss_weight,
                    args.cluster_loss_weight,
                    args.sum_loss_weight,
                    args.cooccurrence_strength,
                )
                valid_total += float(loss.item())
        train_loss = train_total / max(len(train_loader), 1)
        valid_loss = valid_total / max(len(valid_loader), 1)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)

    min_unique_cards = unique_percentile(train_records, 0.05)
    max_unique_cards = unique_percentile(train_records, 0.95)
    metrics = evaluate_model(
        model,
        valid_dataset,
        valid_records,
        known_card_ids,
        count_scales,
        cluster_card_prior,
        card_conditional_prior,
        args.compatibility_strength,
        args.cooccurrence_strength,
        card_meta,
        min_unique_cards,
        max_unique_cards,
        device,
    )
    metrics.update(
        {
            "best_valid_loss": best_valid,
            "source_decks": len(all_records),
            "used_decks": len(records),
            "train_decks": len(train_records),
            "valid_decks": len(valid_records),
        }
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "config": {
                "architecture": "deck_completion_mlp_2",
                "known_card_ids": known_card_ids,
                "cluster_ids": cluster_ids,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "count_scales": count_scales.tolist(),
                "cluster_card_prior": cluster_card_prior.tolist(),
                "card_conditional_prior": card_conditional_prior.tolist(),
                "compatibility_strength": args.compatibility_strength,
                "cooccurrence_strength": args.cooccurrence_strength,
                "min_unique_cards": min_unique_cards,
                "max_unique_cards": max_unique_cards,
                "basic_energy_ids": [
                    card_id
                    for card_id in known_card_ids
                    if card_meta.get(card_id) and card_meta[card_id].is_basic_energy
                ],
                "ace_spec_ids": [
                    card_id
                    for card_id in known_card_ids
                    if card_meta.get(card_id) and card_meta[card_id].is_ace_spec
                ],
                "card_names": {
                    str(card_id): card_meta[card_id].name
                    for card_id in known_card_ids
                    if card_id in card_meta
                },
            },
            "metrics": metrics,
        },
        args.output,
    )
    print(f"saved={args.output}")
    return 0


def parse_observed(args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    for chunk in args.observed:
        values.extend(int(value) for value in chunk.replace(",", " ").split())
    if args.observed_file:
        values.extend(
            int(line.strip())
            for line in args.observed_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return values


def card_meta_from_config(config: dict[str, Any]) -> dict[int, CardMeta]:
    basic = {int(card_id) for card_id in config.get("basic_energy_ids", [])}
    ace = {int(card_id) for card_id in config.get("ace_spec_ids", [])}
    names = {int(card_id): name for card_id, name in config.get("card_names", {}).items()}
    result: dict[int, CardMeta] = {}
    for card_id in config["known_card_ids"]:
        card_id = int(card_id)
        result[card_id] = CardMeta(
            card_id,
            names.get(card_id, str(card_id)),
            "Basic Energy" if card_id in basic else "",
            "ACE SPEC" if card_id in ace else "",
        )
    return result


def predict(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    known_card_ids = [int(card_id) for card_id in config["known_card_ids"]]
    cluster_ids = [int(cluster_id) for cluster_id in config["cluster_ids"]]
    model = DeckCompletionMLP2(
        len(known_card_ids),
        len(cluster_ids),
        int(config["hidden_size"]),
        int(config["layers"]),
        float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scales = torch.tensor(config["count_scales"], dtype=torch.float32, device=device)
    prior = torch.tensor(config["cluster_card_prior"], dtype=torch.float32, device=device)
    conditional_prior = torch.tensor(
        config["card_conditional_prior"],
        dtype=torch.float32,
        device=device,
    )
    card_to_index = {card_id: index for index, card_id in enumerate(known_card_ids)}
    observed_cards = parse_observed(args)
    observed_counts = torch.zeros(len(known_card_ids), dtype=torch.float32)
    unknown = sorted({card_id for card_id in observed_cards if card_id not in card_to_index})
    if unknown:
        raise ValueError(f"cards were not present in training data: {unknown}")
    for card_id in observed_cards:
        observed_counts[card_to_index[card_id]] += 1.0
    features = torch.cat(
        (
            observed_counts.to(device) / scales,
            torch.tensor([len(observed_cards) / DECK_SIZE], device=device),
        )
    )
    with torch.inference_mode():
        presence_logits, normalized_counts, cluster_logits = model(features.unsqueeze(0))
        presence, cluster_probability, compatibility = combine_presence_with_cluster_prior(
            presence_logits,
            cluster_logits,
            prior,
            float(config["compatibility_strength"]),
            observed_counts.unsqueeze(0).to(device),
            conditional_prior,
            float(config["cooccurrence_strength"]),
        )
        predicted_missing = F.softplus(normalized_counts[0]) * scales
    card_meta = card_meta_from_config(config)
    deck = complete_deck(
        observed_cards,
        predicted_missing.cpu(),
        presence[0].cpu(),
        known_card_ids,
        card_meta,
        int(config["min_unique_cards"]),
        int(config["max_unique_cards"]),
    )
    top_clusters = sorted(
        range(len(cluster_ids)),
        key=lambda index: float(cluster_probability[0, index]),
        reverse=True,
    )[:5]
    output = {
        "observed": observed_cards,
        "deck": deck,
        "deck_counts": dict(sorted(Counter(deck).items())),
        "top_clusters": [
            {
                "cluster_id": cluster_ids[index],
                "probability": float(cluster_probability[0, index]),
            }
            for index in top_clusters
        ],
        "mean_selected_compatibility": sum(
            float(compatibility[0, card_to_index[card_id]]) * count
            for card_id, count in Counter(deck).items()
        ) / DECK_SIZE,
    }
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
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
