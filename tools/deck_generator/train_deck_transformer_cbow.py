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
DEFAULT_INDEX = SCRIPT_DIR / "generated" / "deck_candidates_by_wins.jsonl"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DEFAULT_WORD2VEC = SCRIPT_DIR / "generated" / "deck_word2vec.pt"
DEFAULT_MODEL = SCRIPT_DIR / "generated" / "deck_transformer_cbow.pt"
DECK_SIZE = 60


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


class DeckTransformerCBOW(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        heads: int,
        layers: int,
        ff_dim: int,
        dropout: float,
        pad_id: int,
        bos_id: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.token_embedding = torch.nn.Embedding(vocab_size + 2, embedding_dim, padding_idx=pad_id)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output = torch.nn.Linear(embedding_dim + 1, vocab_size)

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(token_ids)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        card_count = ((token_ids != self.pad_id) & (token_ids != self.bos_id)).sum(dim=1, keepdim=True).float()
        context_ratio = (card_count / DECK_SIZE).clamp(max=1.0)
        return self.output(torch.cat([pooled, context_ratio], dim=1))


class DeckTransformerDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[DeckRecord],
        max_context: int,
        pad_id: int,
        bos_id: int,
        samples_per_deck: int,
        min_context: int,
        deck_loss_weights: torch.Tensor,
        seed: int,
    ) -> None:
        self.records = records
        self.max_context = max_context
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.samples_per_deck = samples_per_deck
        self.min_context = min_context
        self.deck_loss_weights = deck_loss_weights
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_deck

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deck_index = index // self.samples_per_deck
        rng = random.Random(self.seed + index)
        deck = self.records[deck_index].deck

        target_pos = rng.randrange(len(deck))
        target_card = deck[target_pos]
        context_pool = deck[:target_pos] + deck[target_pos + 1 :]
        context_size = rng.randint(self.min_context, min(self.max_context, len(context_pool)))
        context_cards = rng.sample(context_pool, context_size)

        tokens = [self.bos_id] + context_cards
        max_tokens = self.max_context + 1
        token_ids = torch.full((max_tokens,), self.pad_id, dtype=torch.long)
        token_ids[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        padding_mask = token_ids == self.pad_id
        return (
            token_ids,
            padding_mask,
            torch.tensor(target_card, dtype=torch.long),
            self.deck_loss_weights[deck_index],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or query a CBOW Transformer deck generator. "
            "The model uses card embeddings from deck_word2vec.pt, no positional encoding, "
            "TransformerEncoder pooling, and next-card classification."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train CBOW Transformer from deck candidate JSONL.")
    train.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    train.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    train.add_argument("--word2vec-checkpoint", type=Path, default=DEFAULT_WORD2VEC)
    train.add_argument("--random-init", action="store_true", help="Do not require a word2vec checkpoint.")
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--embedding-dim", type=int, default=0, help="0 uses the word2vec embedding dimension.")
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--ff-dim", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--samples-per-deck", type=int, default=8)
    train.add_argument("--min-context", type=int, default=1)
    train.add_argument("--max-context", type=int, default=59)
    train.add_argument("--max-decks", type=int, help="Use at most this many decks from the index.")
    train.add_argument("--valid-ratio", type=float, default=0.1)
    train.add_argument("--win-rate-weight", type=float, default=0.0)
    train.add_argument(
        "--deck-sampling",
        choices=["cluster-weight", "cluster-wins"],
        default="cluster-weight",
        help=(
            "How to sample decks. "
            "cluster-weight uses cluster_weight; cluster-wins also multiplies by 1 + log1p(cluster_wins)."
        ),
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    generate = subparsers.add_parser("generate", help="Generate a 60-card deck from observed card IDs.")
    add_generation_args(generate)
    predict = subparsers.add_parser("predict", help="Alias of generate.")
    add_generation_args(predict)
    return parser.parse_args()


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--observed", action="append", default=[], help="Comma-separated observed/seed card IDs.")
    parser.add_argument("--observed-file", type=Path, help="File containing one observed card ID per line.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 uses greedy generation; >0 samples.")
    parser.add_argument("--top-k", type=int, default=20, help="Limit sampling candidates when temperature > 0.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")


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


def build_card_id_set(decks: list[list[int]]) -> list[int]:
    card_ids: set[int] = set()
    for deck in decks:
        card_ids.update(deck)
    return sorted(card_ids)


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


def weights_to_tensor(records: list[DeckRecord], win_rate_weight: float) -> torch.Tensor:
    weights = torch.ones(len(records), dtype=torch.float32)
    for index, record in enumerate(records):
        weight = 1.0
        if win_rate_weight > 0:
            weight *= 1.0 + win_rate_weight * math.log1p(max(0.0, min(1.0, record.win_rate)))
        weights[index] = weight
    return weights


def dataset_sampling_weights(records: list[DeckRecord], samples_per_deck: int, sampling: str) -> torch.Tensor:
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


def load_word2vec_embedding(path: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state"]
    if "card_embedding" not in state:
        raise ValueError(f"{path} does not contain model_state['card_embedding']")
    return state["card_embedding"].detach().float().cpu(), checkpoint.get("config", {})


def initialize_card_embeddings(
    model: DeckTransformerCBOW,
    pretrained: torch.Tensor | None,
    device: torch.device,
) -> None:
    with torch.no_grad():
        torch.nn.init.normal_(model.token_embedding.weight, mean=0.0, std=0.02)
        model.token_embedding.weight[model.pad_id].zero_()
        if pretrained is None:
            return
        rows = min(pretrained.size(0), model.vocab_size)
        cols = min(pretrained.size(1), model.token_embedding.embedding_dim)
        model.token_embedding.weight[:rows, :cols] = pretrained[:rows, :cols].to(device)


def weighted_cross_entropy(logits: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    loss_by_sample = F.cross_entropy(logits, target, reduction="none")
    return (loss_by_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def evaluate_model(
    model: DeckTransformerCBOW,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    top1 = 0.0
    top5 = 0.0
    total_examples = 0
    with torch.no_grad():
        for token_ids, padding_mask, target, sample_weight in loader:
            token_ids = token_ids.to(device)
            padding_mask = padding_mask.to(device)
            target = target.to(device)
            sample_weight = sample_weight.to(device)
            logits = model(token_ids, padding_mask)
            loss_by_sample = F.cross_entropy(logits, target, reduction="none")
            total_loss += float((loss_by_sample * sample_weight).sum().item())
            total_weight += float(sample_weight.sum().item())
            predictions = torch.topk(logits, k=min(5, logits.size(1)), dim=1).indices
            top1 += float((predictions[:, 0] == target).float().sum().item())
            top5 += float((predictions == target.unsqueeze(1)).any(dim=1).float().sum().item())
            total_examples += int(target.numel())
    return (
        total_loss / max(total_weight, 1e-6),
        top1 / max(total_examples, 1),
        top5 / max(total_examples, 1),
    )


def validate_train_args(args: argparse.Namespace) -> None:
    if args.min_context < 0:
        raise ValueError("--min-context must be non-negative")
    if args.max_context < args.min_context:
        raise ValueError("--max-context must be greater than or equal to --min-context")
    if args.max_context > DECK_SIZE - 1:
        raise ValueError(f"--max-context must be at most {DECK_SIZE - 1}")
    if args.heads <= 0:
        raise ValueError("--heads must be positive")
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.embedding_dim < 0:
        raise ValueError("--embedding-dim must be non-negative")


def train_model(args: argparse.Namespace) -> int:
    validate_train_args(args)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    pretrained: torch.Tensor | None = None
    word2vec_config: dict[str, Any] = {}
    if args.random_init:
        embedding_dim = args.embedding_dim or 128
    else:
        pretrained, word2vec_config = load_word2vec_embedding(args.word2vec_checkpoint, device)
        embedding_dim = args.embedding_dim or int(pretrained.size(1))
        if pretrained.size(1) != embedding_dim:
            raise ValueError(
                "--embedding-dim must match deck_word2vec.pt card_embedding dimension "
                f"({pretrained.size(1)}) unless --random-init is used"
            )

    if embedding_dim % args.heads != 0:
        raise ValueError("--embedding-dim must be divisible by --heads")

    card_meta = load_card_meta(args.card_data)
    all_records = read_deck_candidates(args.index)
    records = sample_records(all_records, args.max_decks, args.seed, args.deck_sampling)
    train_records, valid_records = split_records(records, args.valid_ratio, args.seed)
    decks = [record.deck for record in records]
    known_card_ids = build_card_id_set(decks)
    max_pretrained_id = int(pretrained.size(0) - 1) if pretrained is not None else 0
    vocab_size = max(max(known_card_ids), max(card_meta, default=0), max_pretrained_id) + 1
    pad_id = vocab_size
    bos_id = vocab_size + 1

    train_weights = weights_to_tensor(train_records, args.win_rate_weight)
    valid_weights = weights_to_tensor(valid_records, args.win_rate_weight)
    train_dataset = DeckTransformerDataset(
        train_records,
        max_context=args.max_context,
        pad_id=pad_id,
        bos_id=bos_id,
        samples_per_deck=args.samples_per_deck,
        min_context=args.min_context,
        deck_loss_weights=train_weights,
        seed=args.seed,
    )
    valid_dataset = DeckTransformerDataset(
        valid_records,
        max_context=args.max_context,
        pad_id=pad_id,
        bos_id=bos_id,
        samples_per_deck=max(1, args.samples_per_deck // 4),
        min_context=args.min_context,
        deck_loss_weights=valid_weights,
        seed=args.seed + 1_000_000,
    )
    train_sampler = WeightedRandomSampler(
        weights=dataset_sampling_weights(train_records, args.samples_per_deck, args.deck_sampling),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    model = DeckTransformerCBOW(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pad_id=pad_id,
        bos_id=bos_id,
    ).to(device)
    initialize_card_embeddings(model, pretrained, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_valid = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_top1 = 0.0
    best_top5 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for token_ids, padding_mask, target, sample_weight in train_loader:
            token_ids = token_ids.to(device)
            padding_mask = padding_mask.to(device)
            target = target.to(device)
            sample_weight = sample_weight.to(device)
            logits = model(token_ids, padding_mask)
            loss = weighted_cross_entropy(logits, target, sample_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            train_batches += 1

        valid_loss, valid_top1, valid_top5 = evaluate_model(model, valid_loader, device)
        train_avg = train_loss / max(train_batches, 1)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_avg:.6f} "
            f"valid_loss={valid_loss:.6f} "
            f"valid_top1={valid_top1:.4f} "
            f"valid_top5={valid_top5:.4f}"
        )
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_top1 = valid_top1
            best_top5 = valid_top5
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "pad_id": pad_id,
                "bos_id": bos_id,
                "embedding_dim": embedding_dim,
                "heads": args.heads,
                "layers": args.layers,
                "ff_dim": args.ff_dim,
                "dropout": args.dropout,
                "max_context": args.max_context,
                "known_card_ids": known_card_ids,
                "word2vec_checkpoint": str(args.word2vec_checkpoint),
                "word2vec_embedding_dim": int(pretrained.size(1)) if pretrained is not None else None,
                "word2vec_vocab_size": int(word2vec_config.get("vocab_size", 0)),
                "random_init": bool(args.random_init),
                "win_rate_weight": args.win_rate_weight,
                "deck_sampling": args.deck_sampling,
                "device_option": args.device,
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
                "best_valid_top1": best_top1,
                "best_valid_top5": best_top5,
                "source_decks": len(all_records),
                "used_decks": len(records),
                "train_decks": len(train_records),
                "valid_decks": len(valid_records),
                "mean_win_rate": sum(record.win_rate for record in records) / len(records),
                "mean_cluster_weight": sum(record.cluster_weight for record in records) / len(records),
                "mean_cluster_wins": sum(record.cluster_wins for record in records) / len(records),
                "mean_loss_weight": float(train_weights.mean().item()),
                "mean_deck_sampling_weight": sum(deck_sampling_weight(record, args.deck_sampling) for record in records)
                / len(records),
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


def fallback_card_id(known_card_ids: list[int], card_meta: dict[int, CardMeta], deck: list[int]) -> int:
    for card_id in known_card_ids:
        meta = card_meta.get(card_id)
        if meta and meta.is_basic_energy:
            return card_id
    for card_id in known_card_ids:
        if allowed_to_add(card_id, deck, card_meta):
            return card_id
    return 3


def tokens_from_deck(deck: list[int], pad_id: int, bos_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [bos_id] + [card_id for card_id in deck if 0 <= card_id < pad_id]
    token_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    padding_mask = token_ids == pad_id
    return token_ids, padding_mask


def choose_next_card(
    logits: torch.Tensor,
    known_card_ids: list[int],
    deck: list[int],
    card_meta: dict[int, CardMeta],
    temperature: float,
    top_k: int,
    rng: random.Random,
) -> int | None:
    candidates = [card_id for card_id in known_card_ids if card_id < logits.numel() and allowed_to_add(card_id, deck, card_meta)]
    if not candidates:
        return None

    candidate_logits = torch.tensor([float(logits[card_id].item()) for card_id in candidates], dtype=torch.float32)
    if temperature <= 0:
        return candidates[int(torch.argmax(candidate_logits).item())]

    k = min(max(1, top_k), len(candidates))
    values, indices = torch.topk(candidate_logits, k=k)
    probs = torch.softmax(values / max(temperature, 1e-6), dim=0).tolist()
    choice_index = rng.choices(range(k), weights=probs, k=1)[0]
    return candidates[int(indices[choice_index].item())]


def generate_deck(
    model: DeckTransformerCBOW,
    observed_cards: list[int],
    known_card_ids: list[int],
    card_meta: dict[int, CardMeta],
    device: torch.device,
    temperature: float,
    top_k: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    deck = list(observed_cards[:DECK_SIZE])
    model.eval()
    with torch.no_grad():
        while len(deck) < DECK_SIZE:
            token_ids, padding_mask = tokens_from_deck(deck, model.pad_id, model.bos_id, device)
            logits = model(token_ids, padding_mask)[0].detach().cpu()
            next_card = choose_next_card(logits, known_card_ids, deck, card_meta, temperature, top_k, rng)
            if next_card is None:
                next_card = fallback_card_id(known_card_ids, card_meta, deck)
            deck.append(next_card)
    return deck


def generate(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = DeckTransformerCBOW(
        vocab_size=int(config["vocab_size"]),
        embedding_dim=int(config["embedding_dim"]),
        heads=int(config["heads"]),
        layers=int(config["layers"]),
        ff_dim=int(config["ff_dim"]),
        dropout=float(config.get("dropout", 0.0)),
        pad_id=int(config["pad_id"]),
        bos_id=int(config["bos_id"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    observed_cards = parse_observed(args)
    card_meta = meta_from_checkpoint(config)
    known_card_ids = [int(card_id) for card_id in config["known_card_ids"]]
    deck = generate_deck(
        model=model,
        observed_cards=observed_cards,
        known_card_ids=known_card_ids,
        card_meta=card_meta,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
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
    if args.command in {"generate", "predict"}:
        return generate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
