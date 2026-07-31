from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATABASE = SCRIPT_DIR / "deck_candidates_by_wins.jsonl"
DECK_SIZE = 60


@dataclass(frozen=True)
class DeckMatch:
    """The candidate selected for a partial deck observation."""

    deck: list[int]
    matched_cards: int
    observed_cards: int
    candidate_index: int
    record: dict[str, Any]

    @property
    def coverage(self) -> float:
        return self.matched_cards / self.observed_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the deck with the most observed-card matches from the bundled database."
    )
    parser.add_argument(
        "--observed",
        action="append",
        default=[],
        help="Observed card IDs separated by commas or spaces. May be specified more than once.",
    )
    parser.add_argument(
        "--observed-file",
        type=Path,
        help="Text file containing observed card IDs separated by commas, spaces, or newlines.",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--json", action="store_true", help="Include match metadata in JSON output.")
    return parser.parse_args()


def parse_card_ids(chunks: Iterable[str]) -> list[int]:
    card_ids: list[int] = []
    for chunk in chunks:
        for value in chunk.replace(",", " ").split():
            card_id = int(value)
            if card_id < 0:
                raise ValueError(f"card IDs must be non-negative: {card_id}")
            card_ids.append(card_id)
    return card_ids


def observed_from_args(args: argparse.Namespace) -> list[int]:
    chunks = list(args.observed)
    if args.observed_file:
        chunks.append(args.observed_file.read_text(encoding="utf-8"))
    return parse_card_ids(chunks)


def _record_counts(record: dict[str, Any], line_number: int) -> Counter[int]:
    raw_counts = record.get("deck_counts")
    if isinstance(raw_counts, dict):
        counts = Counter({int(card_id): int(count) for card_id, count in raw_counts.items()})
    elif isinstance(record.get("deck"), list):
        counts = Counter(int(card_id) for card_id in record["deck"])
    else:
        raise ValueError(f"candidate line {line_number} has neither deck_counts nor deck")
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"candidate line {line_number} contains a non-positive card count")
    return counts


def load_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            counts = _record_counts(record, line_number)
            deck = record.get("deck")
            if not isinstance(deck, list):
                deck = list(counts.elements())
                record["deck"] = deck
            if len(deck) != DECK_SIZE or sum(counts.values()) != DECK_SIZE:
                raise ValueError(f"candidate line {line_number} is not a {DECK_SIZE}-card deck")
            if Counter(int(card_id) for card_id in deck) != counts:
                raise ValueError(f"candidate line {line_number} has inconsistent deck and deck_counts")
            records.append(record)
    if not records:
        raise ValueError(f"candidate database is empty: {path}")
    return records


def select_best_matching_deck(observed_cards: Iterable[int], records: list[dict[str, Any]]) -> DeckMatch:
    """Select one candidate by matched card count, using performance as a stable tie-breaker."""

    observed = Counter(int(card_id) for card_id in observed_cards)
    if not observed:
        raise ValueError("at least one observed card is required")
    if any(card_id < 0 for card_id in observed):
        raise ValueError("card IDs must be non-negative")
    if not records:
        raise ValueError("at least one candidate deck is required")

    best_match: DeckMatch | None = None
    best_rank: tuple[int, int, int] | None = None
    for index, record in enumerate(records):
        candidate = _record_counts(record, index + 1)
        raw_deck = record.get("deck")
        deck = (
            [int(card_id) for card_id in raw_deck]
            if isinstance(raw_deck, list)
            else list(candidate.elements())
        )
        matched_cards = sum(min(count, candidate.get(card_id, 0)) for card_id, count in observed.items())
        # Match数が同じ候補は勝数だけで比較する。同勝数ならDB順で固定する。
        rank = (
            matched_cards,
            int(record.get("wins", 0)),
            -index,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_match = DeckMatch(
                deck=deck,
                matched_cards=matched_cards,
                observed_cards=sum(observed.values()),
                candidate_index=index,
                record=record,
            )
    assert best_match is not None
    return best_match


def output_payload(observed_cards: list[int], match: DeckMatch) -> dict[str, Any]:
    return {
        "observed": observed_cards,
        "deck": match.deck,
        "deck_counts": {str(card_id): count for card_id, count in sorted(Counter(match.deck).items())},
        "matched_cards": match.matched_cards,
        "coverage": match.coverage,
        "candidate_index": match.candidate_index,
        "candidate_stats": {
            key: match.record[key]
            for key in ("games", "wins", "losses", "draws", "win_rate", "cluster_id")
            if key in match.record
        },
    }


def main() -> int:
    args = parse_args()
    try:
        observed_cards = observed_from_args(args)
        records = load_candidates(args.database)
        match = select_best_matching_deck(observed_cards, records)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(output_payload(observed_cards, match), ensure_ascii=False, indent=2))
    else:
        print("\n".join(str(card_id) for card_id in match.deck))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
