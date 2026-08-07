"""Submission-safe deck reconstruction based on tools/deck_generator_dbrecon.

The development tool itself is not included in Kaggle submissions, so this module keeps
the same database format and matching semantics next to the agent runtime.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DECK_SIZE = 60
DEFAULT_DATABASE = Path(__file__).with_name("deck_candidates_by_wins.jsonl")


@dataclass(frozen=True)
class DeckMatch:
    deck: list[int]
    matched_cards: int
    observed_cards: int
    missing_cards: int
    candidate_index: int
    record: dict[str, Any]

    @property
    def coverage(self) -> float:
        return self.matched_cards / max(self.observed_cards, 1)


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


def load_candidates(path: Path = DEFAULT_DATABASE) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            counts = _record_counts(record, line_number)
            raw_deck = record.get("deck")
            deck = (
                [int(card_id) for card_id in raw_deck]
                if isinstance(raw_deck, list)
                else list(counts.elements())
            )
            if len(deck) != DECK_SIZE or Counter(deck) != counts:
                raise ValueError(f"candidate line {line_number} is not a consistent 60-card deck")
            record["deck"] = deck
            records.append(record)
    if not records:
        raise ValueError(f"candidate database is empty: {path}")
    return records


def rank_matching_decks(
    observed_cards: Iterable[int],
    records: list[dict[str, Any]],
    limit: int | None = None,
) -> list[DeckMatch]:
    """Rank candidates using dbrecon overlap, then empirical usage as prior.

    A candidate missing an observed physical card is always ranked below a fully
    consistent candidate. Wins are intentionally not used: this is an opponent
    population prior, not a deck-strength selector.
    """

    observed = Counter(int(card_id) for card_id in observed_cards)
    observed_count = sum(observed.values())
    matches: list[DeckMatch] = []
    for index, record in enumerate(records):
        candidate = _record_counts(record, index + 1)
        matched = sum(min(count, candidate.get(card_id, 0)) for card_id, count in observed.items())
        matches.append(
            DeckMatch(
                deck=list(record["deck"]),
                matched_cards=matched,
                observed_cards=observed_count,
                missing_cards=observed_count - matched,
                candidate_index=index,
                record=record,
            )
        )

    matches.sort(
        key=lambda match: (
            -match.missing_cards,
            match.matched_cards,
            int(match.record.get("games", 0)),
            -match.candidate_index,
        ),
        reverse=True,
    )
    return matches if limit is None else matches[:limit]


def select_best_matching_deck(
    observed_cards: Iterable[int], records: list[dict[str, Any]]
) -> DeckMatch:
    """Return the best candidate using the dbrecon-compatible overlap rule."""

    if not records:
        raise ValueError("at least one candidate deck is required")
    return rank_matching_decks(observed_cards, records, limit=1)[0]
