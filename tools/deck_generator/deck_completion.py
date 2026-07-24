from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_DIR = SCRIPT_DIR / "daily_dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_INDEX = DEFAULT_OUTPUT_DIR / "deck_candidates.jsonl"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "deck_candidates_summary.json"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DECK_SIZE = 60


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
        text = f"{self.name} {self.rule}".upper()
        return "ACE SPEC" in text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query a simple kNN deck completion model from daily_dataset zip files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index", help="Scan daily_dataset zips and build deck candidate DB.")
    build.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    build.add_argument("--output", type=Path, default=DEFAULT_INDEX)
    build.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    build.add_argument("--max-episodes", type=int, help="Debug limit per whole run.")
    build.add_argument("--progress-interval", type=int, default=100)

    complete = subparsers.add_parser("complete", help="Complete a 60-card deck from observed card IDs.")
    complete.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    complete.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    complete.add_argument(
        "--observed",
        action="append",
        default=[],
        help="Observed card IDs. Accepts comma-separated IDs and can be repeated.",
    )
    complete.add_argument("--observed-file", type=Path, help="File containing one card ID per line.")
    complete.add_argument("--neighbors", type=int, default=32)
    complete.add_argument("--samples", type=int, default=1)
    complete.add_argument("--seed", type=int, default=0)
    complete.add_argument("--json", action="store_true", help="Print JSON instead of card IDs.")

    return parser.parse_args()


def load_card_meta(path: Path) -> dict[int, CardMeta]:
    if not path.exists():
        return {}
    result: dict[int, CardMeta] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            card_id = int(row["Card ID"])
            result[card_id] = CardMeta(
                card_id=card_id,
                name=row["Card Name"],
                kind=row["Stage (Pokemon)/Type (Energy and Trainer)"]
                if "Stage (Pokemon)/Type (Energy and Trainer)" in row
                else row["Stage (Pokémon)/Type (Energy and Trainer)"],
                rule=row.get("Rule", ""),
            )
    return result


def find_zip_files(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        return []
    return sorted(path for path in dataset_dir.glob("*/*.zip") if path.is_file())


def collect_episode_decks(episode: dict[str, Any]) -> list[list[int] | None]:
    decks: list[list[int] | None] = [None, None]
    for step in episode.get("steps", []):
        for player_index, agent_state in enumerate(step[:2]):
            action = agent_state.get("action")
            if (
                decks[player_index] is None
                and isinstance(action, list)
                and len(action) == DECK_SIZE
                and all(isinstance(card_id, int) and card_id > 0 for card_id in action)
            ):
                decks[player_index] = list(action)
        if all(deck is not None for deck in decks):
            break
    return decks


def fill_to_deck_size(
    known_cards: list[int],
    fill_sequence: list[int],
    card_meta: dict[int, CardMeta],
    rng: random.Random,
) -> list[int]:
    deck = list(known_cards[:DECK_SIZE])
    if len(deck) >= DECK_SIZE:
        return deck

    count_by_id = Counter(deck)
    count_by_name: Counter[str] = Counter()
    ace_spec_count = 0
    for card_id in deck:
        meta = card_meta.get(card_id)
        name = meta.name if meta else str(card_id)
        count_by_name[name] += 1
        if meta and meta.is_ace_spec:
            ace_spec_count += 1

    fallback_energy = [
        card_id for card_id, meta in card_meta.items() if meta.is_basic_energy
    ] or [3]

    for card_id in fill_sequence:
        if len(deck) >= DECK_SIZE:
            break
        meta = card_meta.get(card_id)
        name = meta.name if meta else str(card_id)
        if meta and meta.is_ace_spec and ace_spec_count >= 1:
            continue
        if not (meta and meta.is_basic_energy) and count_by_name[name] >= 4:
            continue

        deck.append(card_id)
        count_by_id[card_id] += 1
        count_by_name[name] += 1
        if meta and meta.is_ace_spec:
            ace_spec_count += 1

    while len(deck) < DECK_SIZE:
        deck.append(fallback_energy[0])
    return deck


def counter_to_fill_sequence(counts: Counter[int], limit_per_card: int = DECK_SIZE) -> list[int]:
    result: list[int] = []
    items = [(card_id, min(count, limit_per_card)) for card_id, count in counts.most_common()]
    for copy_index in range(limit_per_card):
        for card_id, count in items:
            if count > copy_index:
                result.append(card_id)
    return result


def deck_counts(cards: list[int]) -> dict[str, int]:
    return {str(card_id): count for card_id, count in sorted(Counter(cards).items())}


def build_index(args: argparse.Namespace) -> int:
    zip_files = find_zip_files(args.dataset_dir)
    if not zip_files:
        print(f"No zip files found under {args.dataset_dir}", file=sys.stderr)
        return 1

    raw_records: list[dict[str, Any]] = []
    global_counts: Counter[int] = Counter()
    episodes_seen = 0
    episodes_failed = 0

    for zip_path in zip_files:
        date = zip_path.parent.name
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(name for name in zf.namelist() if name.endswith(".json"))
            for name in names:
                if args.max_episodes is not None and episodes_seen >= args.max_episodes:
                    break
                episodes_seen += 1
                try:
                    with zf.open(name) as f:
                        episode = json.load(f)
                    decks = collect_episode_decks(episode)
                except Exception as exc:  # noqa: BLE001
                    episodes_failed += 1
                    print(f"warning: failed to parse {zip_path.name}:{name}: {exc}", file=sys.stderr)
                    continue

                teams = episode.get("info", {}).get("TeamNames", ["", ""])
                episode_id = episode.get("id") or Path(name).stem
                for player_index, deck in enumerate(decks):
                    if deck is None:
                        continue
                    global_counts.update(deck)
                    raw_records.append(
                        {
                            "date": date,
                            "zip": str(zip_path.relative_to(SCRIPT_DIR)),
                            "episode_file": name,
                            "episode_id": episode_id,
                            "player_index": player_index,
                            "team": teams[player_index] if player_index < len(teams) else "",
                            "known_count": DECK_SIZE,
                            "known_cards": deck,
                            "source": "action",
                        }
                    )

                if args.progress_interval and episodes_seen % args.progress_interval == 0:
                    print(
                        f"processed {episodes_seen} episodes, candidates {len(raw_records)}",
                        flush=True,
                    )
            if args.max_episodes is not None and episodes_seen >= args.max_episodes:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for record in raw_records:
            out = {
                **record,
                "deck": record["known_cards"],
                "deck_counts": deck_counts(record["known_cards"]),
            }
            f.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "zip_files": [str(path) for path in zip_files],
        "episodes_seen": episodes_seen,
        "episodes_failed": episodes_failed,
        "candidate_count": len(raw_records),
        "deck_source": "first 60-card action per player",
        "global_counts": {str(card_id): count for card_id, count in sorted(global_counts.items())},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {len(raw_records)} deck candidates to {args.output}")
    print(f"wrote summary to {args.summary}")
    return 0


def parse_observed(args: argparse.Namespace) -> list[int]:
    values: list[int] = []
    for chunk in args.observed:
        for item in chunk.replace(",", " ").split():
            values.append(int(item))
    if args.observed_file:
        for line in args.observed_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                values.append(int(text))
    return values


def load_index(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def score_candidate(observed: Counter[int], candidate: Counter[int]) -> float:
    if not observed:
        return 0.0
    overlap = sum(min(count, candidate.get(card_id, 0)) for card_id, count in observed.items())
    extra = sum(candidate.get(card_id, 0) for card_id in observed)
    return overlap / sum(observed.values()) + 0.05 * extra


def complete_deck(
    observed_cards: list[int],
    records: list[dict[str, Any]],
    card_meta: dict[int, CardMeta],
    neighbors: int,
    rng: random.Random,
) -> tuple[list[int], list[dict[str, Any]]]:
    observed = Counter(observed_cards)
    scored: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        candidate = Counter({int(card_id): count for card_id, count in record["deck_counts"].items()})
        scored.append((score_candidate(observed, candidate), record))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[: max(1, neighbors)]

    weights = [max(score, 0.01) for score, _ in top]
    primary = rng.choices([record for _, record in top], weights=weights, k=1)[0]
    fill_sequence = list(primary["deck"])
    for _, record in top:
        if record is not primary:
            fill_sequence.extend(record["deck"])
    deck = fill_to_deck_size(list(observed.elements()), fill_sequence, card_meta, rng)
    neighbor_info = [
        {
            "score": score,
            "date": record["date"],
            "episode_file": record["episode_file"],
            "player_index": record["player_index"],
            "known_count": record["known_count"],
            "team": record["team"],
        }
        for score, record in top[:5]
    ]
    return deck, neighbor_info


def complete(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    records = load_index(args.index)
    if not records:
        print(f"No records in {args.index}", file=sys.stderr)
        return 1
    card_meta = load_card_meta(args.card_data)
    observed_cards = parse_observed(args)
    outputs = []
    for _ in range(args.samples):
        deck, neighbors = complete_deck(observed_cards, records, card_meta, args.neighbors, rng)
        outputs.append(
            {
                "observed": observed_cards,
                "deck": deck,
                "deck_counts": deck_counts(deck),
                "neighbors": neighbors,
            }
        )

    if args.json:
        print(json.dumps(outputs if args.samples != 1 else outputs[0], ensure_ascii=False, indent=2))
    else:
        for i, output in enumerate(outputs, start=1):
            if args.samples > 1:
                print(f"# sample {i}")
            print("\n".join(str(card_id) for card_id in output["deck"]))
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build-index":
        return build_index(args)
    if args.command == "complete":
        return complete(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
