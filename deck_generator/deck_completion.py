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
DEFAULT_WIN_AGGREGATE = DEFAULT_OUTPUT_DIR / "deck_candidates_by_wins.jsonl"
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

    aggregate = subparsers.add_parser(
        "aggregate-wins",
        help="Aggregate same decks by win count using episode rewards.",
    )
    aggregate.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    aggregate.add_argument("--output", type=Path, default=DEFAULT_WIN_AGGREGATE)
    aggregate.add_argument("--limit", type=int, help="Write only the top N aggregated decks.")
    aggregate.add_argument("--progress-interval", type=int, default=10000)

    cluster = subparsers.add_parser(
        "cluster-weights",
        help="Cluster similar decks and add inverse-frequency training weights.",
    )
    cluster.add_argument("--input", type=Path, default=DEFAULT_WIN_AGGREGATE)
    cluster.add_argument("--output", type=Path, default=DEFAULT_WIN_AGGREGATE)
    cluster.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Histogram intersection similarity threshold. 0.75 means 45/60 cards overlap.",
    )
    cluster.add_argument("--progress-interval", type=int, default=1000)

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


def deck_key(counts: dict[str, int]) -> str:
    return json.dumps(
        {str(card_id): counts[str(card_id)] for card_id in sorted(map(int, counts))},
        sort_keys=True,
        separators=(",", ":"),
    )


def count_histogram(record: dict[str, Any]) -> dict[int, int]:
    return {int(card_id): int(count) for card_id, count in record["deck_counts"].items()}


def histogram_intersection_similarity(a: dict[int, int], b: dict[int, int]) -> float:
    if len(a) > len(b):
        a, b = b, a
    overlap = sum(min(count, b.get(card_id, 0)) for card_id, count in a.items())
    return overlap / DECK_SIZE


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


def resolve_record_zip(index_path: Path, record: dict[str, Any]) -> Path:
    zip_value = Path(record["zip"])
    if zip_value.is_absolute():
        return zip_value
    candidates = [
        SCRIPT_DIR / zip_value,
        index_path.parent / zip_value,
        ROOT / zip_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def reward_to_result(reward: int | float | None) -> str:
    if reward is None:
        return "draw"
    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"


def aggregate_wins(args: argparse.Namespace) -> int:
    aggregates: dict[str, dict[str, Any]] = {}
    reward_cache: dict[tuple[str, str], list[int | float | None]] = {}
    zip_cache: dict[str, zipfile.ZipFile] = {}
    processed = 0
    missing_rewards = 0

    try:
        with args.index.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                processed += 1
                key = deck_key(record["deck_counts"])
                item = aggregates.get(key)
                if item is None:
                    item = {
                        "deck": record["deck"],
                        "deck_counts": record["deck_counts"],
                        "games": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "teams": {},
                        "first_seen": {
                            "date": record["date"],
                            "episode_file": record["episode_file"],
                            "player_index": record["player_index"],
                        },
                    }
                    aggregates[key] = item

                zip_path = resolve_record_zip(args.index, record)
                zip_key = str(zip_path)
                cache_key = (zip_key, record["episode_file"])
                rewards = reward_cache.get(cache_key)
                if rewards is None:
                    try:
                        zf = zip_cache.get(zip_key)
                        if zf is None:
                            zf = zipfile.ZipFile(zip_path)
                            zip_cache[zip_key] = zf
                        with zf.open(record["episode_file"]) as episode_file:
                            episode = json.load(episode_file)
                        rewards = episode.get("rewards", [])
                    except Exception as exc:  # noqa: BLE001
                        missing_rewards += 1
                        rewards = []
                        print(
                            f"warning: failed to read reward for {zip_path}:{record['episode_file']}: {exc}",
                            file=sys.stderr,
                        )
                    reward_cache[cache_key] = rewards

                player_index = int(record["player_index"])
                reward = rewards[player_index] if player_index < len(rewards) else None
                result = reward_to_result(reward)
                item["games"] += 1
                if result == "win":
                    item["wins"] += 1
                elif result == "loss":
                    item["losses"] += 1
                else:
                    item["draws"] += 1
                if record.get("team"):
                    team_counts = item["teams"]
                    team_counts[record["team"]] = team_counts.get(record["team"], 0) + 1

                if args.progress_interval and processed % args.progress_interval == 0:
                    print(
                        f"processed {processed} records, unique decks {len(aggregates)}",
                        flush=True,
                    )
    finally:
        for zf in zip_cache.values():
            zf.close()

    rows = list(aggregates.values())
    for row in rows:
        row["win_rate"] = row["wins"] / row["games"] if row["games"] else 0.0
        row["teams"] = dict(sorted(row["teams"].items(), key=lambda pair: (-pair[1], pair[0])))
    rows.sort(key=lambda row: (-row["wins"], -row["win_rate"], -row["games"], row["deck"]))
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"processed {processed} candidate records")
    print(f"aggregated {len(aggregates)} unique decks")
    print(f"wrote {len(rows)} rows to {args.output}")
    if missing_rewards:
        print(f"missing reward episodes: {missing_rewards}", file=sys.stderr)
    return 0


def cluster_weight(total_decks: int, cluster_members: int, cluster_count: int) -> float:
    if cluster_members <= 0 or cluster_count <= 0:
        return 1.0
    return total_decks / (cluster_members * cluster_count)


def cluster_weights(args: argparse.Namespace) -> int:
    records = load_index(args.input)
    if not records:
        print(f"No records in {args.input}", file=sys.stderr)
        return 1

    records.sort(key=lambda row: (-int(row.get("wins", 0)), -float(row.get("win_rate", 0.0)), -int(row.get("games", 0))))
    histograms = [count_histogram(record) for record in records]
    assignments: list[int | None] = [None] * len(records)
    clusters: list[dict[str, Any]] = []

    for i, record in enumerate(records):
        if assignments[i] is not None:
            continue
        cluster_id = len(clusters)
        representative = histograms[i]
        member_indices = [i]
        assignments[i] = cluster_id
        for j in range(i + 1, len(records)):
            if assignments[j] is not None:
                continue
            if histogram_intersection_similarity(representative, histograms[j]) >= args.threshold:
                assignments[j] = cluster_id
                member_indices.append(j)

        games = sum(int(records[index].get("games", 0)) for index in member_indices)
        wins = sum(int(records[index].get("wins", 0)) for index in member_indices)
        losses = sum(int(records[index].get("losses", 0)) for index in member_indices)
        draws = sum(int(records[index].get("draws", 0)) for index in member_indices)
        clusters.append(
            {
                "cluster_id": cluster_id,
                "representative_index": i,
                "member_indices": member_indices,
                "members": len(member_indices),
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / games if games else 0.0,
            }
        )

        if args.progress_interval and len(clusters) % args.progress_interval == 0:
            print(f"clusters {len(clusters)}, assigned {sum(value is not None for value in assignments)}", flush=True)

    total_decks = len(records)
    cluster_count = len(clusters)
    for cluster in clusters:
        cluster["weight"] = cluster_weight(total_decks, cluster["members"], cluster_count)
        for index in cluster["member_indices"]:
            records[index]["cluster_id"] = cluster["cluster_id"]
            records[index]["cluster_members"] = cluster["members"]
            records[index]["cluster_games"] = cluster["games"]
            records[index]["cluster_wins"] = cluster["wins"]
            records[index]["cluster_losses"] = cluster["losses"]
            records[index]["cluster_draws"] = cluster["draws"]
            records[index]["cluster_win_rate"] = cluster["win_rate"]
            records[index]["cluster_weight"] = cluster["weight"]
            records[index]["cluster_similarity_threshold"] = args.threshold
            records[index]["cluster_weight_formula"] = "total_decks/(cluster_members*cluster_count)"
            records[index].pop("cluster_weight_mode", None)
            records[index].pop("cluster_min_weight", None)
            records[index].pop("cluster_max_weight", None)

    records.sort(
        key=lambda row: (
            int(row["cluster_id"]),
            -int(row.get("wins", 0)),
            -float(row.get("win_rate", 0.0)),
            -int(row.get("games", 0)),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"read {len(records)} decks from {args.input}")
    print(f"created {len(clusters)} clusters with threshold {args.threshold}")
    print(f"wrote clustered decks to {args.output}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "build-index":
        return build_index(args)
    if args.command == "complete":
        return complete(args)
    if args.command == "aggregate-wins":
        return aggregate_wins(args)
    if args.command == "cluster-weights":
        return cluster_weights(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
