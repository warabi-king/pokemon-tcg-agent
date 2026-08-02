"""並列対戦episodeを1回だけ走査し、全agentのself/相手モデル用シャードへ振り分ける。

``--agent name=deck.csv`` で指定した各デッキについて、

* ``name_own``: そのデッキを使ったプレイヤーの着手
* ``name_opponent``: そのデッキと対戦したプレイヤーの着手

を出力する。同じepisodeをagent数×2回読み直さないためのループ学習専用前処理。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from episode_io import iter_multi_source  # noqa: E402
from imitation_data import extract_player_samples_from_episode  # noqa: E402


PREENCODED_TRAINING_EPISODE_FORMAT = (
    "pokemon-tcg-agent/preencoded-episode-v1"
)


def read_deck(path: Path) -> list[int]:
    deck = [
        int(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck.csvは60枚必要です: {path} ({len(deck)}枚)")
    return deck


def parse_agent(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--agentはname=deck.csv形式で指定してください。")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"不正な--agentです: {raw}")
    return name.strip(), Path(path.strip()).resolve()


def iter_training_episodes(
    sources: list[Path],
):
    """JSON/zipに加えて、対戦時に事前エンコードしたpickleも読む。"""
    for source in sources:
        source = Path(source)
        if source.is_dir():
            episode_paths = sorted(
                (*source.glob("*.json"), *source.glob("*.pkl")),
                key=lambda candidate: candidate.name,
            )
            for episode_path in episode_paths:
                yield source, episode_path.name, episode_path.read_bytes()
        elif source.suffix.lower() == ".pkl":
            yield source, source.name, source.read_bytes()
        else:
            yield from iter_multi_source([source])


def extract_preencoded_player_samples(
    data: bytes,
) -> tuple[list[list[int]], list[list[tuple]]] | None:
    """対戦中に作成済みの特徴量を再エンコードせず取り出す。"""
    episode = pickle.loads(data)
    if not isinstance(episode, dict):
        return None
    if episode.get("format") != PREENCODED_TRAINING_EPISODE_FORMAT:
        return None
    decks = episode.get("decks")
    if (
        not isinstance(decks, list)
        or len(decks) != 2
        or any(not isinstance(deck, list) or len(deck) != 60 for deck in decks)
    ):
        return None

    packed_players = episode.get("packedPlayerSamples")
    if isinstance(packed_players, list) and len(packed_players) == 2:
        field_names = (
            "encoderIndex",
            "encoderValue",
            "encoderOffset",
            "decoderIndex",
            "decoderValue",
            "decoderOffset",
        )
        player_samples: list[list[tuple]] = [[], []]
        for player, packed in enumerate(packed_players):
            if not isinstance(packed, dict):
                return None
            count = packed.get("count")
            chosen_indices = packed.get("chosenIndex")
            values = packed.get("value")
            if (
                not isinstance(count, int)
                or count < 0
                or not isinstance(chosen_indices, np.ndarray)
                or not isinstance(values, np.ndarray)
                or len(chosen_indices) != count
                or len(values) != count
            ):
                return None

            ragged_fields: list[tuple[np.ndarray, np.ndarray]] = []
            for field_name in field_names:
                ragged = packed.get(field_name)
                if not isinstance(ragged, dict):
                    return None
                flat_values = ragged.get("values")
                boundaries = ragged.get("boundaries")
                if (
                    not isinstance(flat_values, np.ndarray)
                    or not isinstance(boundaries, np.ndarray)
                    or len(boundaries) != count + 1
                    or int(boundaries[0]) != 0
                    or int(boundaries[-1]) != len(flat_values)
                ):
                    return None
                ragged_fields.append((flat_values, boundaries))

            for sample_index in range(count):
                fields = []
                for flat_values, boundaries in ragged_fields:
                    start = int(boundaries[sample_index])
                    end = int(boundaries[sample_index + 1])
                    if start < 0 or end < start or end > len(flat_values):
                        return None
                    fields.append(flat_values[start:end].tolist())
                player_samples[player].append(
                    (*fields, int(chosen_indices[sample_index]), float(values[sample_index]))
                )
        return decks, player_samples

    # 旧preencoded形式も既存データの読み込み用に残す。
    player_samples = episode.get("playerSamples")
    if (
        not isinstance(player_samples, list)
        or len(player_samples) != 2
        or any(not isinstance(samples, list) for samples in player_samples)
    ):
        return None
    if any(
        not isinstance(sample, tuple) or len(sample) != 8
        for samples in player_samples
        for sample in samples
    ):
        return None
    return decks, player_samples


def preprocess_match_agents(
    episodes: list[Path],
    agents: list[tuple[str, Path]],
    output_dir: Path,
    shard_size: int = 20_000,
    max_episodes: int | None = None,
) -> dict[str, dict[str, int]]:
    if shard_size < 1:
        raise ValueError("shard_sizeは1以上で指定してください。")
    if len({name for name, _ in agents}) != len(agents):
        raise ValueError("agent名が重複しています。")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deck_to_agents: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for name, deck_path in agents:
        deck_to_agents[tuple(sorted(read_deck(deck_path)))].append(name)

    keys = [
        (name, role)
        for name, _ in agents
        for role in ("own", "opponent")
    ]
    buffers: dict[tuple[str, str], list] = {key: [] for key in keys}
    shard_indices = {key: 0 for key in keys}
    sample_counts = {key: 0 for key in keys}
    for name, role in keys:
        target = output_dir / f"{name}_{role}"
        target.mkdir(parents=True, exist_ok=True)
        existing = next(target.glob("shard_*.pkl"), None)
        if existing is not None:
            raise FileExistsError(f"既存シャードがあります: {existing}")

    def flush(key: tuple[str, str], force: bool = False) -> None:
        buffer = buffers[key]
        while len(buffer) >= shard_size or (force and buffer):
            count = min(len(buffer), shard_size)
            chunk = buffer[:count]
            del buffer[:count]
            name, role = key
            path = output_dir / f"{name}_{role}" / f"shard_{shard_indices[key]:05d}.pkl"
            with path.open("wb") as file:
                pickle.dump(chunk, file, protocol=pickle.HIGHEST_PROTOCOL)
            shard_indices[key] += 1

    episode_count = 0
    skipped = 0
    started = time.time()
    for _source, filename, data in iter_training_episodes(episodes):
        if Path(filename).name in {
            "manifest.csv",
            "manifest.json",
            "minimal_manifest.json",
            "training_manifest.json",
        }:
            continue
        if max_episodes is not None and episode_count >= max_episodes:
            break
        episode_count += 1
        try:
            if Path(filename).suffix.lower() == ".pkl":
                extracted = extract_preencoded_player_samples(data)
            else:
                extracted = extract_player_samples_from_episode(data)
        except Exception:
            extracted = None
        if extracted is None:
            skipped += 1
            continue

        decks, player_samples = extracted
        for player in range(2):
            own_agents = deck_to_agents.get(tuple(sorted(decks[player])), ())
            opponent_agents = deck_to_agents.get(
                tuple(sorted(decks[1 - player])),
                (),
            )
            samples = player_samples[player]
            for name in own_agents:
                key = (name, "own")
                buffers[key].extend(samples)
                sample_counts[key] += len(samples)
                flush(key)
            for name in opponent_agents:
                key = (name, "opponent")
                buffers[key].extend(samples)
                sample_counts[key] += len(samples)
                flush(key)

        if episode_count % 500 == 0:
            print(
                f"episodes={episode_count} skipped={skipped} "
                f"samples={sum(sample_counts.values())} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    for key in keys:
        flush(key, force=True)

    summary: dict[str, dict[str, int]] = {}
    for key in keys:
        name, role = key
        target = output_dir / f"{name}_{role}"
        manifest = {
            "agent": name,
            "role": role,
            "episodes": episode_count,
            "episodesSkipped": skipped,
            "samples": sample_counts[key],
            "shards": shard_indices[key],
            "shardSize": shard_size,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[f"{name}_{role}"] = manifest

    print(
        f"done: episodes={episode_count} skipped={skipped} "
        f"elapsed={time.time() - started:.1f}s"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--agent", type=parse_agent, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_match_agents(
        args.episodes,
        args.agent,
        args.output_dir,
        shard_size=args.shard_size,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
