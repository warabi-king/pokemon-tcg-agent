"""公式日次ZIPから固定seedでepisodeを抽出し、模倣学習shardを作る。

実行例:
    python agents/belief_puct/train/prepare_imitation.py \
        --archives ../develop_nomura/episodes/official/2026-07-{01..17}/*.zip \
        --episodes-per-archive 20 --seed 20260803 \
        --output-dir agents/belief_puct/train/shards/train

ZIP全体は展開せず、各日から選んだJSONだけをstreamingで解凍する。
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from imitation_data import extract_samples_from_episode


def selected_members(archive: ZipFile, count: int, rng: random.Random) -> list[str]:
    """manifestを除くepisode JSONから固定seedで最大count件を選ぶ。"""

    members = sorted(
        member.filename
        for member in archive.infolist()
        if member.filename.endswith(".json")
    )
    if count <= 0 or count >= len(members):
        return members
    return sorted(rng.sample(members, count))


def flush_shard(buffer: list[tuple], output_dir: Path, shard_index: int) -> None:
    """学習sample bufferをpickle shardとして保存する。"""

    shard_path = output_dir / f"shard_{shard_index:05d}.pkl"
    with shard_path.open("wb") as destination:
        pickle.dump(buffer, destination, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_shards(
    archive_paths: list[Path],
    output_dir: Path,
    episodes_per_archive: int,
    shard_size: int,
    seed: int,
) -> dict[str, object]:
    """選択episodeを特徴量へ変換し、manifestつきshard群を返す。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    buffer: list[tuple] = []
    shard_index = 0
    episodes_read = 0
    episodes_failed = 0
    samples_total = 0
    archive_summary: list[dict[str, object]] = []
    started = time.perf_counter()

    for archive_path in archive_paths:
        archive_episodes = 0
        archive_samples = 0
        try:
            with ZipFile(archive_path) as archive:
                members = selected_members(archive, episodes_per_archive, rng)
                for member in members:
                    try:
                        samples = extract_samples_from_episode(archive.read(member))
                    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                        episodes_failed += 1
                        continue
                    episodes_read += 1
                    archive_episodes += 1
                    archive_samples += len(samples)
                    samples_total += len(samples)
                    buffer.extend(samples)
                    while len(buffer) >= shard_size:
                        flush_shard(buffer[:shard_size], output_dir, shard_index)
                        buffer = buffer[shard_size:]
                        shard_index += 1
        except (BadZipFile, OSError) as error:
            archive_summary.append(
                {"path": str(archive_path), "episodes": 0, "samples": 0, "error": str(error)}
            )
            continue
        archive_summary.append(
            {
                "path": str(archive_path.resolve()),
                "episodes": archive_episodes,
                "samples": archive_samples,
                "error": None,
            }
        )
        print(
            f"archive={archive_path.name} episodes={archive_episodes} "
            f"samples={archive_samples} total_samples={samples_total}",
            flush=True,
        )

    if buffer:
        flush_shard(buffer, output_dir, shard_index)
        shard_index += 1
    manifest = {
        "format": "pokemon-tcg-agent/imitation-shards-v1",
        "seed": seed,
        "episodes_per_archive": episodes_per_archive,
        "shard_size": shard_size,
        "episodes_read": episodes_read,
        "episodes_failed": episodes_failed,
        "samples": samples_total,
        "shards": shard_index,
        "elapsed_seconds": time.perf_counter() - started,
        "archives": archive_summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    """入力ZIP列、抽出数、seed、shard設定を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-archive", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def main() -> None:
    """shard作成を実行し、manifestを標準出力へ表示する。"""

    args = parse_args()
    manifest = prepare_shards(
        sorted(path.resolve() for path in args.archives),
        args.output_dir,
        args.episodes_per_archive,
        args.shard_size,
        args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
