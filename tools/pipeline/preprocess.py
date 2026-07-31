"""エピソード履歴 → 模倣学習シャード（自前フィルタ対応版）。

既存 tools/train/preprocess_episodes.py と同じライブラリ関数
（extract_samples_from_episode / iter_multi_source）を再利用しつつ、
deck_filter を Python の Callable として差し込めるようにしたもの。

- Phase0: cluster_filter（公式リプレイをクラスタ最近傍で絞る）
- 世代ループ: exact_deck_filter（リーグ内の固定デッキで厳密に絞る）

戻り値は生成シャード数。0 のときは学習をスキップする判断に使う。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Callable

import config  # noqa: F401  (sys.path 設定の副作用)
from episode_io import iter_multi_source
from imitation_data import extract_samples_from_episode

DeckFilter = Callable[[list[int], list[int]], bool]


def preprocess(
    episodes: list[Path],
    output_dir: Path,
    deck_filter: DeckFilter | None,
    shard_size: int,
    role: str = "own",
    label: str = "",
    max_episodes: int | None = None,
    verbose: bool = True,
) -> int:
    """episodes（ディレクトリ or .zip の列）を前処理して output_dir にシャードを書く。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list = []
    shard_index = 0
    total_samples = 0
    episode_count = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        with open(output_dir / f"shard_{shard_index:05d}.pkl", "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_index += 1
        buffer = []

    for _source, name, data in iter_multi_source(episodes):
        if name == "manifest.csv":
            continue
        if max_episodes is not None and episode_count >= max_episodes:
            break
        try:
            samples = extract_samples_from_episode(data, deck_filter=deck_filter)
        except Exception:
            samples = []
        buffer.extend(samples)
        total_samples += len(samples)
        episode_count += 1
        if len(buffer) >= shard_size:
            flush()
        if verbose and episode_count % 500 == 0:
            print(
                f"  [{label}] episodes={episode_count} samples={total_samples} "
                f"shards={shard_index} elapsed={time.time()-t0:.1f}s",
                flush=True,
            )
    flush()

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "label": label,
                "role": role,
                "episodes": episode_count,
                "samples": total_samples,
                "shards": shard_index,
                "shard_size": shard_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        print(f"  [{label}] done: episodes={episode_count} samples={total_samples} shards={shard_index}")
    return shard_index


def episode_sources(episodes_dir: Path) -> list[Path]:
    """episodes ディレクトリ配下の .zip、または *.json を含むディレクトリ群を列挙する。

    iter_episode_files はディレクトリ直下の *.json しか読まないため、zip が無い場合は
    *.json を含む全ディレクトリ（サブディレクトリ含む）を返す。
    """
    episodes_dir = Path(episodes_dir)
    if not episodes_dir.exists():
        return []
    zips = sorted(episodes_dir.rglob("*.zip"))
    if zips:
        return zips
    dirs = sorted({p.parent for p in episodes_dir.rglob("*.json")})
    return dirs
