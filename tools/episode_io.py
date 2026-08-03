"""複数の日次エピソードデータセットを横断して読む共通ヘルパー。"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator


def iter_episode_files(source: Path) -> Iterator[tuple[str, bytes]]:
    """ディレクトリまたはZIPから、ファイル名とJSONの生データを返す。"""
    if source.is_dir():
        for path in sorted(source.glob("*.json")):
            yield path.name, path.read_bytes()
        return

    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                yield name, archive.read(name)


def iter_multi_source(
    sources: list[Path],
) -> Iterator[tuple[Path, str, bytes]]:
    """複数ソースから、元ソース・ファイル名・生データを順番に返す。"""
    for source in sources:
        for name, data in iter_episode_files(source):
            yield source, name, data
