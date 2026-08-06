"""複数の日次エピソードデータセット(展開済みディレクトリ or Kaggleの.zip)を
横断して読むための共通ヘルパー。
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterator


def iter_episode_files(source: Path) -> Iterator[tuple[str, bytes]]:
    """1つのソース(展開済みディレクトリ or .zip)から (ファイル名, 生データ) を返す。"""
    if source.is_dir():
        for p in sorted(source.glob("*.json")):
            yield p.name, p.read_bytes()
        return

    with zipfile.ZipFile(source) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                yield name, zf.read(name)


def iter_multi_source(sources: list[Path]) -> Iterator[tuple[Path, str, bytes]]:
    """複数日分のソースを順番に読む。(元ソース, ファイル名, 生データ) を返す。"""
    for source in sources:
        for name, data in iter_episode_files(source):
            yield source, name, data
