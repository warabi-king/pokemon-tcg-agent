"""既存 tools/train/train_imitation.py をサブプロセスで呼ぶ薄いラッパ。

shards が空（学習データ0）のときは学習をスキップして False を返す。
warm-start する場合は initial_model を --initial-model として渡す。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import config

_TRAIN_IMITATION = config.REPO_ROOT / "tools" / "train" / "train_imitation.py"


def has_shards(shards_dir: Path) -> bool:
    return bool(list(Path(shards_dir).glob("shard_*.pkl")))


def train_model(
    shards_dir: Path,
    output_model: Path,
    epochs: int,
    initial_model: Path | None = None,
    metrics_file: Path | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> bool:
    """train_imitation.py を実行して output_model を書く。成功時 True。"""
    if not has_shards(shards_dir):
        print(f"  [train] シャードが無いためスキップ: {shards_dir}")
        return False

    output_model.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.PYTHON,
        str(_TRAIN_IMITATION),
        "--shards", str(shards_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size if batch_size is not None else config.BATCH_SIZE),
        "--lr", str(lr if lr is not None else config.LR),
        "--output-model", str(output_model),
    ]
    if metrics_file is not None:
        cmd += ["--metrics-file", str(metrics_file)]
    if initial_model is not None and Path(initial_model).exists():
        cmd += ["--initial-model", str(initial_model)]

    print(f"  [train] {output_model.name} <- {shards_dir.name} (epochs={epochs}"
          f"{', warm-start' if initial_model and Path(initial_model).exists() else ''})")
    subprocess.run(cmd, check=True, cwd=str(config.REPO_ROOT))
    return True
