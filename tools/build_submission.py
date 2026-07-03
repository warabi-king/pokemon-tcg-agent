"""Kaggle提出用のsubmission tar.gzを作成する。"""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = ROOT / "agents"
DIST_ROOT = ROOT / "dist"
REQUIRED_FILES = ("main.py", "deck.csv")
REQUIRED_DIRS = ("cg",)


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    """提出アーカイブに不要な生成物を除外する。"""
    parts = Path(info.name).parts
    if "__pycache__" in parts or info.name.endswith(".pyc"):
        return None
    return info


def agent_src_dir(agent_name: str) -> Path:
    """agent名から提出対象のsrcディレクトリを返す。"""
    return AGENTS_ROOT / agent_name / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        required=True,
        help="提出物を作成するagent名。agents/{agent}/src を対象にする。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力先tar.gz。省略時は dist/submission_{agent}.tar.gz。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_root = agent_src_dir(args.agent)
    submission = args.output or DIST_ROOT / f"submission_{args.agent}.tar.gz"

    if not src_root.is_dir():
        raise FileNotFoundError(f"agentのsrcディレクトリがありません: {src_root}")

    missing = [name for name in REQUIRED_FILES if not (src_root / name).exists()]
    missing += [name for name in REQUIRED_DIRS if not (src_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "提出に必要なファイルがありません: " + ", ".join(missing)
        )

    submission.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(submission, "w:gz") as archive:
        for name in REQUIRED_FILES:
            archive.add(src_root / name, arcname=name, filter=tar_filter)
        for name in REQUIRED_DIRS:
            archive.add(src_root / name, arcname=name, filter=tar_filter)

        # agent固有のPythonパッケージや学習済み重みなど、提出時に必要な追加物を含める。
        for path in sorted(src_root.iterdir()):
            if path.name in {*REQUIRED_FILES, *REQUIRED_DIRS, "__pycache__"}:
                continue
            if path.suffix == ".pyc":
                continue
            archive.add(path, arcname=path.name, filter=tar_filter)

    print(f"作成しました: {submission}")


if __name__ == "__main__":
    main()
