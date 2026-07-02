"""Kaggle提出用のsubmission.tar.gzを作成する。"""

from __future__ import annotations

from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src_"  # src_{agent名}
SUBMISSION = ROOT / "submission.tar.gz"
REQUIRED_FILES = ("main.py", "deck.csv")
REQUIRED_DIRS = ("cg",)


def main() -> None:
    missing = [name for name in REQUIRED_FILES if not (SRC_ROOT / name).exists()]
    missing += [name for name in REQUIRED_DIRS if not (SRC_ROOT / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "提出に必要なファイルがありません: " + ", ".join(missing)
        )

    with tarfile.open(SUBMISSION, "w:gz") as archive:
        for name in REQUIRED_FILES:
            archive.add(SRC_ROOT / name, arcname=name)
        for name in REQUIRED_DIRS:
            archive.add(SRC_ROOT / name, arcname=name)

    print(f"作成しました: {SUBMISSION}")


if __name__ == "__main__":
    main()
