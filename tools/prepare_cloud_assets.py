"""Trim staged Docker assets to runnable agent sources and their card images."""

from __future__ import annotations

import sys
import shutil
from pathlib import Path


def main(root: Path) -> None:
    agents = root / "agents"
    cards = root / "cards"
    used_card_ids: set[int] = set()

    for deck_path in agents.glob("*/src/deck.csv"):
        used_card_ids.update(
            int(line.strip())
            for line in deck_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    # Cloud agent modes execute src/main.py and may depend on other files in
    # src/. Remove training and documentation assets but preserve the complete
    # submission-time source tree.
    for agent_dir in agents.iterdir():
        if not agent_dir.is_dir():
            continue
        for path in agent_dir.iterdir():
            if path.name != "src":
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
    for cache in agents.rglob("__pycache__"):
        shutil.rmtree(cache)
    for bytecode in agents.rglob("*.pyc"):
        bytecode.unlink()
    print("Prepared agent sources for Cloud Run; card images are served by Cloud Storage.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
