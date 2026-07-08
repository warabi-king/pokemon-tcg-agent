"""Trim staged Docker assets to decks and card images used by those decks."""

from __future__ import annotations

import sys
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

    for path in agents.rglob("*"):
        if path.is_file() and path.name != "deck.csv":
            path.unlink()
    for path in cards.glob("*.jpg"):
        if int(path.stem) not in used_card_ids:
            path.unlink()

    print(f"Prepared {len(used_card_ids)} card images for Cloud Run.")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
