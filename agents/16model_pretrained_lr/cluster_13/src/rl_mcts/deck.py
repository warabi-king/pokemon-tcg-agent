from __future__ import annotations

from pathlib import Path


def read_deck_csv() -> list[int]:
    """提出環境またはローカル環境のdeck.csvを読み込む。"""
    candidates = [
        Path("deck.csv"),
        Path(__file__).resolve().parents[1] / "deck.csv",
        Path("/kaggle_simulations/agent/deck.csv"),
    ]
    for path in candidates:
        if path.exists():
            deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
            if len(deck) != 60:
                raise ValueError(f"deck.csvは60枚である必要があります: {path} ({len(deck)}枚)")
            return deck
    raise FileNotFoundError("deck.csvが見つかりません。")
