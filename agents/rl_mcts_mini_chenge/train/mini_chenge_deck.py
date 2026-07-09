"""デッキ内のm枚をランダムなカードへ変更し、別ファイルに保存する。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from pathlib import Path
import random
import re

AGENT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AGENT_ROOT / "src"
TRAIN_ROOT = Path(__file__).resolve().parent
CARD_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "EN_Card_Data.csv"

DECK_SIZE = 60
MAX_GENERATION_ATTEMPTS = 10_000
DECK_FILE_PATTERN = re.compile(r"deck_(\d+)\.csv")


@dataclass(frozen=True)
class CardInfo:
    """デッキ構築ルールの検証に必要なカード情報。"""

    card_id: int
    name: str
    card_type: str
    basic: bool
    ace_spec: bool


def read_card_data(path: Path) -> list[CardInfo]:
    """カードデータCSVを読み込み、カードIDごとに1件へまとめる。"""
    cards: dict[int, CardInfo] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            card_id = int(row["Card ID"])
            if card_id not in cards:
                card_type = row["Stage (Pokémon)/Type (Energy and Trainer)"]
                cards[card_id] = CardInfo(
                    card_id=card_id,
                    name=row["Card Name"],
                    card_type=card_type,
                    basic=card_type == "Basic Pokémon",
                    ace_spec=row["Rule"] == "ACE SPEC",
                )
    if not cards:
        raise ValueError(f"カードデータが空です: {path}")
    return list(cards.values())


def read_deck(path: Path) -> list[int]:
    """1行1カードIDのデッキCSVを読み込む。"""
    try:
        deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except ValueError as error:
        raise ValueError(f"カードIDではない値が含まれています: {path}") from error

    if len(deck) != DECK_SIZE:
        raise ValueError(f"デッキは{DECK_SIZE}枚である必要があります: {path} ({len(deck)}枚)")
    return deck


def validate_deck(deck: list[int], card_map: dict[int, CardInfo]) -> bool:
    """ゲームの主要なデッキ構築ルールを満たすか確認する。"""
    if len(deck) != DECK_SIZE or any(card_id not in card_map for card_id in deck):
        return False

    cards = [card_map[card_id] for card_id in deck]
    name_counts = Counter(
        card.name for card in cards if card.card_type != "Basic Energy"
    )
    if any(count > 4 for count in name_counts.values()):
        return False
    if sum(card.ace_spec for card in cards) > 1:
        return False
    return any(card.basic for card in cards)


def change_deck(
    deck: list[int],
    m: int,
    cards: list[CardInfo],
    rng: random.Random,
) -> list[int]:
    """m個の異なる位置をランダムな別カードへ変更する。"""
    if not 1 <= m <= len(deck):
        raise ValueError(f"mは1から{len(deck)}の範囲で指定してください: {m}")

    card_ids = [card.card_id for card in cards]
    card_map = {card.card_id: card for card in cards}
    if any(card_id not in card_map for card_id in deck):
        raise ValueError("元デッキに無効なカードIDが含まれています。")

    for _ in range(MAX_GENERATION_ATTEMPTS):
        changed = deck.copy()
        positions = rng.sample(range(len(deck)), m)
        for position in positions:
            original_card_id = deck[position]
            replacement = rng.choice(card_ids)
            while replacement == original_card_id:
                replacement = rng.choice(card_ids)
            changed[position] = replacement

        if validate_deck(changed, card_map):
            return changed

    raise RuntimeError(
        f"{MAX_GENERATION_ATTEMPTS}回試行しましたが、有効なデッキを生成できませんでした。"
        "mを小さくして再実行してください。"
    )


def next_output_path(output_dir: Path) -> Path:
    """deck_N.csvのうち、未使用の最小番号を返す。"""
    used_numbers = {
        int(match.group(1))
        for path in output_dir.glob("deck_*.csv")
        if (match := DECK_FILE_PATTERN.fullmatch(path.name))
    }
    number = 1
    while number in used_numbers:
        number += 1
    return output_dir / f"deck_{number}.csv"


def write_deck(output_dir: Path, deck: list[int]) -> Path:
    """番号が重複しないファイルへデッキを書き込む。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        output_path = next_output_path(output_dir)
        try:
            with output_path.open("x", encoding="utf-8", newline="") as file:
                file.writelines(f"{card_id}\n" for card_id in deck)
            return output_path
        except FileExistsError:
            # 並行実行で同じ番号が先に作られた場合は、次の番号を探す。
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m", type=int, help="変更するカード枚数")
    parser.add_argument(
        "--input",
        type=Path,
        default=SRC_ROOT / "deck.csv",
        help="変更元のデッキCSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRAIN_ROOT,
        help="deck_N.csvの保存先",
    )
    parser.add_argument("--seed", type=int, default=None, help="乱数seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deck = read_deck(args.input)
    cards = read_card_data(CARD_DATA_PATH)
    changed_deck = change_deck(deck, args.m, cards, random.Random(args.seed))
    output_path = write_deck(args.output_dir, changed_deck)
    print(output_path)


if __name__ == "__main__":
    main()
