"""デッキを『主要ポケモン(2枚以上採用)』の集合でシグネチャ化する共通ロジック。

トレーナーズ/エネルギーはデッキの型を特徴づけないため除外し、進化ラインの主力ポケモン
(通常4枚積み)だけを残すことで、エネルギー配分違いなどの表記ゆれを吸収した
大まかなアーキタイプ単位にまとめる。
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARD_DATA = REPO_ROOT / "data" / "EN_Card_Data.csv"

POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}


def load_card_info(card_data_csv: Path | None = None) -> tuple[dict[int, bool], dict[int, str]]:
    """カードID -> ポケモンかどうか、カードID -> カード名 を読み込む。"""
    path = card_data_csv or DEFAULT_CARD_DATA
    is_pokemon: dict[int, bool] = {}
    card_names: dict[int, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = int(row["Card ID"])
            stage = row["Stage (Pokémon)/Type (Energy and Trainer)"]
            is_pokemon[cid] = stage in POKEMON_STAGES
            card_names[cid] = row["Card Name"]
    return is_pokemon, card_names


def deck_signature(
    deck: list[int] | tuple[int, ...],
    is_pokemon: dict[int, bool],
    min_count: int = 2,
) -> frozenset[int]:
    """デッキから『min_count枚以上採用されているポケモン』のID集合を取り出す。"""
    counts = Counter(deck)
    return frozenset(cid for cid, c in counts.items() if c >= min_count and is_pokemon.get(cid, False))
