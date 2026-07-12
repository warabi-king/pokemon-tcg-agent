"""Run the local-only advanced deck builder.

This server is intentionally separate from app/server.py and cloud_server.py so
local deck-building changes do not affect the GCP deployment surface.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
DATA_PATH = ROOT / "data" / "JP_Card_Data.csv"
CARD_ROOT = ROOT / "docs" / "card-assets" / "cards"

COL_ID = "カード ID"
COL_NAME = "カード名"
COL_EXPANSION = "エキスパンションマーク"
COL_COLLECTION = "コレクション番号"
COL_KIND = "ポケモンの進化の段階/エネルギー・トレーナーズの種類"
COL_RULE = "ルール"
COL_CATEGORY = "カテゴリ"
COL_EVOLVES_FROM = "進化前"
COL_HP = "HP"
COL_TYPE = "タイプ"
COL_WEAKNESS = "弱点"
COL_RESISTANCE = "抵抗力"
COL_RETREAT = "にげる"
COL_ATTACK = "ワザ名"
COL_COST = "コスト"
COL_DAMAGE = "ダメージ"
COL_EFFECT = "効果の説明"

FILTER_FIELDS = {
    "kinds": COL_KIND,
    "rules": COL_RULE,
    "categories": COL_CATEGORY,
    "types": COL_TYPE,
}


def clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value == "n/a" else value


def append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def card_search_text(card: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "id",
        "name",
        "expansion",
        "collectionNumber",
        "kind",
        "rule",
        "category",
        "evolvesFrom",
        "hp",
        "type",
        "weakness",
        "resistance",
        "retreat",
    ):
        parts.append(str(card.get(key, "")))
    for attack in card["attacks"]:
        parts.extend(str(attack.get(key, "")) for key in ("name", "cost", "damage", "effect"))
    return " ".join(parts).lower()


def load_cards() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Card data not found: {DATA_PATH}")
    if not CARD_ROOT.is_dir():
        raise FileNotFoundError(f"Card image directory not found: {CARD_ROOT}")

    cards_by_id: OrderedDict[int, dict[str, Any]] = OrderedDict()
    filters = {name: [] for name in FILTER_FIELDS}

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            card_id = int(clean(row[COL_ID]))
            card = cards_by_id.get(card_id)
            if card is None:
                card = {
                    "id": card_id,
                    "name": clean(row[COL_NAME]),
                    "expansion": clean(row[COL_EXPANSION]),
                    "collectionNumber": clean(row[COL_COLLECTION]),
                    "kind": clean(row[COL_KIND]),
                    "rule": clean(row[COL_RULE]),
                    "category": clean(row[COL_CATEGORY]),
                    "evolvesFrom": clean(row[COL_EVOLVES_FROM]),
                    "hp": clean(row[COL_HP]),
                    "type": clean(row[COL_TYPE]),
                    "weakness": clean(row[COL_WEAKNESS]),
                    "resistance": clean(row[COL_RESISTANCE]),
                    "retreat": clean(row[COL_RETREAT]),
                    "attacks": [],
                    "image": f"/cards/{card_id:04d}.webp",
                }
                cards_by_id[card_id] = card
                for filter_name, column in FILTER_FIELDS.items():
                    append_unique(filters[filter_name], clean(row[column]))

            attack_name = clean(row[COL_ATTACK])
            effect = clean(row[COL_EFFECT])
            if attack_name or effect:
                attack = {
                    "name": attack_name,
                    "cost": clean(row[COL_COST]),
                    "damage": clean(row[COL_DAMAGE]),
                    "effect": effect,
                }
                if attack not in card["attacks"]:
                    card["attacks"].append(attack)

    cards = list(cards_by_id.values())
    for card in cards:
        card["searchText"] = card_search_text(card)
    for values in filters.values():
        values.sort()
    return cards, filters


app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(APP_ROOT, "local_deck_builder.html")


@app.get("/static/<path:filename>")
def static_file(filename: str):
    return send_from_directory(APP_ROOT, filename)


@app.get("/cards/<path:filename>")
def card_image(filename: str):
    return send_from_directory(CARD_ROOT, filename)


@app.get("/api/cards")
def cards_api():
    cards, filters = load_cards()
    return jsonify({"cards": cards, "filters": filters})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Advanced local deck builder: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
