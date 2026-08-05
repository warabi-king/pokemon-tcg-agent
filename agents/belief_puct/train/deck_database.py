"""デッキ候補DBを安全に選抜・拡張する共通処理。

公式由来のDBは読み取り専用とし、学習・評価から得た候補は別JSONLへ追記する。
各レコードは60枚のカードID、勝敗、出所を持つ。ここではカードの合法性を推測せず、
既存の60枚構成だけを候補として扱う。
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

DECK_SIZE = 60


def deck_key(deck: Iterable[int]) -> tuple[int, ...]:
    """順序に依存しない60枚デッキの重複判定キーを返す。"""

    cards = tuple(sorted(int(card_id) for card_id in deck))
    if len(cards) != DECK_SIZE:
        raise ValueError(f"デッキは{DECK_SIZE}枚必要です: {len(cards)}")
    return cards


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """指定JSONL群から有効な60枚候補を読み、同一構成を統合する。"""

    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    key = deck_key(record["deck"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"無効な候補DB行: {path}:{line_number}") from error
                previous = merged.get(key)
                if previous is None:
                    normalized = dict(record)
                    normalized["deck"] = list(key)
                    merged[key] = normalized
                    continue
                # 同じ構成は件数を足し、出所を失わないようsourcesを併記する。
                for field in ("games", "wins", "losses", "draws"):
                    previous[field] = int(previous.get(field, 0)) + int(record.get(field, 0))
                sources = set(previous.get("sources", []))
                sources.update(record.get("sources", []))
                if record.get("source"):
                    sources.add(str(record["source"]))
                previous["sources"] = sorted(sources)
    return list(merged.values())


def adjusted_score(record: dict[str, Any], prior_games: int = 20) -> float:
    """少数試合の生勝率をクラスタ平均へ縮約した選抜用scoreを返す。"""

    games = int(record.get("games", 0))
    wins = int(record.get("wins", 0))
    draws = int(record.get("draws", 0))
    cluster_score = float(record.get("cluster_win_rate", 0.5))
    return (wins + 0.5 * draws + prior_games * cluster_score) / (games + prior_games)


def select_diverse_records(
    records: Iterable[dict[str, Any]],
    limit: int,
    min_games: int = 20,
) -> list[dict[str, Any]]:
    """十分な試合数を持つ候補から、クラスタ多様性を優先して選ぶ。

    DBには1試合だけのデッキも含まれるため、補正前に最低試合数を適用する。
    条件を満たす候補がなければ、偶然の単発勝利を採用しないよう明示的に失敗する。
    """

    if limit <= 0:
        raise ValueError("候補数は1以上にしてください")
    if min_games <= 0:
        raise ValueError("最低試合数は1以上にしてください")
    eligible = [record for record in records if int(record.get("games", 0)) >= min_games]
    if not eligible:
        raise ValueError(f"最低試合数{min_games}を満たす候補がありません")
    ranked = sorted(eligible, key=lambda record: (-adjusted_score(record), -int(record.get("games", 0))))
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, ...]] = set()
    selected_clusters: set[int] = set()
    for diversity_first in (True, False):
        for record in ranked:
            key = deck_key(record["deck"])
            cluster_id = int(record.get("cluster_id", -1))
            if key in selected_keys or (diversity_first and cluster_id in selected_clusters):
                continue
            selected.append(record)
            selected_keys.add(key)
            selected_clusters.add(cluster_id)
            if len(selected) >= limit:
                return selected
    return selected


def write_records(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """候補DBをJSONLとして新規保存し、保存件数を返す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as destination:
        for record in records:
            normalized = dict(record)
            normalized["deck"] = list(deck_key(normalized["deck"]))
            normalized["deck_counts"] = dict(sorted(Counter(normalized["deck"]).items()))
            destination.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count
