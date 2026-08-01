"""観測カードから相手の60枚デッキを候補DBで推定する（②方式・提出物同梱）。

候補DB(``deck_candidates_by_wins.jsonl``)をこのモジュールと同じ場所に同梱し、
観測できた相手カードに最も合致する実在デッキを返す。tools/deck_generator_dbrecon
の select_best_matching_deck を提出物向けに自己完結化したもの（外部依存なし）。

DBが無い・観測ゼロ・読み込み失敗のときは None を返し、呼び出し側の
固定デッキ辞書へフォールバックできるようにする（提出環境で決して落とさない）。
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import json

DECK_SIZE = 60
_DB_NAME = "deck_candidates_by_wins.jsonl"


def _database_path() -> Path | None:
    candidates = [
        Path(__file__).resolve().parent / _DB_NAME,          # rl_mcts/ と同梱
        Path(__file__).resolve().parents[1] / _DB_NAME,      # src/ 直下に置いた場合
        Path("/kaggle_simulations/agent") / _DB_NAME,        # Kaggle 実行環境
        Path(_DB_NAME),                                       # カレント
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=1)
def _load_records() -> tuple[tuple[list[int], int], ...]:
    """(deck60, wins) のタプル列を1度だけ読み込む。失敗時は空。"""
    path = _database_path()
    if path is None:
        return ()
    records: list[tuple[list[int], int]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                deck = record.get("deck")
                if not isinstance(deck, list) or len(deck) != DECK_SIZE:
                    continue
                records.append(([int(c) for c in deck], int(record.get("wins", 0))))
    except Exception:
        return ()
    return tuple(records)


def predict_full_deck(observed_cards: Iterable[int]) -> list[int] | None:
    """観測カードに最も合致する候補デッキ(60枚)を返す。推定不能なら None。

    合致度＝観測カードとデッキの重複枚数。同点は勝数、さらにDB順で安定化。
    """
    observed = Counter(int(c) for c in observed_cards if int(c) >= 0)
    if not observed:
        return None
    records = _load_records()
    if not records:
        return None

    best_deck: list[int] | None = None
    best_rank: tuple[int, int, int] | None = None
    for index, (deck, wins) in enumerate(records):
        counts = Counter(deck)
        matched = sum(min(count, counts.get(card_id, 0)) for card_id, count in observed.items())
        rank = (matched, wins, -index)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_deck = deck
    return list(best_deck) if best_deck is not None else None
