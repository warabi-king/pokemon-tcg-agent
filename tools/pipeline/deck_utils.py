"""デッキ入出力と類似度（ヒストグラム交差）の共通ロジック。

`deck_completion.py cluster-weights` と同じ「カードIDヒストグラム交差 / 60」を使い、
- 近いデッキ判定（既存資産のコピー元探索）
- Phase0 のクラスタ最近傍割当フィルタ
に用いる。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

DECK_SIZE = 60


def read_deck_csv(path: str | Path) -> list[int]:
    deck = [int(line.strip()) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != DECK_SIZE:
        raise ValueError(f"deck.csv は {DECK_SIZE} 枚である必要があります: {path} ({len(deck)}枚)")
    return deck


def write_deck_csv(path: str | Path, deck: list[int]) -> None:
    if len(deck) != DECK_SIZE:
        raise ValueError(f"deck は {DECK_SIZE} 枚である必要があります: {len(deck)}枚")
    Path(path).write_text("\n".join(str(c) for c in deck) + "\n", encoding="utf-8")


def hist_intersection(a: Iterable[int], b: Iterable[int]) -> float:
    """カードIDヒストグラム交差類似度: sum(min(countA, countB)) / 60。"""
    ca, cb = Counter(a), Counter(b)
    return sum(min(ca[k], cb[k]) for k in set(ca) | set(cb)) / float(DECK_SIZE)


def nearest_deck(deck: list[int], candidates: list[dict], deck_key: str = "deck") -> tuple[dict | None, float]:
    """candidates（各 dict は deck_key に60枚リストを持つ）から最も近いものと類似度を返す。"""
    best: dict | None = None
    best_sim = -1.0
    for cand in candidates:
        sim = hist_intersection(deck, cand[deck_key])
        if sim > best_sim:
            best, best_sim = cand, sim
    return best, best_sim


def same_deck(a: list[int], b: list[int]) -> bool:
    """完全一致（枚数まで同じ）。リーグ内の exact-deck フィルタ用。"""
    return Counter(a) == Counter(b)


# ---- deck_filter ビルダー（preprocess に渡す Callable[[your, opp], bool]）----

def exact_deck_filter(target_deck: list[int], role: str):
    """リーグ履歴用: 自分/相手のデッキが target と完全一致する手だけ集める。"""
    tgt = Counter(target_deck)

    def _filter(your_deck: list[int], opponent_deck: list[int]) -> bool:
        deck = your_deck if role == "own" else opponent_deck
        return Counter(deck) == tgt

    return _filter


def cluster_filter(target_deck: list[int], all_reps: list[dict], threshold: float, role: str):
    """Phase0 用: デッキを代表デッキ集合へ最近傍割当し、target クラスタに属する手だけ集める。

    all_reps は各エージェントの代表 dict（"name","deck" を持つ）。role 側のデッキの
    最近傍が target_deck（= このエージェントの代表）で、かつ類似度 >= threshold のとき採用。
    """
    target_key = tuple(sorted(target_deck))

    def _filter(your_deck: list[int], opponent_deck: list[int]) -> bool:
        deck = your_deck if role == "own" else opponent_deck
        rep, sim = nearest_deck(deck, all_reps)
        if rep is None or sim < threshold:
            return False
        return tuple(sorted(rep["deck"])) == target_key

    return _filter


def load_agents(clusters_json: str | Path) -> list[dict]:
    """gen_agents が出力した clusters.json から agents リストを読む。"""
    data = json.loads(Path(clusters_json).read_text(encoding="utf-8"))
    return data["agents"]
