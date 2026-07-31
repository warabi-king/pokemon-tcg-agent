"""deck_generator の候補デッキDBから、参加エージェント（約63体）を生成する。

deck_candidates_by_wins.jsonl の各クラスタ(cluster_id)について、勝利数が最大の
デッキを代表として1体ずつエージェント化する。

出力:
  <ROOT>/clusters.json          … agents: [{name, cluster_id, deck, games, wins, win_rate}]
  <ROOT>/decks/<name>.csv       … 各エージェントの60枚デッキ
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from deck_utils import write_deck_csv


def build_agents(jsonl_path: Path) -> list[dict]:
    records = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    # クラスタごとに wins 最大のレコードを代表に選ぶ（ファイル順に依存しないよう明示的に選択）
    best_by_cluster: dict[int, dict] = {}
    for r in records:
        cid = r["cluster_id"]
        cur = best_by_cluster.get(cid)
        if cur is None or r.get("wins", 0) > cur.get("wins", 0):
            best_by_cluster[cid] = r

    agents: list[dict] = []
    for cid in sorted(best_by_cluster):
        r = best_by_cluster[cid]
        deck = list(r["deck"])
        if len(deck) != 60:
            raise ValueError(f"cluster {cid} の代表デッキが60枚ではありません: {len(deck)}枚")
        agents.append(
            {
                "name": f"cl{cid:02d}",
                "cluster_id": cid,
                "deck": deck,
                "games": r.get("games", 0),
                "wins": r.get("wins", 0),
                "win_rate": r.get("win_rate", 0.0),
            }
        )
    return agents


def generate(jsonl_path: Path, root: Path) -> list[dict]:
    agents = build_agents(jsonl_path)
    root.mkdir(parents=True, exist_ok=True)
    decks_dir = root / "decks"
    decks_dir.mkdir(parents=True, exist_ok=True)
    for a in agents:
        write_deck_csv(decks_dir / f"{a['name']}.csv", a["deck"])
    (root / "clusters.json").write_text(
        json.dumps(
            {"source": str(jsonl_path), "count": len(agents), "agents": agents},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"生成: {len(agents)} エージェント -> {root/'clusters.json'} / {decks_dir}/*.csv")
    return agents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", type=Path, default=config.DECKGEN_JSONL)
    parser.add_argument("--root", type=Path, default=config.ROOT)
    args = parser.parse_args()
    generate(args.jsonl, args.root)


if __name__ == "__main__":
    main()
