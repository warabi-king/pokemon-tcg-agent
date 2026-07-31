"""tools/deck_generator の候補デッキDBから、参加エージェント（16体）を生成する。

deck_candidates_by_wins.jsonl は tools/clustering_deck/run_hierarchical_analysis.py
（平均連結法による階層クラスタリング、既定16クラスタ）によって
hierarchical_cluster_id / hierarchical_cluster_representative 等のフィールドが
付与済みであることを前提とする。各 hierarchical_cluster_id について
hierarchical_cluster_representative=true のレコード（対戦数加重の中心性が最大付近で
最も使用実績のある実在デッキ、詳細は tools/clustering_deck/README.md）を代表として
1体ずつエージェント化する。

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

    representatives: dict[int, dict] = {}
    for r in records:
        if not r.get("hierarchical_cluster_representative"):
            continue
        cid = r["hierarchical_cluster_id"]
        if cid in representatives:
            raise ValueError(
                f"hierarchical_cluster_id={cid} の代表デッキが複数あります。"
                " tools/clustering_deck/run_hierarchical_analysis.py の出力を確認してください。"
            )
        representatives[cid] = r

    if not representatives:
        raise ValueError(
            f"{jsonl_path} に hierarchical_cluster_representative=true のレコードがありません。"
            " 先に tools/clustering_deck/run_hierarchical_analysis.py を実行してください。"
        )

    agents: list[dict] = []
    for cid in sorted(representatives):
        r = representatives[cid]
        deck = list(r["deck"])
        if len(deck) != 60:
            raise ValueError(f"cluster {cid} の代表デッキが60枚ではありません: {len(deck)}枚")
        agents.append(
            {
                "name": f"cl{cid:02d}",
                "cluster_id": cid,
                "deck": deck,
                "games": r.get("hierarchical_cluster_games", r.get("games", 0)),
                "wins": r.get("hierarchical_cluster_wins", r.get("wins", 0)),
                "win_rate": r.get("hierarchical_cluster_win_rate", r.get("win_rate", 0.0)),
                "is_major_cluster": bool(r.get("hierarchical_is_major_cluster", False)),
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
