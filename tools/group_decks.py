"""複数日分のエピソードデータからデッキの使用頻度を集計し、
主要ポケモンのシグネチャで大まかなアーキタイプ単位にグループ化する。

集計結果を標準出力に表示するとともに、上位グループを
tools/train/preprocess_episodes.py --deck-groups/--target-group で使えるJSONに保存する。

使い方:
    python tools/group_decks.py \
        --episodes day1.zip day2.zip day3.zip \
        --top-n 3 \
        --output deck_groups.json

    少数だけで動作確認する場合:

    python tools/group_decks.py --episodes day1.zip --max-episodes 200 --output deck_groups.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from deck_signature import deck_signature, load_card_info
from episode_io import iter_multi_source


def collect_deck_counts(sources: list[Path], max_episodes: int | None) -> Counter:
    """各ソースの各試合から両プレイヤーのデッキを集計する。"""
    deck_counter: Counter = Counter()
    episode_count = 0
    t0 = time.time()

    for source, name, data in iter_multi_source(sources):
        if name == "manifest.csv":
            continue
        if max_episodes is not None and episode_count >= max_episodes:
            break

        try:
            j = json.loads(data)
            steps = j["steps"]
            deck0 = tuple(sorted(steps[1][0]["action"]))
            deck1 = tuple(sorted(steps[1][1]["action"]))
            if len(deck0) == 60:
                deck_counter[deck0] += 1
            if len(deck1) == 60:
                deck_counter[deck1] += 1
        except Exception:
            continue

        episode_count += 1
        if episode_count % 1000 == 0:
            elapsed = time.time() - t0
            print(f"episodes={episode_count} unique_decks={len(deck_counter)} elapsed={elapsed:.1f}s", flush=True)

    print(f"collected: episodes={episode_count} unique_decks={len(deck_counter)}")
    return deck_counter


def build_groups(deck_counter: Counter, is_pokemon: dict[int, bool]) -> dict[frozenset, dict]:
    """主要ポケモンのシグネチャでグループ化する。"""
    groups: dict[frozenset, dict] = defaultdict(lambda: {"total_games": 0, "decks": []})
    for deck, count in deck_counter.items():
        sig = deck_signature(deck, is_pokemon)
        groups[sig]["total_games"] += count
        groups[sig]["decks"].append({"deck": list(deck), "count": count})
    return groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True, help="日ごとのディレクトリ or .zip(複数指定可)")
    parser.add_argument("--max-episodes", type=int, default=None, help="全ソース合計で読み込むエピソード数の上限")
    parser.add_argument("--top-n", type=int, default=3, help="出力JSONに保存する上位グループ数")
    parser.add_argument("--output", type=Path, default=Path("deck_groups.json"))
    parser.add_argument("--card-data", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_pokemon, card_names = load_card_info(args.card_data)

    deck_counter = collect_deck_counts(args.episodes, args.max_episodes)
    groups = build_groups(deck_counter, is_pokemon)

    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["total_games"])
    total_instances = sum(info["total_games"] for info in groups.values())

    print()
    print(f"=== グループ数: {len(groups)}（元のユニークデッキ数: {len(deck_counter)}）===")
    print()
    for i, (sig, info) in enumerate(ranked[: max(args.top_n, 10)]):
        names = ", ".join(sorted(card_names.get(cid, f"#{cid}") for cid in sig))
        share = info["total_games"] / total_instances * 100 if total_instances else 0.0
        marker = "*" if i < args.top_n else " "
        print(
            f"{marker}{i + 1:2d}位 出現{info['total_games']:5d}回 ({share:4.1f}%) "
            f"デッキ種{len(info['decks']):3d}種: {names}"
        )

    top_groups = []
    for i, (sig, info) in enumerate(ranked[: args.top_n]):
        label = " / ".join(sorted(card_names.get(cid, f"#{cid}") for cid in sig)[:3])
        top_groups.append(
            {
                "group_id": i,
                "label": label,
                "signature": sorted(sig),
                "total_games": info["total_games"],
                "deck_variants": len(info["decks"]),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"top_groups": top_groups}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"saved: {args.output} (top {args.top_n} groups)")


if __name__ == "__main__":
    main()
