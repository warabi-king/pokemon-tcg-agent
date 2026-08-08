"""エピソードJSON(複数日分)を模倣学習サンプルに変換し、シャード(.pkl)としてディスクに保存する。

一度に全サンプルをメモリに載せる代わりに --shard-size 件たまるごとにディスクへ書き出すため、
数万〜数十万試合規模のデータでもメモリを使い切らずに前処理できる。書き出したシャードは
train_imitation.py がストリーミングで読み込む。

--deck-groups/--target-group を指定すると、tools/group_decks.py が出力したグループのうち
指定した1グループのデッキを使った対戦データだけを抽出する。

--role で、絞り込みをどちらの視点で行うか選べる。
"own"(既定)は自分のデッキが対象グループのプレイヤーの手を集める
(そのデッキアーキタイプを使うagent用)。"opponent"は対象グループのデッキを
相手にしたプレイヤーの手を集める(そのデッキアーキタイプに対する応手を
学ぶagent用)。同じ対象グループに対して両方を前処理すれば、
「アーキタイプを使うagent」と「アーキタイプと戦うagent」を1対1で用意できる。

使い方:
    # 全データを対象に前処理
    python tools/train/preprocess_episodes.py \
        --episodes day1.zip day2.zip day3.zip \
        --output-dir shards/all \
        --shard-size 20000

    # デッキグループ0番を使う側だけを対象に前処理
    python tools/train/preprocess_episodes.py \
        --episodes day1.zip day2.zip day3.zip \
        --deck-groups deck_groups.json --target-group 0 \
        --output-dir shards/group0 \
        --shard-size 20000

    # デッキグループ0番と対戦した相手側だけを対象に前処理
    python tools/train/preprocess_episodes.py \
        --episodes day1.zip day2.zip day3.zip \
        --deck-groups deck_groups.json --target-group 0 --role opponent \
        --output-dir shards/group0_opponent \
        --shard-size 20000

    少数だけで動作確認する場合:

    python tools/train/preprocess_episodes.py \
        --episodes day1.zip --max-episodes 50 --output-dir shards/test
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from deck_signature import deck_signature, load_card_info  # noqa: E402
from episode_io import iter_multi_source  # noqa: E402
from imitation_data import extract_samples_from_episode  # noqa: E402


def build_deck_filter(deck_groups_path: Path, target_group: int, role: str):
    data = json.loads(deck_groups_path.read_text(encoding="utf-8"))
    group = next((g for g in data["top_groups"] if g["group_id"] == target_group), None)
    if group is None:
        raise SystemExit(f"group_id={target_group} が {deck_groups_path} に見つかりません。")

    target_signature = frozenset(group["signature"])
    is_pokemon, _ = load_card_info()

    def deck_filter(your_deck: list[int], opponent_deck: list[int]) -> bool:
        deck = your_deck if role == "own" else opponent_deck
        return deck_signature(deck, is_pokemon) == target_signature

    label = group["label"] if role == "own" else f"{group['label']} 対戦相手"
    return deck_filter, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True, help="日ごとのディレクトリ or .zip(複数指定可)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=20000, help="1シャードあたりのサンプル数上限")
    parser.add_argument("--max-episodes", type=int, default=None, help="全ソース合計で読み込むエピソード数の上限")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--deck-groups", type=Path, default=None, help="tools/group_decks.pyが出力したJSON")
    parser.add_argument("--target-group", type=int, default=None, help="--deck-groups内のgroup_id")
    parser.add_argument(
        "--role",
        choices=["own", "opponent"],
        default="own",
        help="own: 対象グループのデッキを使うプレイヤーの手を集める(既定)。"
        "opponent: 対象グループのデッキと対戦した相手プレイヤーの手を集める。",
    )
    parser.add_argument(
        "--first-move-discount",
        type=float,
        default=1.0,
        help="value教師=割引リターンで、対局ごとに『最初の手の割引がこの値』になるよう"
        "割引率を決める(既定1.0=割引なし=全局面に最終結果をそのまま付与)。0.3にすると"
        "対局長に依らず最初の手≈0.3・終局直前=full±1になる(pipelineのPIPE_FIRST_MOVE_DISCOUNT"
        "既定は0.3)。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.deck_groups is None) != (args.target_group is None):
        raise SystemExit("--deck-groups と --target-group は両方指定するか、両方省略してください。")
    if args.deck_groups is None and args.role != "own":
        raise SystemExit("--role opponent は --deck-groups/--target-group と併用してください。")

    deck_filter = None
    label = "全デッキ"
    if args.deck_groups is not None:
        deck_filter, label = build_deck_filter(args.deck_groups, args.target_group, args.role)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list = []
    shard_index = 0
    total_samples = 0
    episode_count = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buffer, shard_index
        if not buffer:
            return
        shard_path = args.output_dir / f"shard_{shard_index:05d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        shard_index += 1
        buffer = []

    print(f"target: {label}", flush=True)

    stop = False
    for _source, name, data in iter_multi_source(args.episodes):
        if stop:
            break
        if name == "manifest.csv":
            continue
        if args.max_episodes is not None and episode_count >= args.max_episodes:
            break

        try:
            samples = extract_samples_from_episode(
                data, deck_filter=deck_filter, first_move_discount=args.first_move_discount
            )
        except Exception:
            samples = []

        buffer.extend(samples)
        total_samples += len(samples)
        episode_count += 1

        if len(buffer) >= args.shard_size:
            flush()

        if episode_count % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"episodes={episode_count} samples={total_samples} shards={shard_index} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if args.max_samples is not None and total_samples >= args.max_samples:
            stop = True

    flush()

    manifest = {
        "label": label,
        "role": args.role,
        "episodes": episode_count,
        "samples": total_samples,
        "shards": shard_index,
        "shard_size": args.shard_size,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"done: episodes={episode_count} samples={total_samples} shards={shard_index}")
    print(f"saved manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
