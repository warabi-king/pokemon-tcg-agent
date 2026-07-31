"""既存の60枚デッキを平均連結法の凝集型階層クラスタリングで分類する。

カードID別の採用枚数からヒストグラム交差類似度を計算し、その補数を距離として
UPGMA（平均連結法）でボトムアップにクラスタを統合する。既存の ``cluster_*``
フィールドは保持し、結果を ``hierarchical_*`` フィールドとしてJSONLへ追加する。

実行例:
    python tools/clustering_deck/hierarchical_clustering.py --clusters 16
    python tools/clustering_deck/hierarchical_clustering.py \
        --input tools/deck_generator/generated/deck_candidates_by_wins.jsonl \
        --output /tmp/hierarchical_decks.jsonl --clusters 40
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TOOLS_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = TOOLS_DIR / "deck_generator" / "generated" / "deck_candidates_by_wins.jsonl"
DECK_SIZE = 60
LINKAGE_NAME = "average"
DISTANCE_METRIC_NAME = "1-histogram-intersection"


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    """指定クラスタ数で切断した階層クラスタリング結果を保持する。"""

    clusters: list[list[int]]
    last_merge_distance: float | None
    next_merge_distance: float | None


@dataclass(frozen=True)
class _ActiveCluster:
    """平均連結法の計算中に有効なクラスタの構成要素を保持する。"""

    member_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        """クラスタに含まれるデッキ数を返す。"""

        return len(self.member_indices)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力JSONL。省略時は入力ファイルを安全に置換する。",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=16,
        help="階層を切断した後のクラスタ数（デフォルト: 16）",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """JSONLから空行を除いてデッキレコードを読み込む。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} のJSONを解析できません。") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} はJSONオブジェクトではありません。")
            records.append(record)
    return records


def count_histogram(record: dict[str, Any]) -> dict[int, int]:
    """デッキレコードをカードID別の採用枚数へ変換し、60枚であることを検証する。"""

    try:
        histogram = {
            int(card_id): int(count)
            for card_id, count in record["deck_counts"].items()
        }
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("deck_countsをカードID別枚数として読み込めません。") from exc
    if any(card_id <= 0 or count <= 0 for card_id, count in histogram.items()):
        raise ValueError("deck_countsのカードIDと枚数は正の整数である必要があります。")
    if sum(histogram.values()) != DECK_SIZE:
        raise ValueError(
            f"階層クラスタリング対象は{DECK_SIZE}枚デッキである必要があります。"
            f"現在: {sum(histogram.values())}枚"
        )
    return histogram


def histogram_intersection_distance(
    left: dict[int, int],
    right: dict[int, int],
) -> float:
    """カード枚数ヒストグラムの交差類似度を1から引いた距離を返す。"""

    if len(left) > len(right):
        left, right = right, left
    overlap = sum(
        min(left_count, right.get(card_id, 0))
        for card_id, left_count in left.items()
    )
    return 1.0 - overlap / DECK_SIZE


def _distance_key(left_cluster_id: int, right_cluster_id: int) -> tuple[int, int]:
    """クラスタIDの順序に依存しない距離辞書用キーを返す。"""

    if left_cluster_id < right_cluster_id:
        return left_cluster_id, right_cluster_id
    return right_cluster_id, left_cluster_id


def _pop_valid_distance(
    distance_heap: list[tuple[float, int, int]],
    active_clusters: dict[int, _ActiveCluster],
    distances: dict[tuple[int, int], float],
) -> tuple[float, int, int] | None:
    """統合済みクラスタを参照する古い候補を除外し、現在有効な最近接対を返す。"""

    while distance_heap:
        distance, left_cluster_id, right_cluster_id = heapq.heappop(distance_heap)
        if left_cluster_id not in active_clusters or right_cluster_id not in active_clusters:
            continue
        current_distance = distances.get(
            _distance_key(left_cluster_id, right_cluster_id)
        )
        if current_distance is None or current_distance != distance:
            continue
        return distance, left_cluster_id, right_cluster_id
    return None


def agglomerative_average_linkage(
    histograms: list[dict[int, int]],
    target_cluster_count: int,
) -> HierarchicalClusteringResult:
    """UPGMAでデッキを統合し、指定クラスタ数になった時点の所属を返す。"""

    deck_count = len(histograms)
    if deck_count == 0:
        raise ValueError("クラスタリング対象のデッキがありません。")
    if not 1 <= target_cluster_count <= deck_count:
        raise ValueError(
            f"clustersは1以上{deck_count}以下で指定してください。"
            f"現在: {target_cluster_count}"
        )

    active_clusters = {
        cluster_id: _ActiveCluster((cluster_id,))
        for cluster_id in range(deck_count)
    }
    distances: dict[tuple[int, int], float] = {}
    distance_heap: list[tuple[float, int, int]] = []
    for left_index in range(deck_count):
        for right_index in range(left_index + 1, deck_count):
            distance = histogram_intersection_distance(
                histograms[left_index], histograms[right_index]
            )
            distances[(left_index, right_index)] = distance
            heapq.heappush(distance_heap, (distance, left_index, right_index))

    next_cluster_id = deck_count
    last_merge_distance: float | None = None
    while len(active_clusters) > target_cluster_count:
        closest = _pop_valid_distance(distance_heap, active_clusters, distances)
        if closest is None:
            raise RuntimeError("統合可能なクラスタ対が見つかりませんでした。")
        merge_distance, left_cluster_id, right_cluster_id = closest
        left_cluster = active_clusters[left_cluster_id]
        right_cluster = active_clusters[right_cluster_id]
        other_cluster_ids = [
            cluster_id
            for cluster_id in active_clusters
            if cluster_id not in (left_cluster_id, right_cluster_id)
        ]

        new_distances: list[tuple[int, float]] = []
        for other_cluster_id in other_cluster_ids:
            left_distance = distances.pop(
                _distance_key(left_cluster_id, other_cluster_id)
            )
            right_distance = distances.pop(
                _distance_key(right_cluster_id, other_cluster_id)
            )
            average_distance = (
                left_cluster.size * left_distance
                + right_cluster.size * right_distance
            ) / (left_cluster.size + right_cluster.size)
            new_distances.append((other_cluster_id, average_distance))

        distances.pop(_distance_key(left_cluster_id, right_cluster_id), None)
        del active_clusters[left_cluster_id]
        del active_clusters[right_cluster_id]
        active_clusters[next_cluster_id] = _ActiveCluster(
            left_cluster.member_indices + right_cluster.member_indices
        )
        for other_cluster_id, average_distance in new_distances:
            key = _distance_key(next_cluster_id, other_cluster_id)
            distances[key] = average_distance
            heapq.heappush(
                distance_heap,
                (average_distance, key[0], key[1]),
            )

        last_merge_distance = merge_distance
        next_cluster_id += 1

    next_merge = _pop_valid_distance(distance_heap, active_clusters, distances)
    next_merge_distance = next_merge[0] if next_merge is not None else None
    clusters = [
        list(cluster.member_indices)
        for cluster in active_clusters.values()
    ]
    return HierarchicalClusteringResult(
        clusters=clusters,
        last_merge_distance=last_merge_distance,
        next_merge_distance=next_merge_distance,
    )


def _stable_deck_key(record: dict[str, Any]) -> str:
    """入力順に依存しない初期デッキ順を作るための正規化キーを返す。"""

    histogram = count_histogram(record)
    normalized = {str(card_id): histogram[card_id] for card_id in sorted(histogram)}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def add_hierarchical_fields(
    records: list[dict[str, Any]],
    target_cluster_count: int,
) -> HierarchicalClusteringResult:
    """階層クラスタを計算し、既存レコードへhierarchicalフィールドを追加する。"""

    if not records:
        raise ValueError("入力JSONLにデッキレコードがありません。")
    ordered_record_indices = sorted(
        range(len(records)),
        key=lambda record_index: _stable_deck_key(records[record_index]),
    )
    histograms = [
        count_histogram(records[record_index])
        for record_index in ordered_record_indices
    ]
    clustering_result = agglomerative_average_linkage(
        histograms,
        target_cluster_count,
    )

    clusters_with_stats: list[dict[str, Any]] = []
    for ordered_member_indices in clustering_result.clusters:
        member_indices = [
            ordered_record_indices[ordered_index]
            for ordered_index in ordered_member_indices
        ]
        games = sum(int(records[index].get("games", 0)) for index in member_indices)
        wins = sum(int(records[index].get("wins", 0)) for index in member_indices)
        losses = sum(int(records[index].get("losses", 0)) for index in member_indices)
        draws = sum(int(records[index].get("draws", 0)) for index in member_indices)
        representative_index = max(
            member_indices,
            key=lambda index: (
                int(records[index].get("games", 0)),
                int(records[index].get("wins", 0)),
                float(records[index].get("win_rate", 0.0)),
                _stable_deck_key(records[index]),
            ),
        )
        clusters_with_stats.append(
            {
                "member_indices": member_indices,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / games if games else 0.0,
                "representative_index": representative_index,
            }
        )

    clusters_with_stats.sort(
        key=lambda cluster: (
            -int(cluster["games"]),
            -int(cluster["wins"]),
            -len(cluster["member_indices"]),
            _stable_deck_key(records[int(cluster["representative_index"])]),
        )
    )
    total_decks = len(records)
    cluster_count = len(clusters_with_stats)
    for cluster_id, cluster in enumerate(clusters_with_stats):
        member_indices = cluster["member_indices"]
        member_count = len(member_indices)
        cluster_weight = total_decks / (member_count * cluster_count)
        for record_index in member_indices:
            record = records[record_index]
            record["hierarchical_cluster_id"] = cluster_id
            record["hierarchical_cluster_members"] = member_count
            record["hierarchical_cluster_games"] = cluster["games"]
            record["hierarchical_cluster_wins"] = cluster["wins"]
            record["hierarchical_cluster_losses"] = cluster["losses"]
            record["hierarchical_cluster_draws"] = cluster["draws"]
            record["hierarchical_cluster_win_rate"] = cluster["win_rate"]
            record["hierarchical_cluster_weight"] = cluster_weight
            record["hierarchical_cluster_representative"] = (
                record_index == cluster["representative_index"]
            )
            record["hierarchical_linkage"] = LINKAGE_NAME
            record["hierarchical_distance_metric"] = DISTANCE_METRIC_NAME
            record["hierarchical_cut_mode"] = "cluster_count"
            record["hierarchical_cluster_count"] = cluster_count
            record["hierarchical_last_merge_distance"] = (
                clustering_result.last_merge_distance
            )
            record["hierarchical_next_merge_distance"] = (
                clustering_result.next_merge_distance
            )
            record["hierarchical_cluster_weight_formula"] = (
                "total_decks/(hierarchical_cluster_members*hierarchical_cluster_count)"
            )

    return clustering_result


def write_records_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """途中失敗で既存JSONLを壊さないよう、一時ファイル経由で原子的に書き換える。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> int:
    """階層クラスタリングを実行し、指定JSONLへ結果を書き込む。"""

    args = parse_args()
    output_path = args.output or args.input
    records = load_records(args.input)
    result = add_hierarchical_fields(records, args.clusters)
    write_records_atomic(output_path, records)
    print(f"read {len(records)} decks from {args.input}")
    print(f"created {len(result.clusters)} hierarchical clusters with average linkage")
    print(f"last merge distance: {result.last_merge_distance}")
    print(f"next merge distance: {result.next_merge_distance}")
    print(f"wrote hierarchical cluster fields to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
