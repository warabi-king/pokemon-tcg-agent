"""切断済みデッキクラスタより上位の階層をデンドログラムとして描画する。

全デッキを葉にすると判読しにくいため、指定クラスタ数で切断した各クラスタを
1つの葉として扱う。葉にはクラスタID、デッキ数、主要タイプ、主要ポケモンを表示し、
平均連結法でそれらが上位階層へ統合される過程をSVGとPNGへ出力する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from hierarchical_clustering import count_histogram, histogram_intersection_distance


@dataclass(frozen=True)
class DendrogramMerge:
    """切断後クラスタ同士を統合した1回分の平均連結結果を保持する。"""

    left_node_id: int
    right_node_id: int
    new_node_id: int
    distance: float
    member_count: int


@dataclass(frozen=True)
class _TreeNode:
    """デンドログラム構築中のノードサイズと子ノードを保持する。"""

    member_count: int
    left_node_id: int | None = None
    right_node_id: int | None = None


def _distance_key(left_node_id: int, right_node_id: int) -> tuple[int, int]:
    """ノード順序に依存しない距離辞書のキーを返す。"""

    if left_node_id < right_node_id:
        return left_node_id, right_node_id
    return right_node_id, left_node_id


def build_cut_cluster_dendrogram(
    records: list[dict[str, Any]],
    cluster_field: str = "hierarchical_cluster_id",
) -> tuple[list[int], list[DendrogramMerge], int]:
    """切断済みクラスタを葉として、その上位の平均連結木を構築する。"""

    if not records:
        raise ValueError("デンドログラム対象のデッキがありません。")
    groups: dict[int, list[int]] = {}
    for record_index, record in enumerate(records):
        try:
            cluster_id = int(record[cluster_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"全レコードに整数の{cluster_field}が必要です。") from exc
        groups.setdefault(cluster_id, []).append(record_index)
    cluster_ids = sorted(groups)
    if len(cluster_ids) < 2:
        raise ValueError("デンドログラムには2クラスタ以上が必要です。")

    histograms = [count_histogram(record) for record in records]
    leaf_node_ids = list(range(len(cluster_ids)))
    cluster_to_leaf = {
        cluster_id: leaf_node_id
        for leaf_node_id, cluster_id in enumerate(cluster_ids)
    }
    active_nodes = {
        cluster_to_leaf[cluster_id]: _TreeNode(member_count=len(groups[cluster_id]))
        for cluster_id in cluster_ids
    }
    distances: dict[tuple[int, int], float] = {}
    for left_position, left_cluster_id in enumerate(cluster_ids):
        for right_cluster_id in cluster_ids[left_position + 1 :]:
            pair_distances = [
                histogram_intersection_distance(
                    histograms[left_index], histograms[right_index]
                )
                for left_index in groups[left_cluster_id]
                for right_index in groups[right_cluster_id]
            ]
            distances[
                _distance_key(
                    cluster_to_leaf[left_cluster_id],
                    cluster_to_leaf[right_cluster_id],
                )
            ] = fmean(pair_distances)

    merges: list[DendrogramMerge] = []
    next_node_id = len(cluster_ids)
    while len(active_nodes) > 1:
        (left_node_id, right_node_id), merge_distance = min(
            distances.items(),
            key=lambda item: (item[1], item[0]),
        )
        left_node = active_nodes[left_node_id]
        right_node = active_nodes[right_node_id]
        other_node_ids = [
            node_id
            for node_id in active_nodes
            if node_id not in (left_node_id, right_node_id)
        ]
        new_distances: dict[int, float] = {}
        for other_node_id in other_node_ids:
            left_distance = distances.pop(
                _distance_key(left_node_id, other_node_id)
            )
            right_distance = distances.pop(
                _distance_key(right_node_id, other_node_id)
            )
            new_distances[other_node_id] = (
                left_node.member_count * left_distance
                + right_node.member_count * right_distance
            ) / (left_node.member_count + right_node.member_count)

        distances.pop(_distance_key(left_node_id, right_node_id), None)
        del active_nodes[left_node_id]
        del active_nodes[right_node_id]
        member_count = left_node.member_count + right_node.member_count
        active_nodes[next_node_id] = _TreeNode(
            member_count=member_count,
            left_node_id=left_node_id,
            right_node_id=right_node_id,
        )
        for other_node_id, distance in new_distances.items():
            distances[_distance_key(next_node_id, other_node_id)] = distance
        merges.append(
            DendrogramMerge(
                left_node_id=left_node_id,
                right_node_id=right_node_id,
                new_node_id=next_node_id,
                distance=merge_distance,
                member_count=member_count,
            )
        )
        next_node_id += 1

    root_node_id = next(iter(active_nodes))
    return cluster_ids, merges, root_node_id


def _leaf_order(
    node_id: int,
    merge_by_node_id: dict[int, DendrogramMerge],
) -> list[int]:
    """統合木を左からたどり、描画に使う葉ノード順を返す。"""

    merge = merge_by_node_id.get(node_id)
    if merge is None:
        return [node_id]
    return _leaf_order(merge.left_node_id, merge_by_node_id) + _leaf_order(
        merge.right_node_id, merge_by_node_id
    )


def _cluster_labels(cluster_summaries: list[dict[str, Any]]) -> dict[int, str]:
    """評価レポートから各葉に表示する短い日本語ラベルを作る。"""

    labels: dict[int, str] = {}
    for cluster in cluster_summaries:
        pokemon_names = "・".join(
            item["name"] for item in cluster.get("top_pokemon", [])[:2]
        )
        labels[int(cluster["cluster_id"])] = (
            f"C{cluster['cluster_id']}  n={cluster['members']}  "
            f"{cluster['dominant_pokemon_type']}  {pokemon_names}"
        )
    return labels


def render_cut_cluster_dendrogram(
    records: list[dict[str, Any]],
    cluster_summaries: list[dict[str, Any]],
    output_svg: Path,
    output_png: Path,
    cluster_field: str = "hierarchical_cluster_id",
) -> None:
    """切断位置より上を要約した横向きデンドログラムをSVGとPNGへ保存する。"""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("デンドログラム描画にはmatplotlibが必要です。") from exc

    cluster_ids, merges, root_node_id = build_cut_cluster_dendrogram(
        records, cluster_field=cluster_field
    )
    merge_by_node_id = {merge.new_node_id: merge for merge in merges}
    leaf_order = _leaf_order(root_node_id, merge_by_node_id)
    leaf_to_cluster_id = {
        leaf_node_id: cluster_id
        for leaf_node_id, cluster_id in enumerate(cluster_ids)
    }
    labels = _cluster_labels(cluster_summaries)

    plt.rcParams["font.family"] = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "sans-serif",
    ]
    figure_height = max(8.0, len(cluster_ids) * 0.38)
    figure, axis = plt.subplots(figsize=(15, figure_height))
    coordinates: dict[int, tuple[float, float]] = {
        leaf_node_id: (0.0, float(position))
        for position, leaf_node_id in enumerate(leaf_order)
    }
    for merge in merges:
        left_x, left_y = coordinates[merge.left_node_id]
        right_x, right_y = coordinates[merge.right_node_id]
        merge_y = (left_y + right_y) / 2
        axis.plot(
            [left_x, merge.distance, merge.distance, right_x],
            [left_y, left_y, right_y, right_y],
            color="#4b5563",
            linewidth=1.2,
        )
        coordinates[merge.new_node_id] = (merge.distance, merge_y)

    ordered_cluster_ids = [leaf_to_cluster_id[leaf_id] for leaf_id in leaf_order]
    axis.set_yticks(range(len(ordered_cluster_ids)))
    axis.set_yticklabels(
        [labels.get(cluster_id, f"C{cluster_id}") for cluster_id in ordered_cluster_ids]
    )
    last_merge_values = {
        record.get("hierarchical_last_merge_distance") for record in records
    }
    next_merge_values = {
        record.get("hierarchical_next_merge_distance") for record in records
    }
    last_merge_distance = next(iter(last_merge_values)) if len(last_merge_values) == 1 else None
    next_merge_distance = next(iter(next_merge_values)) if len(next_merge_values) == 1 else None
    if last_merge_distance is not None and next_merge_distance is not None:
        cut_distance = (float(last_merge_distance) + float(next_merge_distance)) / 2
        axis.axvline(
            cut_distance,
            color="#dc2626",
            linestyle="--",
            linewidth=1.2,
            label=f"{len(cluster_ids)}クラスタ切断位置",
        )
        axis.legend(loc="lower right")

    axis.set_title(f"平均連結法による階層クラスタリング（{len(cluster_ids)}クラスタで切断）")
    axis.set_xlabel("クラスタ間距離（1 - カード構成類似度）")
    axis.set_ylabel("切断位置のクラスタ")
    axis.grid(axis="x", color="#d1d5db", linewidth=0.6, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.tight_layout()
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_svg, format="svg", bbox_inches="tight")
    figure.savefig(output_png, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
