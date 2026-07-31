"""階層型デッキクラスタリングを定量・定性の両面から評価する。

クラスタ内外のカード構成類似度、シルエット係数、最近傍整合率、境界上の
高類似ペアを定量評価する。さらに日本語カードデータを使い、主要ポケモンの
タイプ純度とポケモン構成の多重集合Jaccard類似度を定性評価の補助指標として
集計する。

実行例:
    python tools/clustering_deck/evaluate_hierarchical_clustering.py
    python tools/clustering_deck/evaluate_hierarchical_clustering.py \
        --input tools/deck_generator/generated/deck_candidates_by_wins.jsonl \
        --output-json tools/clustering_deck/generated/hierarchical_cluster_evaluation_16.json \
        --output-markdown tools/clustering_deck/generated/hierarchical_cluster_evaluation_16.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from hierarchical_clustering import (
    DEFAULT_INPUT,
    count_histogram,
    histogram_intersection_distance,
    load_records,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CARD_DATA = ROOT / "data" / "JP_Card_Data.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_CLUSTER_FIELD = "hierarchical_cluster_id"
DEFAULT_BOUNDARY_SIMILARITY = 0.75


@dataclass(frozen=True)
class CardMetadata:
    """定性評価に必要な日本語カード属性を保持する。"""

    card_id: int
    name: str
    kind: str
    pokemon_type: str

    @property
    def is_pokemon(self) -> bool:
        """カードがポケモンとして分類されているかを返す。"""

        return self.kind.startswith("ポケモン/")


def parse_args() -> argparse.Namespace:
    """評価コマンドの引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="評価対象JSONL")
    parser.add_argument(
        "--card-data",
        type=Path,
        default=DEFAULT_CARD_DATA,
        help="日本語カードデータCSV",
    )
    parser.add_argument(
        "--cluster-field",
        default=DEFAULT_CLUSTER_FIELD,
        help="クラスタIDを保持するフィールド名",
    )
    parser.add_argument(
        "--boundary-similarity",
        type=float,
        default=DEFAULT_BOUNDARY_SIMILARITY,
        help="別クラスタ高類似ペアとして数える類似度閾値",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def load_card_metadata(path: Path) -> dict[int, CardMetadata]:
    """日本語カードCSVを読み、カードID別の属性辞書を返す。"""

    metadata: dict[int, CardMetadata] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as card_file:
        for row in csv.DictReader(card_file):
            card_id = int(row["カード ID"])
            metadata[card_id] = CardMetadata(
                card_id=card_id,
                name=row["カード名"],
                kind=row["ポケモンの進化の段階/エネルギー・トレーナーズの種類"],
                pokemon_type=row["タイプ"],
            )
    return metadata


def _quantile(values: list[float], probability: float) -> float:
    """ソート済み要素の最近傍を使って指定確率の分位点を返す。"""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = round(probability * (len(sorted_values) - 1))
    return sorted_values[index]


def _distribution(values: list[float]) -> dict[str, float | int]:
    """類似度などの配列を比較しやすい基本統計量へ変換する。"""

    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "minimum": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": len(values),
        "mean": fmean(values),
        "minimum": min(values),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "maximum": max(values),
    }


def _pokemon_histogram(
    deck_histogram: dict[int, int],
    card_metadata: dict[int, CardMetadata],
) -> dict[int, int]:
    """60枚デッキからポケモンカードだけの枚数ヒストグラムを返す。"""

    return {
        card_id: count
        for card_id, count in deck_histogram.items()
        if card_metadata.get(card_id) is not None
        and card_metadata[card_id].is_pokemon
    }


def _multiset_jaccard(left: dict[int, int], right: dict[int, int]) -> float:
    """採用枚数を考慮した多重集合Jaccard類似度を返す。"""

    card_ids = set(left) | set(right)
    union = sum(max(left.get(card_id, 0), right.get(card_id, 0)) for card_id in card_ids)
    if union == 0:
        return 0.0
    intersection = sum(
        min(left.get(card_id, 0), right.get(card_id, 0)) for card_id in card_ids
    )
    return intersection / union


def _dominant_pokemon_type_signature(
    pokemon_histogram: dict[int, int],
    card_metadata: dict[int, CardMetadata],
) -> str:
    """デッキ内で採用枚数が最大となるポケモンタイプの組を返す。"""

    type_counts: Counter[str] = Counter()
    for card_id, count in pokemon_histogram.items():
        pokemon_type = card_metadata[card_id].pokemon_type or "不明"
        type_counts[pokemon_type] += count
    if not type_counts:
        return "ポケモンなし"
    maximum_count = max(type_counts.values())
    return "+".join(
        sorted(
            pokemon_type
            for pokemon_type, count in type_counts.items()
            if count == maximum_count
        )
    )


def _card_category(metadata: CardMetadata | None) -> str:
    """カード属性をクラスタ評価用の日本語カテゴリへ正規化する。"""

    if metadata is None:
        return "その他"
    if metadata.is_pokemon:
        return "ポケモン"
    if "エネルギー" in metadata.kind:
        return "エネルギー"
    if metadata.kind in {"グッズ", "サポート", "スタジアム", "ポケモンのどうぐ"}:
        return metadata.kind
    return metadata.kind if metadata.kind and metadata.kind != "n/a" else "その他"


def _ratio_distribution(
    counts: Counter[str],
    label_key: str,
) -> list[dict[str, str | int | float]]:
    """カテゴリ別枚数を、枚数の多い順の割合付きリストへ変換する。"""

    total = sum(counts.values())
    return [
        {
            label_key: label,
            "copies": count,
            "ratio": count / total if total else 0.0,
        }
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _build_similarity_matrix(
    histograms: list[dict[int, int]],
) -> list[list[float]]:
    """全デッキ総当たりのカード構成類似度行列を作る。"""

    deck_count = len(histograms)
    similarities = [[1.0] * deck_count for _ in range(deck_count)]
    for left_index in range(deck_count):
        for right_index in range(left_index + 1, deck_count):
            similarity = 1.0 - histogram_intersection_distance(
                histograms[left_index], histograms[right_index]
            )
            similarities[left_index][right_index] = similarity
            similarities[right_index][left_index] = similarity
    return similarities


def _cluster_summary(
    cluster_id: int,
    member_indices: list[int],
    records: list[dict[str, Any]],
    deck_histograms: list[dict[int, int]],
    pokemon_histograms: list[dict[int, int]],
    type_signatures: list[str],
    card_metadata: dict[int, CardMetadata],
) -> dict[str, Any]:
    """1クラスタの勝敗、タイプ純度、主要ポケモンを集計する。"""

    games = sum(int(records[index].get("games", 0)) for index in member_indices)
    wins = sum(int(records[index].get("wins", 0)) for index in member_indices)
    losses = sum(int(records[index].get("losses", 0)) for index in member_indices)
    draws = sum(int(records[index].get("draws", 0)) for index in member_indices)
    signature_counts = Counter(type_signatures[index] for index in member_indices)
    dominant_type, dominant_type_members = signature_counts.most_common(1)[0]
    signature_games: Counter[str] = Counter()
    appearances: Counter[int] = Counter()
    total_copies: Counter[int] = Counter()
    card_category_counts: Counter[str] = Counter()
    pokemon_type_counts: Counter[str] = Counter()
    energy_type_counts: Counter[str] = Counter()
    for index in member_indices:
        signature_games[type_signatures[index]] += int(records[index].get("games", 0))
        for card_id, count in deck_histograms[index].items():
            metadata = card_metadata.get(card_id)
            category = _card_category(metadata)
            card_category_counts[category] += count
            if metadata is not None and metadata.is_pokemon:
                pokemon_type_counts[metadata.pokemon_type or "不明"] += count
            elif category == "エネルギー":
                energy_type = metadata.pokemon_type if metadata is not None else "不明"
                energy_type_counts[energy_type or "不明"] += count
        for card_id, count in pokemon_histograms[index].items():
            appearances[card_id] += 1
            total_copies[card_id] += count

    top_pokemon = []
    for card_id in sorted(
        appearances,
        key=lambda value: (-appearances[value], -total_copies[value], value),
    )[:8]:
        metadata = card_metadata[card_id]
        top_pokemon.append(
            {
                "card_id": card_id,
                "name": metadata.name,
                "type": metadata.pokemon_type,
                "adoption_rate": appearances[card_id] / len(member_indices),
                "average_copies": total_copies[card_id] / len(member_indices),
            }
        )

    return {
        "cluster_id": cluster_id,
        "members": len(member_indices),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "dominant_pokemon_type": dominant_type,
        "type_purity": dominant_type_members / len(member_indices),
        "game_weighted_type_purity_numerator": max(signature_games.values(), default=0),
        "card_category_distribution": _ratio_distribution(
            card_category_counts, "category"
        ),
        "pokemon_type_distribution": _ratio_distribution(
            pokemon_type_counts, "type"
        ),
        "energy_type_distribution": _ratio_distribution(
            energy_type_counts, "type"
        ),
        "top_pokemon": top_pokemon,
    }


def evaluate_records(
    records: list[dict[str, Any]],
    card_metadata: dict[int, CardMetadata],
    cluster_field: str = DEFAULT_CLUSTER_FIELD,
    boundary_similarity: float = DEFAULT_BOUNDARY_SIMILARITY,
) -> dict[str, Any]:
    """クラスタリング済みレコードを評価し、JSON化可能なレポートを返す。"""

    if not records:
        raise ValueError("評価対象のデッキがありません。")
    if not 0.0 <= boundary_similarity <= 1.0:
        raise ValueError("boundary_similarityは0以上1以下で指定してください。")

    try:
        cluster_ids = [int(record[cluster_field]) for record in records]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"全レコードに整数の{cluster_field}が必要です。") from exc

    histograms = [count_histogram(record) for record in records]
    pokemon_histograms = [
        _pokemon_histogram(histogram, card_metadata) for histogram in histograms
    ]
    type_signatures = [
        _dominant_pokemon_type_signature(histogram, card_metadata)
        for histogram in pokemon_histograms
    ]
    similarities = _build_similarity_matrix(histograms)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, cluster_id in enumerate(cluster_ids):
        groups[cluster_id].append(index)

    within_similarities: list[float] = []
    between_similarities: list[float] = []
    within_pokemon_jaccards: list[float] = []
    between_pokemon_jaccards: list[float] = []
    within_below_boundary = 0
    between_at_or_above_boundary = 0
    maximum_between_pair: dict[str, Any] | None = None
    for left_index in range(len(records)):
        for right_index in range(left_index + 1, len(records)):
            similarity = similarities[left_index][right_index]
            pokemon_jaccard = _multiset_jaccard(
                pokemon_histograms[left_index], pokemon_histograms[right_index]
            )
            same_cluster = cluster_ids[left_index] == cluster_ids[right_index]
            if same_cluster:
                within_similarities.append(similarity)
                within_pokemon_jaccards.append(pokemon_jaccard)
                if similarity < boundary_similarity:
                    within_below_boundary += 1
            else:
                between_similarities.append(similarity)
                between_pokemon_jaccards.append(pokemon_jaccard)
                if similarity >= boundary_similarity:
                    between_at_or_above_boundary += 1
                if (
                    maximum_between_pair is None
                    or similarity > maximum_between_pair["similarity"]
                ):
                    maximum_between_pair = {
                        "similarity": similarity,
                        "left_record_index": left_index,
                        "right_record_index": right_index,
                        "left_cluster_id": cluster_ids[left_index],
                        "right_cluster_id": cluster_ids[right_index],
                    }

    silhouettes: list[float] = []
    strict_nearest_same_cluster = 0
    non_singleton_decks = 0
    for deck_index, cluster_id in enumerate(cluster_ids):
        own_members = [index for index in groups[cluster_id] if index != deck_index]
        if not own_members:
            silhouettes.append(0.0)
            continue
        non_singleton_decks += 1
        mean_own_distance = fmean(
            1.0 - similarities[deck_index][index] for index in own_members
        )
        other_group_distances = [
            fmean(1.0 - similarities[deck_index][index] for index in member_indices)
            for other_cluster_id, member_indices in groups.items()
            if other_cluster_id != cluster_id
        ]
        nearest_other_distance = min(other_group_distances)
        denominator = max(mean_own_distance, nearest_other_distance)
        silhouette = (
            (nearest_other_distance - mean_own_distance) / denominator
            if denominator
            else 0.0
        )
        silhouettes.append(silhouette)
        maximum_same_similarity = max(
            similarities[deck_index][index] for index in own_members
        )
        maximum_other_similarity = max(
            similarities[deck_index][index]
            for other_cluster_id, member_indices in groups.items()
            if other_cluster_id != cluster_id
            for index in member_indices
        )
        if maximum_same_similarity > maximum_other_similarity:
            strict_nearest_same_cluster += 1

    cluster_summaries = [
        _cluster_summary(
            cluster_id,
            member_indices,
            records,
            histograms,
            pokemon_histograms,
            type_signatures,
            card_metadata,
        )
        for cluster_id, member_indices in sorted(groups.items())
    ]
    total_games = sum(summary["games"] for summary in cluster_summaries)
    unique_deck_type_purity = sum(
        summary["type_purity"] * summary["members"]
        for summary in cluster_summaries
    ) / len(records)
    game_weighted_type_purity = (
        sum(
            summary["game_weighted_type_purity_numerator"]
            for summary in cluster_summaries
        )
        / total_games
        if total_games
        else 0.0
    )
    for summary in cluster_summaries:
        del summary["game_weighted_type_purity_numerator"]

    cluster_sizes = [len(member_indices) for member_indices in groups.values()]
    return {
        "method": {
            "cluster_field": cluster_field,
            "cluster_count": len(groups),
            "distance_metric": "1-histogram-intersection",
            "linkage": records[0].get("hierarchical_linkage"),
            "boundary_similarity": boundary_similarity,
        },
        "dataset": {
            "unique_decks": len(records),
            "total_games": total_games,
        },
        "cluster_sizes": {
            "minimum": min(cluster_sizes),
            "median": _quantile([float(size) for size in cluster_sizes], 0.5),
            "maximum": max(cluster_sizes),
            "singletons": sum(size == 1 for size in cluster_sizes),
        },
        "deck_similarity": {
            "within_cluster": _distribution(within_similarities),
            "between_clusters": _distribution(between_similarities),
            "within_pairs_below_boundary": within_below_boundary,
            "between_pairs_at_or_above_boundary": between_at_or_above_boundary,
            "maximum_between_pair": maximum_between_pair,
        },
        "silhouette": {
            "mean": fmean(silhouettes),
            "median": _quantile(silhouettes, 0.5),
            "negative_count": sum(value < 0.0 for value in silhouettes),
            "negative_rate": sum(value < 0.0 for value in silhouettes) / len(records),
        },
        "nearest_neighbor": {
            "non_singleton_decks": non_singleton_decks,
            "strict_same_cluster_count": strict_nearest_same_cluster,
            "strict_same_cluster_rate": (
                strict_nearest_same_cluster / non_singleton_decks
                if non_singleton_decks
                else 0.0
            ),
        },
        "pokemon_composition": {
            "within_cluster_jaccard": _distribution(within_pokemon_jaccards),
            "between_clusters_jaccard": _distribution(between_pokemon_jaccards),
            "dominant_type_purity": unique_deck_type_purity,
            "game_weighted_dominant_type_purity": game_weighted_type_purity,
        },
        "clusters": cluster_summaries,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """評価レポートを確認しやすい日本語Markdownへ変換する。"""

    method = report["method"]
    dataset = report["dataset"]
    sizes = report["cluster_sizes"]
    deck_similarity = report["deck_similarity"]
    within = deck_similarity["within_cluster"]
    between = deck_similarity["between_clusters"]
    silhouette = report["silhouette"]
    nearest = report["nearest_neighbor"]
    pokemon = report["pokemon_composition"]
    lines = [
        "# 階層型デッキクラスタリング評価",
        "",
        "## 評価条件",
        "",
        f"- クラスタ数: {method['cluster_count']}",
        f"- 連結法: {method['linkage']}",
        f"- 距離: {method['distance_metric']}",
        f"- ユニークデッキ数: {dataset['unique_decks']}",
        f"- 対戦記録数: {dataset['total_games']}",
        f"- 境界確認用類似度: {method['boundary_similarity']:.2f}",
        "",
        "## 定量評価",
        "",
        "|指標|結果|",
        "|---|---:|",
        f"|クラスタ内平均類似度|{within['mean']:.4f}|",
        f"|クラスタ間平均類似度|{between['mean']:.4f}|",
        f"|クラスタ内類似度中央値|{within['median']:.4f}|",
        f"|クラスタ間類似度中央値|{between['median']:.4f}|",
        f"|平均シルエット係数|{silhouette['mean']:.4f}|",
        f"|負のシルエット率|{silhouette['negative_rate']:.2%}|",
        f"|最近傍が同一クラスタの割合|{nearest['strict_same_cluster_rate']:.2%}|",
        f"|別クラスタ高類似ペア数|{deck_similarity['between_pairs_at_or_above_boundary']}|",
        f"|クラスタ内低類似ペア数|{deck_similarity['within_pairs_below_boundary']}|",
        "",
        "## 定性評価の補助指標",
        "",
        "|指標|結果|",
        "|---|---:|",
        f"|クラスタ内ポケモン構成Jaccard平均|{pokemon['within_cluster_jaccard']['mean']:.4f}|",
        f"|クラスタ間ポケモン構成Jaccard平均|{pokemon['between_clusters_jaccard']['mean']:.4f}|",
        f"|主要ポケモンタイプ純度|{pokemon['dominant_type_purity']:.2%}|",
        f"|対戦数加重タイプ純度|{pokemon['game_weighted_dominant_type_purity']:.2%}|",
        "",
        "## クラスタサイズ",
        "",
        f"- 最小: {sizes['minimum']}",
        f"- 中央値: {sizes['median']:.0f}",
        f"- 最大: {sizes['maximum']}",
        f"- 単独クラスタ: {sizes['singletons']}",
        "",
        "## クラスタ別概要",
        "",
        "|ID|デッキ数|対戦数|勝率|主要タイプ|タイプ純度|ポケモン/グッズ/エネルギー|主要ポケモン|",
        "|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for cluster in report["clusters"]:
        pokemon_names = "、".join(
            item["name"] for item in cluster["top_pokemon"][:5]
        )
        category_ratios = {
            item["category"]: item["ratio"]
            for item in cluster["card_category_distribution"]
        }
        overview_ratios = (
            f"{category_ratios.get('ポケモン', 0.0):.1%} / "
            f"{category_ratios.get('グッズ', 0.0):.1%} / "
            f"{category_ratios.get('エネルギー', 0.0):.1%}"
        )
        lines.append(
            f"|{cluster['cluster_id']}|{cluster['members']}|{cluster['games']}|"
            f"{cluster['win_rate']:.2%}|{cluster['dominant_pokemon_type']}|"
            f"{cluster['type_purity']:.2%}|{overview_ratios}|{pokemon_names}|"
        )
    lines.extend(["", "## クラスタ別カード構成", ""])
    for cluster in report["clusters"]:
        category_text = "、".join(
            f"{item['category']} {item['ratio']:.1%}"
            for item in cluster["card_category_distribution"]
        )
        pokemon_type_text = "、".join(
            f"{item['type']} {item['ratio']:.1%}"
            for item in cluster["pokemon_type_distribution"]
        ) or "ポケモンなし"
        energy_type_text = "、".join(
            f"{item['type']} {item['ratio']:.1%}"
            for item in cluster["energy_type_distribution"]
        ) or "エネルギーなし"
        lines.extend(
            [
                f"### クラスタ {cluster['cluster_id']}",
                "",
                f"- カード種別構成: {category_text}",
                f"- ポケモンタイプ構成: {pokemon_type_text}",
                f"- エネルギータイプ構成: {energy_type_text}",
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_text_atomic(path: Path, text: str) -> None:
    """評価結果を一時ファイル経由で原子的に書き込む。"""

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
            temporary_file.write(text)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> int:
    """指定JSONLを評価し、JSONと日本語Markdownを出力する。"""

    args = parse_args()
    cluster_count = len(
        {
            int(record[args.cluster_field])
            for record in load_records(args.input)
        }
    )
    output_json = args.output_json or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_cluster_evaluation_{cluster_count}.json"
    )
    output_markdown = args.output_markdown or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_cluster_evaluation_{cluster_count}.md"
    )
    records = load_records(args.input)
    card_metadata = load_card_metadata(args.card_data)
    report = evaluate_records(
        records,
        card_metadata,
        cluster_field=args.cluster_field,
        boundary_similarity=args.boundary_similarity,
    )
    write_text_atomic(
        output_json,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(output_markdown, render_markdown(report))
    print(f"evaluated {len(records)} decks in {cluster_count} clusters")
    print(f"wrote JSON evaluation to {output_json}")
    print(f"wrote Markdown evaluation to {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
