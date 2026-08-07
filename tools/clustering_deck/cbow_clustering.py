"""CBOWカード埋め込みに基づいてデッキ類似度を計算し、デッキを分類する。

各デッキを次の2成分からなる単位ベクトルへ変換する。

* CBOW意味成分: IDF付きカード埋め込みの加重平均
* 構成成分: カード採用枚数のTF-IDFベクトル

両成分をそれぞれL2正規化してから重み付きで連結するため、デッキ間の内積は
``semantic_weight * semantic_cosine + (1-semantic_weight) * composition_cosine``
に一致する。この類似度に対してspherical k-meansを適用する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = ROOT / "tools" / "deck_generator_dbrecon" / "deck_candidates_by_wins.jsonl"
DEFAULT_CHECKPOINT = ROOT / "tools" / "deck_generator" / "generated" / "deck_word2vec.pt"
DEFAULT_CARD_DATA = ROOT / "data" / "EN_Card_Data.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "deck_candidates_cbow_clustered.jsonl"
DEFAULT_REPORT = SCRIPT_DIR / "generated" / "cbow_cluster_summary.json"
DEFAULT_MARKDOWN_REPORT = SCRIPT_DIR / "generated" / "cbow_cluster_summary.md"
DECK_SIZE = 60
EPSILON = 1e-12


@dataclass(frozen=True)
class DeckFeatureModel:
    """デッキ特徴量と、その再計算に必要な統計量。"""

    features: torch.Tensor
    semantic_features: torch.Tensor
    composition_features: torch.Tensor
    card_ids: list[int]
    idf: torch.Tensor
    semantic_weight: float


@dataclass(frozen=True)
class KMeansResult:
    """spherical k-meansの最良試行結果。"""

    labels: torch.Tensor
    centroids: torch.Tensor
    inertia: float
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力JSONL")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="CBOWチェックポイント")
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA, help="カード名を含むCSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="クラスタID付きJSONL")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="クラスタ概要JSON")
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
        help="人が読むためのクラスタ概要Markdown",
    )
    parser.add_argument(
        "--clusters",
        default="auto",
        help="クラスタ数。整数またはauto（デフォルト: auto）",
    )
    parser.add_argument("--min-clusters", type=int, default=4, help="auto探索の最小クラスタ数")
    parser.add_argument("--max-clusters", type=int, default=24, help="auto探索の最大クラスタ数")
    parser.add_argument("--cluster-step", type=int, default=2, help="auto探索のクラスタ数刻み")
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.7,
        help="CBOW意味成分の重み。構成成分の重みは1からこの値を引いた値",
    )
    parser.add_argument("--n-init", type=int, default=8, help="k-meansの初期化試行数")
    parser.add_argument("--max-iterations", type=int, default=100, help="各k-means試行の最大反復数")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
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
    if not records:
        raise ValueError(f"{path} にデッキがありません。")
    return records


def deck_histogram(record: dict[str, Any]) -> dict[int, int]:
    """deckまたはdeck_countsから60枚のカード枚数辞書を返す。"""

    try:
        if "deck" in record:
            histogram = dict(Counter(int(card_id) for card_id in record["deck"]))
        else:
            histogram = {
                int(card_id): int(count)
                for card_id, count in record["deck_counts"].items()
            }
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("deckまたはdeck_countsを読み込めません。") from exc
    if any(card_id < 0 or count <= 0 for card_id, count in histogram.items()):
        raise ValueError("カードIDは0以上、採用枚数は正の整数である必要があります。")
    if sum(histogram.values()) != DECK_SIZE:
        raise ValueError(f"対象デッキは{DECK_SIZE}枚である必要があります。")
    return histogram


def load_card_embeddings(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    """train_deck_word2vec.py形式のCBOW埋め込みをロードする。"""

    checkpoint = torch.load(path, map_location="cpu")
    try:
        embeddings = checkpoint["model_state"]["card_embedding"].detach().float().cpu()
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"{path} にCBOW card_embeddingがありません。") from exc
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("card_embeddingは空でない2次元テンソルである必要があります。")
    return embeddings, checkpoint


def _l2_normalize(matrix: torch.Tensor) -> torch.Tensor:
    return F.normalize(matrix, p=2, dim=1, eps=EPSILON)


def build_deck_features(
    histograms: list[dict[int, int]],
    card_embeddings: torch.Tensor,
    semantic_weight: float = 0.7,
) -> DeckFeatureModel:
    """CBOW意味成分とTF-IDF構成成分からデッキ特徴量を作る。"""

    if not histograms:
        raise ValueError("デッキがありません。")
    if not 0.0 <= semantic_weight <= 1.0:
        raise ValueError("semantic_weightは0以上1以下で指定してください。")

    card_ids = sorted({card_id for histogram in histograms for card_id in histogram})
    if not card_ids:
        raise ValueError("カードがありません。")
    maximum_card_id = max(card_ids)
    if maximum_card_id >= card_embeddings.shape[0]:
        raise ValueError(
            f"カードID {maximum_card_id} は埋め込み語彙サイズ "
            f"{card_embeddings.shape[0]} の範囲外です。"
        )

    deck_count = len(histograms)
    card_to_column = {card_id: index for index, card_id in enumerate(card_ids)}
    counts = torch.zeros((deck_count, len(card_ids)), dtype=torch.float32)
    document_frequency = torch.zeros(len(card_ids), dtype=torch.float32)
    for deck_index, histogram in enumerate(histograms):
        for card_id, count in histogram.items():
            column = card_to_column[card_id]
            counts[deck_index, column] = float(count)
            document_frequency[column] += 1.0

    idf = torch.log((1.0 + deck_count) / (1.0 + document_frequency)) + 1.0

    # CBOW空間の共通方向を除去し、カード間の相対的な役割を強調する。
    used_embeddings = _l2_normalize(card_embeddings[card_ids])
    used_embeddings = used_embeddings - used_embeddings.mean(dim=0, keepdim=True)
    used_embeddings = _l2_normalize(used_embeddings)
    semantic_weights = counts * idf.unsqueeze(0)
    semantic = semantic_weights @ used_embeddings
    semantic = _l2_normalize(semantic)

    # 採用枚数はsublinear TFにして、4枚採用などの差を残しつつ過度な支配を抑える。
    sublinear_tf = torch.where(counts > 0, 1.0 + torch.log(counts.clamp_min(1.0)), counts)
    composition = _l2_normalize(sublinear_tf * idf.unsqueeze(0))

    components: list[torch.Tensor] = []
    if semantic_weight > 0:
        components.append(math.sqrt(semantic_weight) * semantic)
    if semantic_weight < 1:
        components.append(math.sqrt(1.0 - semantic_weight) * composition)
    features = _l2_normalize(torch.cat(components, dim=1))
    return DeckFeatureModel(
        features=features,
        semantic_features=semantic,
        composition_features=composition,
        card_ids=card_ids,
        idf=idf,
        semantic_weight=semantic_weight,
    )


def deck_similarity(feature_model: DeckFeatureModel, left_index: int, right_index: int) -> float:
    """2デッキの複合cosine類似度を返す。範囲は原則[-1, 1]。"""

    value = torch.dot(
        feature_model.features[left_index],
        feature_model.features[right_index],
    )
    return float(value.clamp(-1.0, 1.0).item())


def _kmeans_plus_plus(features: torch.Tensor, clusters: int, seed: int) -> torch.Tensor:
    """cosine距離によるk-means++初期中心を選ぶ。"""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    first = int(torch.randint(features.shape[0], (1,), generator=generator).item())
    chosen = [first]
    closest_distance = (1.0 - features @ features[first]).clamp_min(0.0)
    for _ in range(1, clusters):
        probabilities = closest_distance.square()
        probabilities[chosen] = 0.0
        total = probabilities.sum()
        if float(total.item()) <= EPSILON:
            remaining = [index for index in range(features.shape[0]) if index not in chosen]
            chosen.append(remaining[0])
        else:
            next_index = int(torch.multinomial(probabilities / total, 1, generator=generator).item())
            chosen.append(next_index)
        distance = (1.0 - features @ features[chosen[-1]]).clamp_min(0.0)
        closest_distance = torch.minimum(closest_distance, distance)
    return features[chosen].clone()


def spherical_kmeans(
    features: torch.Tensor,
    clusters: int,
    *,
    n_init: int = 8,
    max_iterations: int = 100,
    seed: int = 0,
) -> KMeansResult:
    """単位ベクトルをcosine距離でクラスタリングする。"""

    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("featuresは空でない2次元テンソルである必要があります。")
    if not 1 <= clusters <= features.shape[0]:
        raise ValueError("clustersは1以上デッキ数以下で指定してください。")
    if n_init <= 0 or max_iterations <= 0:
        raise ValueError("n_initとmax_iterationsは正の整数で指定してください。")

    best: KMeansResult | None = None
    for initialization in range(n_init):
        centroids = _kmeans_plus_plus(features, clusters, seed + initialization * 104729)
        previous_labels: torch.Tensor | None = None
        for iteration in range(1, max_iterations + 1):
            similarities = features @ centroids.T
            labels = similarities.argmax(dim=1)

            # 空クラスタには、現在の中心から最も遠いデッキを割り当てる。
            occupied = torch.bincount(labels, minlength=clusters)
            if torch.any(occupied == 0):
                nearest_similarity = similarities.gather(1, labels.unsqueeze(1)).squeeze(1)
                candidates = torch.argsort(nearest_similarity)
                used: set[int] = set()
                for empty_cluster in torch.where(occupied == 0)[0].tolist():
                    replacement = next(
                        int(index) for index in candidates.tolist() if int(index) not in used
                    )
                    labels[replacement] = int(empty_cluster)
                    used.add(replacement)

            if previous_labels is not None and torch.equal(labels, previous_labels):
                break
            previous_labels = labels.clone()
            new_centroids = torch.zeros_like(centroids)
            new_centroids.index_add_(0, labels, features)
            centroids = _l2_normalize(new_centroids)

        final_similarities = features @ centroids.T
        labels = final_similarities.argmax(dim=1)
        assigned = final_similarities.gather(1, labels.unsqueeze(1)).squeeze(1)
        inertia = float((1.0 - assigned).sum().item())
        result = KMeansResult(labels, centroids, inertia, iteration)
        if best is None or result.inertia < best.inertia - 1e-9:
            best = result
    assert best is not None
    return best


def silhouette_score(features: torch.Tensor, labels: torch.Tensor) -> float:
    """cosine距離の平均silhouette係数を返す。単独クラスタの標本は0とする。"""

    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    distances = (1.0 - features @ features.T).clamp_min(0.0)
    silhouettes = torch.zeros(features.shape[0], dtype=torch.float32)
    for index in range(features.shape[0]):
        own_label = labels[index]
        own_members = torch.where(labels == own_label)[0]
        own_members = own_members[own_members != index]
        if own_members.numel() == 0:
            continue
        within = distances[index, own_members].mean()
        between = min(
            distances[index, torch.where(labels == other_label)[0]].mean()
            for other_label in unique_labels
            if int(other_label.item()) != int(own_label.item())
        )
        denominator = torch.maximum(within, between)
        if float(denominator.item()) > EPSILON:
            silhouettes[index] = (between - within) / denominator
    return float(silhouettes.mean().item())


def choose_cluster_count(
    features: torch.Tensor,
    candidates: Iterable[int],
    *,
    n_init: int,
    max_iterations: int,
    seed: int,
) -> tuple[KMeansResult, int, list[dict[str, float | int]]]:
    """候補ごとのsilhouetteを評価し、最大のクラスタ数を選ぶ。"""

    evaluations: list[dict[str, float | int]] = []
    best_result: KMeansResult | None = None
    best_clusters = 0
    best_silhouette = -float("inf")
    valid_candidates = sorted(set(int(value) for value in candidates))
    if not valid_candidates:
        raise ValueError("クラスタ数候補がありません。")
    for clusters in valid_candidates:
        result = spherical_kmeans(
            features,
            clusters,
            n_init=n_init,
            max_iterations=max_iterations,
            seed=seed,
        )
        score = silhouette_score(features, result.labels)
        evaluations.append(
            {
                "clusters": clusters,
                "silhouette": score,
                "inertia": result.inertia,
                "iterations": result.iterations,
            }
        )
        if score > best_silhouette + 1e-9 or (
            abs(score - best_silhouette) <= 1e-9 and clusters < best_clusters
        ):
            best_result = result
            best_clusters = clusters
            best_silhouette = score
    assert best_result is not None
    return best_result, best_clusters, evaluations


def load_card_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        for row in csv.DictReader(input_file):
            names[int(row["Card ID"])] = row["Card Name"]
    return names


def _stable_deck_key(record: dict[str, Any]) -> str:
    histogram = deck_histogram(record)
    return json.dumps(sorted(histogram.items()), separators=(",", ":"))


def add_cluster_fields(
    records: list[dict[str, Any]],
    histograms: list[dict[int, int]],
    feature_model: DeckFeatureModel,
    result: KMeansResult,
) -> list[dict[str, Any]]:
    """クラスタを対戦数順に安定採番し、cbow_cluster_*フィールドを付与する。"""

    raw_clusters: list[dict[str, Any]] = []
    for raw_id in sorted(set(result.labels.tolist())):
        members = torch.where(result.labels == raw_id)[0].tolist()
        games = sum(int(records[index].get("games", 0)) for index in members)
        wins = sum(int(records[index].get("wins", 0)) for index in members)
        losses = sum(int(records[index].get("losses", 0)) for index in members)
        draws = sum(int(records[index].get("draws", 0)) for index in members)
        similarities = feature_model.features[members] @ result.centroids[raw_id]
        representative = max(
            zip(members, similarities.tolist()),
            key=lambda item: (
                item[1],
                int(records[item[0]].get("games", 0)),
                _stable_deck_key(records[item[0]]),
            ),
        )[0]
        raw_clusters.append(
            {
                "raw_id": raw_id,
                "members": members,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "representative": representative,
            }
        )
    raw_clusters.sort(
        key=lambda cluster: (
            -int(cluster["games"]),
            -len(cluster["members"]),
            _stable_deck_key(records[int(cluster["representative"])]),
        )
    )

    cluster_count = len(raw_clusters)
    total_decks = len(records)
    for stable_id, cluster in enumerate(raw_clusters):
        members = list(cluster["members"])
        raw_id = int(cluster["raw_id"])
        for index in members:
            semantic_similarity = float(
                torch.dot(
                    feature_model.semantic_features[index],
                    feature_model.semantic_features[int(cluster["representative"])],
                ).item()
            )
            composition_similarity = float(
                torch.dot(
                    feature_model.composition_features[index],
                    feature_model.composition_features[int(cluster["representative"])],
                ).item()
            )
            record = records[index]
            record["cbow_cluster_id"] = stable_id
            record["cbow_cluster_members"] = len(members)
            record["cbow_cluster_games"] = int(cluster["games"])
            record["cbow_cluster_wins"] = int(cluster["wins"])
            record["cbow_cluster_losses"] = int(cluster["losses"])
            record["cbow_cluster_draws"] = int(cluster["draws"])
            record["cbow_cluster_win_rate"] = (
                int(cluster["wins"]) / int(cluster["games"]) if int(cluster["games"]) else 0.0
            )
            record["cbow_cluster_similarity"] = float(
                torch.dot(feature_model.features[index], result.centroids[raw_id]).item()
            )
            record["cbow_representative_similarity"] = (
                feature_model.semantic_weight * semantic_similarity
                + (1.0 - feature_model.semantic_weight) * composition_similarity
            )
            record["cbow_cluster_representative"] = index == int(cluster["representative"])
            record["cbow_cluster_weight"] = total_decks / (len(members) * cluster_count)
            record["cbow_semantic_weight"] = feature_model.semantic_weight
            record["cbow_similarity_metric"] = "idf-cbow-cosine+tfidf-count-cosine"
    return raw_clusters


def build_report(
    records: list[dict[str, Any]],
    histograms: list[dict[int, int]],
    clusters: list[dict[str, Any]],
    evaluations: list[dict[str, float | int]],
    feature_model: DeckFeatureModel,
    checkpoint: dict[str, Any],
    input_path: Path,
    checkpoint_path: Path,
    card_names: dict[int, str],
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for stable_id, cluster in enumerate(clusters):
        members = list(cluster["members"])
        aggregate = Counter()
        for index in members:
            aggregate.update(histograms[index])
        card_summaries = [
            {
                "card_id": card_id,
                "name": card_names.get(card_id, str(card_id)),
                "decks": sum(card_id in histograms[index] for index in members),
                "copies": copies,
            }
            for card_id, copies in aggregate.items()
        ]
        card_summaries.sort(
            key=lambda card: (
                -int(card["decks"]),
                -int(card["copies"]),
                int(card["card_id"]),
            )
        )
        top_cards = card_summaries[:12]
        representative = int(cluster["representative"])
        representative_cards = [
            {
                "card_id": card_id,
                "name": card_names.get(card_id, str(card_id)),
                "count": count,
            }
            for card_id, count in sorted(
                histograms[representative].items(),
                key=lambda item: (-item[1], card_names.get(item[0], str(item[0])), item[0]),
            )
        ]
        summaries.append(
            {
                "cluster_id": stable_id,
                "members": len(members),
                "games": int(cluster["games"]),
                "wins": int(cluster["wins"]),
                "losses": int(cluster["losses"]),
                "draws": int(cluster["draws"]),
                "win_rate": (
                    int(cluster["wins"]) / int(cluster["games"])
                    if int(cluster["games"])
                    else 0.0
                ),
                "mean_centroid_similarity": sum(
                    float(records[index]["cbow_cluster_similarity"]) for index in members
                ) / len(members),
                "representative_record_index": representative,
                "representative_deck_counts": {
                    str(card_id): count
                    for card_id, count in sorted(histograms[representative].items())
                },
                "representative_cards": representative_cards,
                "top_cards": top_cards,
            }
        )
    selected_evaluation = max(
        evaluations,
        key=lambda item: (
            float(item["silhouette"]),
            -int(item["clusters"]),
        ),
    )
    return {
        "method": {
            "name": "idf-cbow-spherical-kmeans",
            "similarity": "semantic_weight*CBOW_cosine+(1-semantic_weight)*TFIDF_count_cosine",
            "semantic_weight": feature_model.semantic_weight,
            "composition_weight": 1.0 - feature_model.semantic_weight,
            "cluster_selection": "maximum cosine silhouette",
            "selected_clusters": int(selected_evaluation["clusters"]),
            "selected_silhouette": float(selected_evaluation["silhouette"]),
        },
        "source": {
            "input": str(input_path),
            "checkpoint": str(checkpoint_path),
            "decks": len(records),
            "cards": len(feature_model.card_ids),
            "embedding_dimension": int(feature_model.semantic_features.shape[1]),
            "checkpoint_metrics": checkpoint.get("metrics", {}),
        },
        "cluster_count_evaluations": evaluations,
        "clusters": summaries,
    }


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown_report(report: dict[str, Any]) -> str:
    """クラスタ集計JSONと同じ内容から、人が読みやすいMarkdownを作る。"""

    method = report["method"]
    source = report["source"]
    clusters = report["clusters"]
    lines = [
        "# CBOWデッキクラスタ概要",
        "",
        "## 全体",
        "",
        f"- 対象デッキ数: {int(source['decks']):,}",
        f"- 使用カード種類数: {int(source['cards']):,}",
        f"- クラスタ数: {int(method['selected_clusters'])}",
        f"- cosine silhouette: {float(method['selected_silhouette']):.6f}",
        f"- CBOW意味成分の重み: {float(method['semantic_weight']):.0%}",
        f"- カード枚数構成成分の重み: {float(method['composition_weight']):.0%}",
        "",
        "クラスタIDはクラスタ内の総試合数が多い順です。勝率はクラスタに属するデッキの"
        "勝利数を総試合数で割った値です。頻出カードは採用デッキ率、総採用枚数の順で選んでいます。",
        "",
        "## クラスタ一覧",
        "",
        "|ID|デッキ数|試合数|勝-敗-分|勝率|中心類似度|主な構成カード|",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cluster in clusters:
        top_names = "、".join(
            _escape_markdown(card["name"])
            for card in cluster["top_cards"][:5]
        )
        lines.append(
            f"|{int(cluster['cluster_id'])}|{int(cluster['members']):,}|"
            f"{int(cluster['games']):,}|{int(cluster['wins']):,}-"
            f"{int(cluster['losses']):,}-{int(cluster['draws']):,}|"
            f"{float(cluster['win_rate']):.2%}|"
            f"{float(cluster['mean_centroid_similarity']):.4f}|{top_names}|"
        )

    lines.extend(["", "## クラスタ別詳細", ""])
    for cluster in clusters:
        members = int(cluster["members"])
        lines.extend(
            [
                f"### クラスタ {int(cluster['cluster_id'])}",
                "",
                f"- デッキ数: {members:,}",
                f"- 戦績: {int(cluster['wins']):,}勝 / {int(cluster['losses']):,}敗 / "
                f"{int(cluster['draws']):,}分（{int(cluster['games']):,}試合）",
                f"- 勝率: {float(cluster['win_rate']):.2%}",
                f"- 平均中心類似度: {float(cluster['mean_centroid_similarity']):.4f}",
                "",
                "主な構成カード:",
                "",
                "|カードID|カード名|採用デッキ数|採用率|総枚数|採用時平均枚数|",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for card in cluster["top_cards"]:
            decks = int(card["decks"])
            copies = int(card["copies"])
            lines.append(
                f"|{int(card['card_id'])}|{_escape_markdown(card['name'])}|{decks:,}|"
                f"{decks / members:.1%}|{copies:,}|{copies / decks:.2f}|"
            )
        lines.extend(
            [
                "",
                f"代表デッキ（元レコード番号: {int(cluster['representative_record_index'])}）:",
                "",
                "|カードID|カード名|枚数|",
                "|---:|---|---:|",
            ]
        )
        for card in cluster["representative_cards"]:
            lines.append(
                f"|{int(card['card_id'])}|{_escape_markdown(card['name'])}|{int(card['count'])}|"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_json_atomic(path: Path, value: Any, *, json_lines: bool = False) -> None:
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            if json_lines:
                for item in value:
                    temporary.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                    temporary.write("\n")
            else:
                json.dump(value, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_text_atomic(path: Path, text: str) -> None:
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_cluster_candidates(args: argparse.Namespace, deck_count: int) -> list[int]:
    if args.clusters != "auto":
        try:
            clusters = int(args.clusters)
        except ValueError as exc:
            raise ValueError("--clustersは正の整数またはautoで指定してください。") from exc
        if not 1 <= clusters <= deck_count:
            raise ValueError(f"--clustersは1以上{deck_count}以下で指定してください。")
        return [clusters]
    if args.cluster_step <= 0:
        raise ValueError("--cluster-stepは正の整数で指定してください。")
    minimum = max(2, args.min_clusters)
    maximum = min(args.max_clusters, deck_count - 1)
    if minimum > maximum:
        raise ValueError("auto探索のクラスタ数範囲が空です。")
    return list(range(minimum, maximum + 1, args.cluster_step))


def main() -> int:
    args = parse_args()
    records = load_records(args.input)
    histograms = [deck_histogram(record) for record in records]
    embeddings, checkpoint = load_card_embeddings(args.checkpoint)
    feature_model = build_deck_features(histograms, embeddings, args.semantic_weight)
    candidates = resolve_cluster_candidates(args, len(records))
    result, selected_clusters, evaluations = choose_cluster_count(
        feature_model.features,
        candidates,
        n_init=args.n_init,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )
    clusters = add_cluster_fields(records, histograms, feature_model, result)
    report = build_report(
        records,
        histograms,
        clusters,
        evaluations,
        feature_model,
        checkpoint,
        args.input,
        args.checkpoint,
        load_card_names(args.card_data),
    )
    write_json_atomic(args.output, records, json_lines=True)
    write_json_atomic(args.report, report)
    write_text_atomic(args.markdown_report, render_markdown_report(report))
    chosen = next(item for item in evaluations if int(item["clusters"]) == selected_clusters)
    print(
        f"clustered {len(records)} decks into {selected_clusters} clusters "
        f"(silhouette={float(chosen['silhouette']):.6f})"
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(f"wrote {args.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
