"""階層クラスタごとに代表デッキと競技向けデッキを選出する。

クラスタ内で観測された実在の60枚デッキだけを候補とする。代表デッキは、各構築の
使用試合数で重み付けした平均類似度（中心性）を基準に選ぶ。競技向けデッキは中心性を
大きく損なわない候補から、クラスタ平均へ縮約した補正勝率を基準に選ぶ。

実行例:
    python tools/clustering_deck/select_cluster_major_decks.py
    python tools/clustering_deck/select_cluster_major_decks.py \
        --input tools/deck_generator/generated/deck_candidates_by_wins.jsonl \
        --output-dir tools/clustering_deck/generated/cluster_major_decks_32

出力:
    cluster_00_representative.csv: 代表デッキのカードIDを1行1枚で記載した60行CSV
    cluster_00_competitive.csv: 競技向けデッキの同形式CSV
    manifest.csv / selection_summary.json / README.md: 選出根拠とクラスタ統計
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluate_hierarchical_clustering import (
    DEFAULT_CARD_DATA,
    CardMetadata,
    load_card_metadata,
    write_text_atomic,
)
from hierarchical_clustering import (
    DEFAULT_INPUT,
    count_histogram,
    histogram_intersection_distance,
    load_records,
    write_records_atomic,
)


DEFAULT_CLUSTER_FIELD = "hierarchical_cluster_id"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_REPRESENTATIVE_TOLERANCE = 0.01
DEFAULT_COMPETITIVE_TOLERANCE = 0.03
DEFAULT_WIN_RATE_PRIOR_GAMES = 20.0
DEFAULT_MINIMUM_COMPETITIVE_GAMES = 20
DEFAULT_MINIMUM_MAJOR_GAME_SHARE = 0.005


@dataclass(frozen=True)
class DeckCandidateScore:
    """クラスタ内の実在デッキ1件に対する選出指標を保持する。"""

    record_index: int
    centrality: float
    adjusted_win_rate: float
    games: int
    wins: int
    raw_win_rate: float


@dataclass(frozen=True)
class ClusterDeckSelection:
    """1クラスタの代表・競技向けデッキと選出統計を保持する。"""

    cluster_id: int
    member_count: int
    games: int
    wins: int
    cluster_win_rate: float
    game_share: float
    is_major_cluster: bool
    representative: DeckCandidateScore
    competitive: DeckCandidateScore
    competitive_minimum_games_applied: bool


def parse_args() -> argparse.Namespace:
    """主要デッキ選出コマンドの引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力JSONL")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="選出フィールド追加後のJSONL。省略時は入力を安全に置換する。",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    parser.add_argument("--cluster-field", default=DEFAULT_CLUSTER_FIELD)
    parser.add_argument(
        "--representative-tolerance",
        type=float,
        default=DEFAULT_REPRESENTATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--competitive-tolerance",
        type=float,
        default=DEFAULT_COMPETITIVE_TOLERANCE,
    )
    parser.add_argument(
        "--win-rate-prior-games",
        type=float,
        default=DEFAULT_WIN_RATE_PRIOR_GAMES,
    )
    parser.add_argument(
        "--minimum-competitive-games",
        type=int,
        default=DEFAULT_MINIMUM_COMPETITIVE_GAMES,
    )
    parser.add_argument(
        "--minimum-major-game-share",
        type=float,
        default=DEFAULT_MINIMUM_MAJOR_GAME_SHARE,
    )
    return parser.parse_args()


def _stable_deck_key(record: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """同点時の選出結果を入力順に依存させないカード構成キーを返す。"""

    return tuple(sorted(count_histogram(record).items()))


def _validate_parameters(
    representative_tolerance: float,
    competitive_tolerance: float,
    win_rate_prior_games: float,
    minimum_competitive_games: int,
    minimum_major_game_share: float,
) -> None:
    """選出閾値が意味のある範囲にあることを検証する。"""

    if not 0.0 <= representative_tolerance <= 1.0:
        raise ValueError("representative_toleranceは0以上1以下で指定してください。")
    if not 0.0 <= competitive_tolerance <= 1.0:
        raise ValueError("competitive_toleranceは0以上1以下で指定してください。")
    if win_rate_prior_games < 0.0:
        raise ValueError("win_rate_prior_gamesは0以上で指定してください。")
    if minimum_competitive_games < 0:
        raise ValueError("minimum_competitive_gamesは0以上で指定してください。")
    if not 0.0 <= minimum_major_game_share <= 1.0:
        raise ValueError("minimum_major_game_shareは0以上1以下で指定してください。")


def _adjusted_win_rate(
    wins: int,
    games: int,
    cluster_win_rate: float,
    prior_games: float,
) -> float:
    """少数試合の勝率をクラスタ平均へ縮約した補正勝率を返す。"""

    denominator = games + prior_games
    if denominator == 0:
        return cluster_win_rate
    return (wins + prior_games * cluster_win_rate) / denominator


def _score_cluster_candidates(
    records: list[dict[str, Any]],
    member_indices: list[int],
    cluster_win_rate: float,
    win_rate_prior_games: float,
) -> list[DeckCandidateScore]:
    """対戦数加重の平均類似度と補正勝率をクラスタ内候補ごとに計算する。"""

    histograms = {index: count_histogram(records[index]) for index in member_indices}
    weights = {index: max(0, int(records[index].get("games", 0))) for index in member_indices}
    total_weight = sum(weights.values())
    if total_weight == 0:
        weights = {index: 1 for index in member_indices}
        total_weight = len(member_indices)

    pair_similarities: dict[tuple[int, int], float] = {}
    scores: list[DeckCandidateScore] = []
    for left_index in member_indices:
        weighted_similarity = 0.0
        for right_index in member_indices:
            pair = tuple(sorted((left_index, right_index)))
            if pair not in pair_similarities:
                pair_similarities[pair] = 1.0 - histogram_intersection_distance(
                    histograms[left_index],
                    histograms[right_index],
                )
            weighted_similarity += weights[right_index] * pair_similarities[pair]
        games = max(0, int(records[left_index].get("games", 0)))
        wins = max(0, int(records[left_index].get("wins", 0)))
        scores.append(
            DeckCandidateScore(
                record_index=left_index,
                centrality=weighted_similarity / total_weight,
                adjusted_win_rate=_adjusted_win_rate(
                    wins,
                    games,
                    cluster_win_rate,
                    win_rate_prior_games,
                ),
                games=games,
                wins=wins,
                raw_win_rate=wins / games if games else 0.0,
            )
        )
    return scores


def _select_representative(
    records: list[dict[str, Any]],
    scores: list[DeckCandidateScore],
    tolerance: float,
) -> DeckCandidateScore:
    """最大中心性付近の候補から実績の安定した代表デッキを選ぶ。"""

    maximum_centrality = max(score.centrality for score in scores)
    eligible = [
        score
        for score in scores
        if score.centrality >= maximum_centrality - tolerance
    ]
    return max(
        eligible,
        key=lambda score: (
            score.games,
            score.adjusted_win_rate,
            score.centrality,
            _stable_deck_key(records[score.record_index]),
        ),
    )


def _select_competitive(
    records: list[dict[str, Any]],
    scores: list[DeckCandidateScore],
    tolerance: float,
    minimum_games: int,
) -> tuple[DeckCandidateScore, bool]:
    """中心性を保つ候補から補正勝率が高い競技向けデッキを選ぶ。"""

    maximum_centrality = max(score.centrality for score in scores)
    central_candidates = [
        score
        for score in scores
        if score.centrality >= maximum_centrality - tolerance
    ]
    sufficiently_observed = [
        score for score in central_candidates if score.games >= minimum_games
    ]
    eligible = sufficiently_observed or central_candidates
    return (
        max(
            eligible,
            key=lambda score: (
                score.adjusted_win_rate,
                score.games,
                score.centrality,
                _stable_deck_key(records[score.record_index]),
            ),
        ),
        bool(sufficiently_observed),
    )


def select_cluster_decks(
    records: list[dict[str, Any]],
    cluster_field: str = DEFAULT_CLUSTER_FIELD,
    representative_tolerance: float = DEFAULT_REPRESENTATIVE_TOLERANCE,
    competitive_tolerance: float = DEFAULT_COMPETITIVE_TOLERANCE,
    win_rate_prior_games: float = DEFAULT_WIN_RATE_PRIOR_GAMES,
    minimum_competitive_games: int = DEFAULT_MINIMUM_COMPETITIVE_GAMES,
    minimum_major_game_share: float = DEFAULT_MINIMUM_MAJOR_GAME_SHARE,
) -> list[ClusterDeckSelection]:
    """全クラスタの主要デッキを選び、選出指標を入力レコードへ追加する。"""

    _validate_parameters(
        representative_tolerance,
        competitive_tolerance,
        win_rate_prior_games,
        minimum_competitive_games,
        minimum_major_game_share,
    )
    if not records:
        raise ValueError("入力JSONLにデッキレコードがありません。")

    groups: dict[int, list[int]] = defaultdict(list)
    for record_index, record in enumerate(records):
        if cluster_field not in record:
            raise ValueError(f"入力レコードに{cluster_field}がありません。")
        groups[int(record[cluster_field])].append(record_index)

    total_games = sum(max(0, int(record.get("games", 0))) for record in records)
    selections: list[ClusterDeckSelection] = []
    for cluster_id in sorted(groups):
        member_indices = groups[cluster_id]
        cluster_games = sum(max(0, int(records[index].get("games", 0))) for index in member_indices)
        cluster_wins = sum(max(0, int(records[index].get("wins", 0))) for index in member_indices)
        cluster_win_rate = cluster_wins / cluster_games if cluster_games else 0.0
        game_share = cluster_games / total_games if total_games else 0.0
        scores = _score_cluster_candidates(
            records,
            member_indices,
            cluster_win_rate,
            win_rate_prior_games,
        )
        representative = _select_representative(
            records,
            scores,
            representative_tolerance,
        )
        competitive, minimum_games_applied = _select_competitive(
            records,
            scores,
            competitive_tolerance,
            minimum_competitive_games,
        )
        is_major_cluster = game_share >= minimum_major_game_share

        score_by_index = {score.record_index: score for score in scores}
        for record_index in member_indices:
            record = records[record_index]
            score = score_by_index[record_index]
            record["hierarchical_cluster_centrality"] = score.centrality
            record["hierarchical_adjusted_win_rate"] = score.adjusted_win_rate
            record["hierarchical_cluster_representative"] = (
                record_index == representative.record_index
            )
            record["hierarchical_cluster_competitive"] = (
                record_index == competitive.record_index
            )
            record["hierarchical_is_major_cluster"] = is_major_cluster

        selections.append(
            ClusterDeckSelection(
                cluster_id=cluster_id,
                member_count=len(member_indices),
                games=cluster_games,
                wins=cluster_wins,
                cluster_win_rate=cluster_win_rate,
                game_share=game_share,
                is_major_cluster=is_major_cluster,
                representative=representative,
                competitive=competitive,
                competitive_minimum_games_applied=minimum_games_applied,
            )
        )
    return selections


def _deck_csv(record: dict[str, Any]) -> str:
    """対戦ツール互換のカードID1列・60行CSV文字列を返す。"""

    histogram = count_histogram(record)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for card_id, count in sorted(histogram.items()):
        for _ in range(count):
            writer.writerow([card_id])
    return output.getvalue()


def _main_pokemon(
    record: dict[str, Any],
    card_metadata: dict[int, CardMetadata],
    limit: int = 8,
) -> str:
    """選出デッキの採用ポケモンを枚数順の短い文字列にする。"""

    pokemon = [
        (count, card_metadata[card_id].name, card_id)
        for card_id, count in count_histogram(record).items()
        if card_id in card_metadata and card_metadata[card_id].is_pokemon
    ]
    pokemon.sort(key=lambda item: (-item[0], item[1], item[2]))
    return " / ".join(f"{name}×{count}" for count, name, _ in pokemon[:limit])


def _selection_dict(
    selection: ClusterDeckSelection,
    records: list[dict[str, Any]],
    card_metadata: dict[int, CardMetadata],
) -> dict[str, Any]:
    """選出結果をJSON・CSVへ共通利用できる辞書へ変換する。"""

    representative_record = records[selection.representative.record_index]
    competitive_record = records[selection.competitive.record_index]
    return {
        "cluster_id": selection.cluster_id,
        "member_count": selection.member_count,
        "cluster_games": selection.games,
        "cluster_wins": selection.wins,
        "cluster_win_rate": selection.cluster_win_rate,
        "cluster_game_share": selection.game_share,
        "is_major_cluster": selection.is_major_cluster,
        "representative_filename": f"cluster_{selection.cluster_id:02d}_representative.csv",
        "representative_games": selection.representative.games,
        "representative_win_rate": selection.representative.raw_win_rate,
        "representative_adjusted_win_rate": selection.representative.adjusted_win_rate,
        "representative_centrality": selection.representative.centrality,
        "representative_main_pokemon": _main_pokemon(representative_record, card_metadata),
        "competitive_filename": f"cluster_{selection.cluster_id:02d}_competitive.csv",
        "competitive_games": selection.competitive.games,
        "competitive_win_rate": selection.competitive.raw_win_rate,
        "competitive_adjusted_win_rate": selection.competitive.adjusted_win_rate,
        "competitive_centrality": selection.competitive.centrality,
        "competitive_main_pokemon": _main_pokemon(competitive_record, card_metadata),
        "competitive_minimum_games_applied": selection.competitive_minimum_games_applied,
        "same_deck_for_both": (
            selection.representative.record_index == selection.competitive.record_index
        ),
    }


def _manifest_csv(rows: list[dict[str, Any]]) -> str:
    """クラスタ別選出根拠を一覧化したCSV文字列を返す。"""

    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render_selection_markdown(
    rows: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> str:
    """主要デッキ一覧と選出ロジックを説明する日本語Markdownを返す。"""

    lines = [
        "# 階層クラスタ主要デッキ",
        "",
        "各CSVは観測済みの実在60枚デッキで、カードIDを1行1枚で記載しています。",
        "",
        "## 選出方法",
        "",
        "- 代表デッキ: 対戦数加重のクラスタ内平均類似度（中心性）が最大付近の候補から、使用試合数を優先して選出。",
        "- 競技向けデッキ: 最大中心性を大きく外れない候補から、クラスタ平均へ縮約した補正勝率を優先して選出。",
        f"- 代表中心性許容差: {parameters['representative_tolerance']:.3f}",
        f"- 競技向け中心性許容差: {parameters['competitive_tolerance']:.3f}",
        f"- 勝率事前試合数: {parameters['win_rate_prior_games']:.1f}",
        f"- 競技向け最低試合数: {parameters['minimum_competitive_games']}",
        f"- 主要クラスタ判定の最低対戦シェア: {parameters['minimum_major_game_share']:.2%}",
        "",
        "## クラスタ別一覧",
        "",
        "|ID|主要|デッキ数|対戦数|シェア|代表中心性|代表CSV|競技向け補正勝率|競技向けCSV|同一構築|",
        "|---:|:---:|---:|---:|---:|---:|---|---:|---|:---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['cluster_id']}|{'○' if row['is_major_cluster'] else '-'}|"
            f"{row['member_count']}|{row['cluster_games']}|{row['cluster_game_share']:.2%}|"
            f"{row['representative_centrality']:.4f}|"
            f"[{row['representative_filename']}]({row['representative_filename']})|"
            f"{row['competitive_adjusted_win_rate']:.2%}|"
            f"[{row['competitive_filename']}]({row['competitive_filename']})|"
            f"{'○' if row['same_deck_for_both'] else '-'}|"
        )
    lines.extend(
        [
            "",
            "`is_major_cluster` は対戦シェアによる運用上の目印です。小規模クラスタも削除せず、比較・再現用にCSVを出力しています。",
            "",
        ]
    )
    return "\n".join(lines)


def write_selection_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    selections: list[ClusterDeckSelection],
    card_metadata: dict[int, CardMetadata],
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    """60枚CSVと選出根拠のmanifest・JSON・Markdownを原子的に書き出す。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _selection_dict(selection, records, card_metadata)
        for selection in selections
    ]
    for selection, row in zip(selections, rows, strict=True):
        write_text_atomic(
            output_dir / row["representative_filename"],
            _deck_csv(records[selection.representative.record_index]),
        )
        write_text_atomic(
            output_dir / row["competitive_filename"],
            _deck_csv(records[selection.competitive.record_index]),
        )
    write_text_atomic(output_dir / "manifest.csv", _manifest_csv(rows))
    summary = {"parameters": parameters, "clusters": rows}
    write_text_atomic(
        output_dir / "selection_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(
        output_dir / "README.md",
        render_selection_markdown(rows, parameters),
    )
    return rows


def main() -> int:
    """入力JSONLからクラスタ主要デッキ一式を生成する。"""

    args = parse_args()
    output_jsonl = args.output_jsonl or args.input
    records = load_records(args.input)
    cluster_count = len({int(record[args.cluster_field]) for record in records})
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_DIR / f"cluster_major_decks_{cluster_count}"
    )
    parameters = {
        "cluster_field": args.cluster_field,
        "representative_tolerance": args.representative_tolerance,
        "competitive_tolerance": args.competitive_tolerance,
        "win_rate_prior_games": args.win_rate_prior_games,
        "minimum_competitive_games": args.minimum_competitive_games,
        "minimum_major_game_share": args.minimum_major_game_share,
    }
    selections = select_cluster_decks(records, **parameters)
    card_metadata = load_card_metadata(args.card_data)
    write_selection_outputs(
        output_dir,
        records,
        selections,
        card_metadata,
        parameters,
    )
    write_records_atomic(output_jsonl, records)
    print(f"selected representative and competitive decks for {len(selections)} clusters")
    print(f"wrote selected deck files to {output_dir}")
    print(f"wrote selection fields to {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
