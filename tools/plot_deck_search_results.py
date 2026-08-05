"""固定リーグのデッキ探索結果をPNGグラフへ出力する。

実行例:
    python tools/plot_deck_search_results.py \
        --results-dir results/deck_search_phase2a-001

``ranking.json`` を読み、順位・総合勝率・最苦手相手の勝率を示す
``deck_ranking.png`` と、候補ごとの相手別勝率を示す
``deck_matchup_matrix.png`` を保存する。入力結果は変更しない。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "plots"

# 読み取り専用の結果解析でも、Matplotlib がユーザー設定を書き換えないようにする。
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_ROOT / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CandidateResult:
    """順位表と相手別成績を描くための、候補1件の正規化済み結果。"""

    rank: int
    index: int
    weighted_score: float
    worst_score: float
    weakest_opponent: str
    games_scored: int
    opponent_scores: dict[str, float]


def read_ranking(results_dir: Path) -> list[CandidateResult]:
    """deck searchのranking.jsonを検証し、描画用の候補リストとして返す。"""

    ranking_path = results_dir / "ranking.json"
    try:
        content: Any = json.loads(ranking_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"ranking.jsonがありません: {ranking_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"ranking.jsonを解釈できません: {ranking_path}: {error}") from error
    raw_ranking = content.get("ranking") if isinstance(content, dict) else None
    if not isinstance(raw_ranking, list) or not raw_ranking:
        raise ValueError(f"ranking配列がありません: {ranking_path}")

    candidates: list[CandidateResult] = []
    for raw in raw_ranking:
        if not isinstance(raw, dict):
            raise ValueError("rankingの各要素はobjectである必要があります")
        raw_matchups = raw.get("per_opponent")
        if not isinstance(raw_matchups, list):
            raise ValueError("rankingの各候補にはper_opponent配列が必要です")
        opponent_scores = {
            str(matchup["name"]): float(matchup["score"])
            for matchup in raw_matchups
            if isinstance(matchup, dict) and "name" in matchup and "score" in matchup
        }
        candidates.append(
            CandidateResult(
                rank=int(raw["rank"]),
                index=int(raw["candidate_index"]),
                weighted_score=float(raw["weighted_score"]),
                worst_score=float(raw["worst_opponent_score"]),
                weakest_opponent=str(raw["weakest_opponent"]),
                games_scored=int(raw["games_scored"]),
                opponent_scores=opponent_scores,
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.rank)


def configure_style() -> None:
    """比較用PNGで共通の視認性設定を適用する。"""

    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})


def plot_ranking(candidates: list[CandidateResult], output_path: Path) -> None:
    """総合勝率と最苦手相手勝率を候補順位順の横棒として保存する。"""

    labels = [f"#{item.rank} candidate_{item.index:03d} (n={item.games_scored})" for item in candidates]
    positions = list(range(len(candidates)))
    weighted = [100 * item.weighted_score for item in candidates]
    worst = [100 * item.worst_score for item in candidates]
    height = max(5, 0.42 * len(candidates) + 1.8)
    figure, axis = plt.subplots(figsize=(11, height))
    axis.barh(positions, weighted, label="weighted league score")
    axis.barh(positions, worst, label="worst-opponent score", alpha=0.72)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("score (%)")
    axis.set_title("Deck search fixed-league ranking")
    axis.invert_yaxis()
    axis.legend(loc="lower right")
    for position, item in zip(positions, candidates):
        axis.text(
            min(100, 100 * item.weighted_score + 1.2),
            position,
            f"weakest: {item.weakest_opponent}",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_matchup_matrix(candidates: list[CandidateResult], output_path: Path) -> None:
    """候補×固定相手の勝率を色と数値で比較できる行列として保存する。"""

    opponent_names = sorted({name for item in candidates for name in item.opponent_scores})
    if not opponent_names:
        raise ValueError("相手別スコアがありません")
    matrix = [[100 * item.opponent_scores.get(name, 0.0) for name in opponent_names] for item in candidates]
    figure, axis = plt.subplots(figsize=(max(7, 2.1 * len(opponent_names)), max(5, 0.42 * len(candidates) + 1.8)))
    image = axis.imshow(matrix, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
    axis.set_xticks(range(len(opponent_names)), opponent_names, rotation=20, ha="right")
    axis.set_yticks(range(len(candidates)), [f"#{item.rank} candidate_{item.index:03d}" for item in candidates])
    axis.set_title("Win rate by candidate deck and frozen opponent")
    for row_index, row in enumerate(matrix):
        for column_index, score in enumerate(row):
            axis.text(column_index, row_index, f"{score:.1f}%", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="win rate (%)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """入力リーグ結果と出力先のCLI引数を読む。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="ranking.jsonを含む固定リーグ結果ディレクトリ")
    parser.add_argument("--output-dir", type=Path, default=None, help="PNG出力先。省略時はresults/plots/<結果名>")
    return parser.parse_args()


def main() -> None:
    """固定リーグの順位・相手別成績グラフを作成する。"""

    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else DEFAULT_OUTPUT_ROOT / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    candidates = read_ranking(results_dir)
    plot_ranking(candidates, output_dir / "deck_ranking.png")
    plot_matchup_matrix(candidates, output_dir / "deck_matchup_matrix.png")
    print(f"Graphs saved: {output_dir}")


if __name__ == "__main__":
    main()
