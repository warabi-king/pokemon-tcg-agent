"""学習ログCSVと対戦結果JSONを読み、比較用PNGグラフを出力する。

実行例:
    python tools/plot_experiment_results.py
    python tools/plot_experiment_results.py --league-dir results/upsize1_vs_belief_20260804

既定では ``results/plots`` に以下を保存する。

* ``imitation_learning.png``: imitation 学習の損失・正解率
* ``league_ranking.png``: 完了済み試合だけに基づくリーグ順位
* ``belief_puct_matchups.png``: belief_puct 対各 cluster の直接対戦成績

入力ファイルは変更せず、試合数が 0 の参加者は順位グラフから除外する。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_FILE = PROJECT_ROOT / "agents" / "belief_puct" / "train" / "logs" / "imitation_metrics.csv"
DEFAULT_LEAGUE_DIR = PROJECT_ROOT / "results" / "upsize1_vs_belief_20260804"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"

# 読み取り専用の結果解析でも、Matplotlib がユーザー設定を書き換えないようにする。
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_OUTPUT_DIR / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class LeagueRow:
    """リーグ順位グラフで使用する、1参加者分の集計値。"""

    name: str
    games_completed: int
    win_rate: float
    lower_95: float
    upper_95: float


def read_imitation_metrics(metrics_file: Path) -> list[dict[str, float]]:
    """imitation_metrics.csv を数値行として読み込む。"""
    rows: list[dict[str, float]] = []
    with metrics_file.open(newline="", encoding="utf-8") as file:
        for raw_row in csv.DictReader(file):
            row: dict[str, float] = {}
            for key, value in raw_row.items():
                if value not in (None, ""):
                    row[key] = float(value)
            if row:
                rows.append(row)
    if not rows:
        raise ValueError(f"学習ログが空です: {metrics_file}")
    return rows


def read_json(json_file: Path) -> dict[str, Any]:
    """JSONオブジェクトを読み込み、形式が不正なら原因を示して失敗する。"""
    with json_file.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSONオブジェクトではありません: {json_file}")
    return value


def read_league_ranking(aggregate_file: Path) -> list[LeagueRow]:
    """aggregate.json のうち、少なくとも1試合完了した参加者だけを抽出する。"""
    aggregate = read_json(aggregate_file)
    ranking = aggregate.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"ranking がありません: {aggregate_file}")

    rows: list[LeagueRow] = []
    for raw_row in ranking:
        if not isinstance(raw_row, dict):
            continue
        games_completed = int(raw_row.get("games_completed", 0))
        if games_completed <= 0:
            continue
        rows.append(
            LeagueRow(
                name=str(raw_row["name"]),
                games_completed=games_completed,
                win_rate=float(raw_row["win_rate"]),
                lower_95=float(raw_row["wilson_lower_95"]),
                upper_95=float(raw_row["wilson_upper_95"]),
            )
        )
    return sorted(rows, key=lambda row: row.win_rate, reverse=True)


def read_belief_matchups(pairings_dir: Path) -> list[LeagueRow]:
    """pairings内の belief_puct を agent_a とする完了済み対戦を読み込む。"""
    rows: list[LeagueRow] = []
    for pairing_file in sorted(pairings_dir.glob("belief_puct_vs_*.json")):
        report = read_json(pairing_file)
        configuration = report.get("configuration", {})
        summary = report.get("summary", {})
        if not isinstance(configuration, dict) or not isinstance(summary, dict):
            continue
        games_completed = int(summary.get("games_completed", 0))
        if games_completed <= 0:
            continue
        rows.append(
            LeagueRow(
                name=str(configuration.get("name_b", pairing_file.stem)),
                games_completed=games_completed,
                win_rate=float(summary["score_a"]),
                lower_95=float(summary["wilson_lower_95"]),
                upper_95=float(summary["wilson_upper_95"]),
            )
        )
    return sorted(rows, key=lambda row: row.name)


def configure_style() -> None:
    """全PNGで共通の読みやすい描画設定を適用する。"""
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.25})


def plot_imitation_learning(rows: list[dict[str, float]], output_file: Path) -> None:
    """損失と train/validation accuracy を上下2段のグラフとして保存する。"""
    train_rows = [row for row in rows if row.get("epoch", -1) >= 0]
    if not train_rows:
        raise ValueError("epoch 0 以降の学習ログがありません")

    epochs = [row["epoch"] for row in train_rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(epochs, [row["loss"] for row in train_rows], label="total loss")
    axes[0].plot(epochs, [row["loss_policy"] for row in train_rows], label="policy loss")
    axes[0].plot(epochs, [row["loss_value"] for row in train_rows], label="value loss")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Belief PUCT imitation learning")
    axes[0].legend()

    axes[1].plot(epochs, [100 * row["train_accuracy"] for row in train_rows], label="train accuracy")
    axes[1].plot(epochs, [100 * row["val_accuracy"] for row in train_rows], label="validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def plot_win_rates(rows: list[LeagueRow], title: str, output_file: Path) -> None:
    """Wilson 95%区間を含む横棒グラフとして勝率を保存する。"""
    if not rows:
        print(f"skip: 対象となる完了済み試合がありません ({title})")
        return

    labels = [f"{row.name} (n={row.games_completed})" for row in rows]
    win_rates = [100 * row.win_rate for row in rows]
    lower_errors = [100 * (row.win_rate - row.lower_95) for row in rows]
    upper_errors = [100 * (row.upper_95 - row.win_rate) for row in rows]
    height = max(4, 0.48 * len(rows) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    positions = list(range(len(rows)))
    axis.barh(positions, win_rates, xerr=[lower_errors, upper_errors], capsize=3)
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("win rate (%) with Wilson 95% interval")
    axis.set_title(title)
    axis.invert_yaxis()
    for position, value in zip(positions, win_rates):
        axis.text(min(value + 1.5, 97), position, f"{value:.1f}%", va="center")
    figure.tight_layout()
    figure.savefig(output_file)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を読み取り、既定の入力・出力先を提供する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS_FILE)
    parser.add_argument("--league-dir", type=Path, default=DEFAULT_LEAGUE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """指定されたログと結果から、再実行可能なPNG群を生成する。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    plot_imitation_learning(read_imitation_metrics(args.metrics_file), args.output_dir / "imitation_learning.png")
    plot_win_rates(
        read_league_ranking(args.league_dir / "aggregate.json"),
        "Cross-worktree league ranking (completed games only)",
        args.output_dir / "league_ranking.png",
    )
    plot_win_rates(
        read_belief_matchups(args.league_dir / "pairings"),
        "belief_puct vs each cluster",
        args.output_dir / "belief_puct_matchups.png",
    )
    print(f"Graphs saved: {args.output_dir}")


if __name__ == "__main__":
    main()
