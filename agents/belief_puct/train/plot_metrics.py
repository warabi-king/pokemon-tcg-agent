"""rl_mctsの学習メトリクスCSVからPNGグラフを生成する。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = AGENT_ROOT / "train" / "logs"
DEFAULT_METRICS = DEFAULT_LOG_DIR / "train_metrics.csv"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_LOG_DIR / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(metrics_path: Path) -> list[dict[str, float]]:
    """CSVから数値メトリクスを読み込む。"""
    rows: list[dict[str, float]] = []
    with metrics_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "iteration": float(row["iteration"]),
                    "eval_win_rate": float(row["eval_win_rate"]),
                    "samples": float(row["samples"]),
                    "batches": float(row["batches"]),
                    "loss": float(row["loss"]),
                    "loss_value": float(row["loss_value"]),
                    "loss_policy": float(row["loss_policy"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                }
            )
    if not rows:
        raise ValueError(f"メトリクスが空です: {metrics_path}")
    return rows


def save_line_plot(
    output_path: Path,
    title: str,
    x: list[float],
    series: dict[str, list[float]],
    ylabel: str,
) -> None:
    """複数系列の折れ線グラフを保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for label, values in series.items():
        plt.plot(x, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metrics(metrics_path: Path = DEFAULT_METRICS, output_dir: Path = DEFAULT_LOG_DIR) -> None:
    """train_metrics.csvから標準グラフを生成する。"""
    rows = read_metrics(metrics_path)
    iterations = [row["iteration"] for row in rows]

    save_line_plot(
        output_dir / "loss.png",
        "Training Loss",
        iterations,
        {
            "total": [row["loss"] for row in rows],
            "value": [row["loss_value"] for row in rows],
            "policy": [row["loss_policy"] for row in rows],
        },
        "loss",
    )
    save_line_plot(
        output_dir / "eval_win_rate.png",
        "Evaluation Win Rate",
        iterations,
        {"win_rate": [row["eval_win_rate"] for row in rows]},
        "win rate (%)",
    )
    save_line_plot(
        output_dir / "samples_batches.png",
        "Samples and Batches",
        iterations,
        {
            "samples": [row["samples"] for row in rows],
            "batches": [row["batches"] for row in rows],
        },
        "count",
    )
    save_line_plot(
        output_dir / "elapsed_seconds.png",
        "Elapsed Seconds",
        iterations,
        {"elapsed_seconds": [row["elapsed_seconds"] for row in rows]},
        "seconds",
    )
    print(f"Graphs saved: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metrics(args.metrics_file, args.output_dir)


if __name__ == "__main__":
    main()
