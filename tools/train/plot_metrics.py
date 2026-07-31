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


def read_dual_metrics(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    """log_dir/a/train_metrics.csv と log_dir/b/train_metrics.csv を読み込む。"""
    metrics = {}
    for side in ("a", "b"):
        path = log_dir / side / "train_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dual metrics file not found: {path}")
        metrics[side] = read_metrics(path)
    return metrics


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
    if metrics_path.is_dir():
        metrics_map = read_dual_metrics(metrics_path)
        iterations_a = [row["iteration"] for row in metrics_map["a"]]
        iterations_b = [row["iteration"] for row in metrics_map["b"]]

        save_line_plot(
            output_dir / "loss_a_b.png",
            "Training Loss (A/B)",
            sorted(set(iterations_a + iterations_b)),
            {
                "A total": [row["loss"] for row in metrics_map["a"]],
                "A value": [row["loss_value"] for row in metrics_map["a"]],
                "A policy": [row["loss_policy"] for row in metrics_map["a"]],
                "B total": [row["loss"] for row in metrics_map["b"]],
                "B value": [row["loss_value"] for row in metrics_map["b"]],
                "B policy": [row["loss_policy"] for row in metrics_map["b"]],
            },
            "loss",
        )
        save_line_plot(
            output_dir / "eval_win_rate_a_b.png",
            "Evaluation Win Rate (A/B)",
            sorted(set(iterations_a + iterations_b)),
            {
                "A win_rate": [row["eval_win_rate"] for row in metrics_map["a"]],
                "B win_rate": [row["eval_win_rate"] for row in metrics_map["b"]],
            },
            "win rate (%)",
        )
        save_line_plot(
            output_dir / "samples_batches_a_b.png",
            "Samples and Batches (A/B)",
            sorted(set(iterations_a + iterations_b)),
            {
                "A samples": [row["samples"] for row in metrics_map["a"]],
                "A batches": [row["batches"] for row in metrics_map["a"]],
                "B samples": [row["samples"] for row in metrics_map["b"]],
                "B batches": [row["batches"] for row in metrics_map["b"]],
            },
            "count",
        )
        save_line_plot(
            output_dir / "elapsed_seconds_a_b.png",
            "Elapsed Seconds (A/B)",
            sorted(set(iterations_a + iterations_b)),
            {
                "A elapsed_seconds": [row["elapsed_seconds"] for row in metrics_map["a"]],
                "B elapsed_seconds": [row["elapsed_seconds"] for row in metrics_map["b"]],
            },
            "seconds",
        )
        print(f"Graphs saved: {output_dir}")
        print("\nDual model loss summary:")
        print("iteration,A_loss,B_loss,A_value,B_value,A_policy,B_policy")
        for i in range(max(len(metrics_map["a"]), len(metrics_map["b"]))):
            row_a = metrics_map["a"][i] if i < len(metrics_map["a"]) else None
            row_b = metrics_map["b"][i] if i < len(metrics_map["b"]) else None
            print(
                ",".join(
                    [
                        str(i),
                        str(row_a["loss"]) if row_a else "",
                        str(row_b["loss"]) if row_b else "",
                        str(row_a["loss_value"]) if row_a else "",
                        str(row_b["loss_value"]) if row_b else "",
                        str(row_a["loss_policy"]) if row_a else "",
                        str(row_b["loss_policy"]) if row_b else "",
                    ]
                )
            )
        return

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
