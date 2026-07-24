"""Create plots for train_match_agents.py metrics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "agents" / "match_agents" / "train" / "logs"
DEFAULT_METRICS = DEFAULT_LOG_DIR / "train_metrics.csv"
DEFAULT_PAIR_METRICS = DEFAULT_LOG_DIR / "pair_metrics.csv"
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_LOG_DIR / "matplotlib_config"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else 0.0


def grouped_by_agent(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["agent"]].append(row)
    for agent_rows in grouped.values():
        agent_rows.sort(key=lambda row: to_float(row, "iteration"))
    return dict(grouped)


def save_line_plot(
    output_path: Path,
    title: str,
    grouped_rows: dict[str, list[dict[str, str]]],
    key: str,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for agent, rows in grouped_rows.items():
        x = [to_float(row, "iteration") for row in rows]
        y = [to_float(row, key) for row in rows]
        plt.plot(x, y, marker="o", label=agent)
    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_loss_components(output_path: Path, rows: list[dict[str, str]]) -> None:
    grouped = grouped_by_agent(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for key, ylabel, axis in [
        ("loss", "total loss", axes[0]),
        ("loss_value", "value loss", axes[1]),
        ("loss_policy", "policy loss", axes[2]),
    ]:
        for agent, agent_rows in grouped.items():
            x = [to_float(row, "iteration") for row in agent_rows]
            y = [to_float(row, key) for row in agent_rows]
            axis.plot(x, y, marker="o", label=agent)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[0].set_title("Training Loss Components")
    axes[-1].set_xlabel("iteration")
    axes[0].legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_pair_heatmap(output_path: Path, pair_rows: list[dict[str, str]]) -> None:
    if not pair_rows:
        return

    latest_iteration = max(to_float(row, "iteration") for row in pair_rows)
    latest_rows = [row for row in pair_rows if to_float(row, "iteration") == latest_iteration]
    agents = sorted({row["agent"] for row in latest_rows} | {row["opponent"] for row in latest_rows})
    index = {agent: i for i, agent in enumerate(agents)}
    matrix = [[float("nan") for _ in agents] for _ in agents]
    for row in latest_rows:
        matrix[index[row["agent"]]][index[row["opponent"]]] = to_float(row, "win_rate")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    image = plt.imshow(matrix, cmap="viridis", vmin=0, vmax=100)
    plt.colorbar(image, label="win rate (%)")
    plt.title(f"Pair Win Rate Matrix (iteration {int(latest_iteration)})")
    plt.xticks(range(len(agents)), agents, rotation=45, ha="right")
    plt.yticks(range(len(agents)), agents)
    for y, agent in enumerate(agents):
        for x, opponent in enumerate(agents):
            value = matrix[y][x]
            if value == value:
                plt.text(x, y, f"{value:.0f}", ha="center", va="center", color="white")
    plt.xlabel("opponent")
    plt.ylabel("agent")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_match_metrics(
    metrics_path: Path = DEFAULT_METRICS,
    pair_metrics_path: Path = DEFAULT_PAIR_METRICS,
    output_dir: Path = DEFAULT_LOG_DIR,
) -> None:
    rows = read_rows(metrics_path)
    if not rows:
        raise ValueError(f"No metrics rows found: {metrics_path}")
    grouped = grouped_by_agent(rows)

    save_line_plot(output_dir / "match_win_rate.png", "Tournament Win Rate", grouped, "win_rate", "win rate (%)")
    save_line_plot(output_dir / "match_samples.png", "Training Samples", grouped, "samples", "samples")
    save_line_plot(output_dir / "match_batches.png", "Training Batches", grouped, "batches", "batches")
    save_line_plot(output_dir / "match_elapsed_seconds.png", "Elapsed Seconds", grouped, "elapsed_seconds", "seconds")
    save_loss_components(output_dir / "match_loss_components.png", rows)

    if pair_metrics_path.exists():
        save_pair_heatmap(output_dir / "match_pair_win_rate.png", read_rows(pair_metrics_path))

    print(f"Graphs saved: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-file", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--pair-metrics-file", type=Path, default=DEFAULT_PAIR_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LOG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_match_metrics(args.metrics_file, args.pair_metrics_file, args.output_dir)


if __name__ == "__main__":
    main()
