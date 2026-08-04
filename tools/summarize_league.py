"""複数の固定リーグ評価JSONを一つの判定用JSONへ集約する。

実行例:
    python tools/summarize_league.py \
        --inputs results/stage1_final_cluster05_vs_pretrained_cluster_*_20.json \
        --label stage1-final --output results/stage1_final_aggregate_100.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_fixed_league import wilson_interval


def aggregate_reports(reports: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """互換形式の評価結果を合算し、勝率とWilson区間を再計算する。"""

    if not reports:
        raise ValueError("集約対象の評価JSONがありません")
    for report in reports:
        if report.get("format") != "pokemon-tcg-agent/fixed-league-evaluation-v1":
            raise ValueError("未対応の評価JSON形式です")

    summaries = [report["summary"] for report in reports]
    wins_a = sum(int(summary["wins_a"]) for summary in summaries)
    wins_b = sum(int(summary["wins_b"]) for summary in summaries)
    draws = sum(int(summary["draws"]) for summary in summaries)
    errors = sum(int(summary["errors"]) for summary in summaries)
    requested = sum(int(summary["games_requested"]) for summary in summaries)
    completed = wins_a + wins_b + draws
    successes = wins_a + 0.5 * draws
    score = successes / completed if completed else 0.0
    lower, upper = wilson_interval(successes, completed)
    elapsed = sum(float(summary["elapsed_seconds"]) for summary in summaries)

    return {
        "format": "pokemon-tcg-agent/fixed-league-aggregate-v1",
        "label": label,
        "summary": {
            "games_requested": requested,
            "games_completed": completed,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "draws": draws,
            "errors": errors,
            "score_a": score,
            "wilson_lower_95": lower,
            "wilson_upper_95": upper,
            "elapsed_seconds": elapsed,
            "mean_seconds_per_game": elapsed / requested if requested else 0.0,
        },
        "components": [
            {
                "configuration": report["configuration"],
                "summary": report["summary"],
            }
            for report in reports
        ],
    }


def parse_args() -> argparse.Namespace:
    """入力JSON列、ラベル、出力先を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """評価JSONを読み、集約結果を保存して表示する。"""

    args = parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    aggregate = aggregate_reports(reports, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate["summary"], ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
