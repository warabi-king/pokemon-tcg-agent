"""既存デッキの16クラスタ化、精度評価、主要デッキ選出を1コマンドで実行する。

既存の ``cluster_*`` フィールドを保持したまま、平均連結法の凝集型階層
クラスタリング結果を ``hierarchical_*`` として追加する。その後、定量・定性の
評価結果をJSONと日本語Markdownへ出力し、各クラスタの代表・競技向け60枚
デッキCSVも生成する。

実行例:
    python tools/clustering_deck/run_hierarchical_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_hierarchical_clustering import (
    DEFAULT_BOUNDARY_SIMILARITY,
    DEFAULT_CARD_DATA,
    evaluate_records,
    load_card_metadata,
    render_markdown,
    write_text_atomic,
)
from hierarchical_clustering import (
    DEFAULT_INPUT,
    add_hierarchical_fields,
    load_records,
    write_records_atomic,
)
from render_hierarchical_dendrogram import render_cut_cluster_dendrogram
from select_cluster_major_decks import (
    DEFAULT_COMPETITIVE_TOLERANCE,
    DEFAULT_MINIMUM_COMPETITIVE_GAMES,
    DEFAULT_MINIMUM_MAJOR_GAME_SHARE,
    DEFAULT_REPRESENTATIVE_TOLERANCE,
    DEFAULT_WIN_RATE_PRIOR_GAMES,
    select_cluster_decks,
    write_selection_outputs,
)


DEFAULT_CLUSTER_COUNT = 16
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"


def parse_args() -> argparse.Namespace:
    """クラスタリングと評価を一括実行するための引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="入力JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="クラスタ追加後のJSONL。省略時は入力ファイルを安全に置換する。",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=DEFAULT_CLUSTER_COUNT,
        help="階層を切断するクラスタ数（デフォルト: 16）",
    )
    parser.add_argument("--card-data", type=Path, default=DEFAULT_CARD_DATA)
    parser.add_argument(
        "--boundary-similarity",
        type=float,
        default=DEFAULT_BOUNDARY_SIMILARITY,
    )
    parser.add_argument("--evaluation-json", type=Path, default=None)
    parser.add_argument("--evaluation-markdown", type=Path, default=None)
    parser.add_argument("--dendrogram-svg", type=Path, default=None)
    parser.add_argument("--dendrogram-png", type=Path, default=None)
    parser.add_argument("--major-decks-dir", type=Path, default=None)
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
    parser.add_argument(
        "--no-dendrogram",
        action="store_true",
        help="デンドログラムを出力しない。",
    )
    return parser.parse_args()


def main() -> int:
    """既存デッキを階層クラスタリングし、検証後に全成果物を書き込む。"""

    args = parse_args()
    output_path = args.output or args.input
    evaluation_json = args.evaluation_json or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_cluster_evaluation_{args.clusters}.json"
    )
    evaluation_markdown = args.evaluation_markdown or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_cluster_evaluation_{args.clusters}.md"
    )
    dendrogram_svg = args.dendrogram_svg or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_dendrogram_{args.clusters}.svg"
    )
    dendrogram_png = args.dendrogram_png or (
        DEFAULT_OUTPUT_DIR / f"hierarchical_dendrogram_{args.clusters}.png"
    )
    major_decks_dir = args.major_decks_dir or (
        DEFAULT_OUTPUT_DIR / f"cluster_major_decks_{args.clusters}"
    )

    records = load_records(args.input)
    clustering_result = add_hierarchical_fields(records, args.clusters)
    card_metadata = load_card_metadata(args.card_data)
    report = evaluate_records(
        records,
        card_metadata,
        boundary_similarity=args.boundary_similarity,
    )
    if report["method"]["cluster_count"] != args.clusters:
        raise RuntimeError("指定したクラスタ数と評価対象のクラスタ数が一致しません。")

    selection_parameters = {
        "cluster_field": "hierarchical_cluster_id",
        "representative_tolerance": args.representative_tolerance,
        "competitive_tolerance": args.competitive_tolerance,
        "win_rate_prior_games": args.win_rate_prior_games,
        "minimum_competitive_games": args.minimum_competitive_games,
        "minimum_major_game_share": args.minimum_major_game_share,
    }
    selections = select_cluster_decks(records, **selection_parameters)
    write_selection_outputs(
        major_decks_dir,
        records,
        selections,
        card_metadata,
        selection_parameters,
    )

    write_records_atomic(output_path, records)
    write_text_atomic(
        evaluation_json,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomic(evaluation_markdown, render_markdown(report))
    if not args.no_dendrogram:
        render_cut_cluster_dendrogram(
            records,
            report["clusters"],
            dendrogram_svg,
            dendrogram_png,
        )
    print(f"clustered {len(records)} decks into {len(clustering_result.clusters)} clusters")
    print(f"last merge distance: {clustering_result.last_merge_distance}")
    print(f"next merge distance: {clustering_result.next_merge_distance}")
    print(f"wrote clustered JSONL to {output_path}")
    print(f"wrote evaluation JSON to {evaluation_json}")
    print(f"wrote evaluation Markdown to {evaluation_markdown}")
    print(f"wrote cluster major decks to {major_decks_dir}")
    if not args.no_dendrogram:
        print(f"wrote dendrogram SVG to {dendrogram_svg}")
        print(f"wrote dendrogram PNG to {dendrogram_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
