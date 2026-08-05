"""belief_puctの学習・デッキ選抜・DB拡張をrun単位で管理する。

この初期版は、入力DBから多様な既存60枚候補を選び、全工程の設定と成果物を
``agents/belief_puct/train/runs/<run-id>/`` に分離する。実対戦・学習は既存ツールを
明示的に呼ぶ設計で、``--dry-run`` は入力を変更せず計画とログ構造だけを検証する。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agents" / "belief_puct"
sys.path.insert(0, str(AGENT_ROOT / "train"))

from deck_database import load_records, select_diverse_records, write_records  # noqa: E402


def sha256(path: Path) -> str:
    """ファイルの内容SHA-256を返し、実行時入力を固定する。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """UTF-8 JSONを親ディレクトリごと保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_deck(record: dict[str, Any], output_path: Path) -> None:
    """DBレコードの60枚を評価用deck.csvとして保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(str(card_id) for card_id in record["deck"]) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """run識別子、候補DB、候補数、dry-runを受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, action="append", required=True, help="候補DB JSONL。複数指定可")
    parser.add_argument("--run-id", default=None, help="省略時はUTC時刻から生成する識別子")
    parser.add_argument("--output-root", type=Path, default=AGENT_ROOT / "train" / "runs")
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--min-candidate-games", type=int, default=20, help="候補選抜に必要な最低対戦数")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """候補選抜・run manifest・次工程の実行計画を保存する。"""

    args = parse_args()
    databases = [path.resolve() for path in args.database]
    for path in databases:
        if not path.is_file():
            raise SystemExit(f"候補DBがありません: {path}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root.resolve() / run_id
    if run_dir.exists():
        raise SystemExit(f"run-idが既に存在します: {run_dir}")

    records = load_records(databases)
    selected = select_diverse_records(records, args.candidate_count, args.min_candidate_games)
    manifest = {
        "format": "pokemon-tcg-agent/belief-training-pipeline-v1",
        "run_id": run_id,
        "dry_run": args.dry_run,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in databases],
        "phases": [
            {"name": "imitation_pretrain", "status": "pending"},
            {"name": "deck_search_fixed_league", "status": "pending", "candidate_count": len(selected), "min_candidate_games": args.min_candidate_games},
            {"name": "mixed_opponent_rl", "status": "pending"},
            {"name": "database_extension", "status": "pending"},
            {"name": "holdout_and_submission", "status": "pending"},
        ],
        "candidate_selection": [
            {"index": index, "cluster_id": record.get("cluster_id"), "games": record.get("games"), "deck": record["deck"]}
            for index, record in enumerate(selected)
        ],
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    run_dir.mkdir(parents=True)
    write_json(run_dir / "manifest.json", manifest)
    write_records(run_dir / "database" / "selected_candidates.jsonl", selected)
    for index, record in enumerate(selected):
        copy_deck(record, run_dir / "candidates" / f"candidate_{index:03d}" / "deck.csv")
    shutil.copy2(AGENT_ROOT / "src" / "deck.csv", run_dir / "baseline_deck.csv")
    write_json(run_dir / "logs" / "events.json", [{"event": "run_initialized", "run_id": run_id}])
    print(f"run_dir={run_dir}")
    print(f"candidates={len(selected)}")


if __name__ == "__main__":
    main()
