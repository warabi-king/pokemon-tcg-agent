"""Kaggle提出アーカイブの構造、import、複数seed自己対戦を検証する。

実行例:
    python tools/validate_submission.py \
        --archive dist/submission_belief_puct.tar.gz \
        --games 6 --output results/submission_validation.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from evaluate_fixed_league import IsolatedAgent, read_deck, run_evaluation

REQUIRED_MEMBERS = {"main.py", "deck.csv"}
REQUIRED_PREFIXES = ("cg/",)
FORBIDDEN_PARTS = {"train", "logs", "__pycache__", ".git"}


def sha256_file(path: Path) -> str:
    """ファイル内容のSHA-256を返す。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_members(archive: tarfile.TarFile) -> list[str]:
    """member名を検査し、path traversalや不要物を拒否する。"""

    names = [member.name for member in archive.getmembers()]
    missing = sorted(REQUIRED_MEMBERS - set(names))
    if missing:
        raise ValueError("必須ファイルがありません: " + ", ".join(missing))
    for prefix in REQUIRED_PREFIXES:
        if not any(name.startswith(prefix) for name in names):
            raise ValueError(f"必須ディレクトリがありません: {prefix}")
    for name in names:
        member_path = Path(name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"危険なアーカイブパスです: {name}")
        if FORBIDDEN_PARTS.intersection(member_path.parts) or name.endswith(".pyc"):
            raise ValueError(f"提出不要物が含まれています: {name}")
    return names


def validate_archive(
    archive_path: Path,
    games: int,
    seed_start: int,
    action_timeout_seconds: float,
    game_timeout_seconds: float,
) -> dict[str, object]:
    """アーカイブを一時展開し、60枚deckと隔離自己対戦を検証する。"""

    with tempfile.TemporaryDirectory(prefix="belief-puct-submission-") as temp_raw:
        extracted = Path(temp_raw) / "kaggle_simulations" / "agent"
        extracted.mkdir(parents=True)
        with tarfile.open(archive_path, "r:gz") as archive:
            names = inspect_members(archive)
            archive.extractall(extracted, filter="data")

        deck = read_deck(extracted / "deck.csv")
        proxy_a = IsolatedAgent(
            extracted,
            deck,
            "submission_validation_a",
            action_timeout_seconds,
        )
        proxy_b = IsolatedAgent(
            extracted,
            deck,
            "submission_validation_b",
            action_timeout_seconds,
        )
        try:
            summary, game_results = run_evaluation(
                proxy_a,
                proxy_b,
                "submission_copy_a",
                "submission_copy_b",
                games,
                seed_start,
                game_timeout_seconds,
            )
        finally:
            proxy_a.close()
            proxy_b.close()

    if summary.errors:
        raise RuntimeError(f"自己対戦で{summary.errors}試合が失敗しました")
    return {
        "format": "pokemon-tcg-agent/submission-validation-v1",
        "archive": str(archive_path.resolve()),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "member_count": len(names),
        "members": names,
        "deck_cards": len(deck),
        "execution_path": "/kaggle_simulations/agent/を模した一時ディレクトリ",
        "self_play": {
            "seed_start": seed_start,
            "summary": asdict(summary),
            "games": [asdict(game) for game in game_results],
        },
    }


def parse_args() -> argparse.Namespace:
    """アーカイブ、自己対戦条件、出力先を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=99000)
    parser.add_argument("--action-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--game-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """提出物検証を実行し、機械可読な証拠を保存する。"""

    args = parse_args()
    report = validate_archive(
        args.archive.resolve(),
        args.games,
        args.seed_start,
        args.action_timeout_seconds,
        args.game_timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "members"}, ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
