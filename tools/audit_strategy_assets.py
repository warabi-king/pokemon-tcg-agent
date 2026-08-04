"""公式対戦ZIPとデッキ候補DBを軽量監査する。

実行例:
    python tools/audit_strategy_assets.py \
        --episodes-root ../develop_nomura/episodes/official \
        --deck-database ../develop_nomura/agents/rl_mcts/src/rl_mcts/deck_candidates_by_wins.jsonl \
        --output work/audits/strategy-assets.json

ZIP内の巨大な対戦JSONは展開せず、中央ディレクトリとmanifest.csvだけを読む。
入力はread-onlyとして扱い、結果だけを指定したJSONへ保存する。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class DailyArchiveAudit:
    """日次ZIP一つの構造とmanifest統計を保持する。"""

    date: str
    path: str
    episode_json_files: int
    manifest_rows: int
    compressed_bytes: int
    uncompressed_bytes: int
    mean_average_rating: float | None
    min_average_rating: float | None
    max_average_rating: float | None
    error: str | None


def wilson_interval(successes: float, trials: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    """成功数と試行数から二項比率のWilson区間を返す。

    drawを0.5成功として扱う用途を許すため、successesは小数を受け取る。
    trialsが0以下の場合は比較不能なので(0, 1)を返す。
    """

    if trials <= 0:
        return 0.0, 1.0
    probability = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (probability + z_score**2 / (2.0 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z_score**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _float_or_none(raw_value: str | None) -> float | None:
    """空文字を許容しつつ文字列をfloatへ変換する。"""

    if raw_value is None or raw_value == "":
        return None
    return float(raw_value)


def audit_daily_archive(archive_path: Path) -> DailyArchiveAudit:
    """日次ZIPを展開せず、ファイル数、容量、manifestのratingを監査する。"""

    archive_date = archive_path.stem
    try:
        with ZipFile(archive_path) as archive:
            members = archive.infolist()
            json_members = [member for member in members if member.filename.endswith(".json")]
            manifest_members = [member for member in members if member.filename.endswith("manifest.csv")]
            if len(manifest_members) != 1:
                raise ValueError(f"manifest.csvが1個ではありません: {len(manifest_members)}")
            manifest_text = archive.read(manifest_members[0]).decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(manifest_text)))
            ratings = [
                rating
                for row in rows
                if (rating := _float_or_none(row.get("avg_score"))) is not None
            ]
            return DailyArchiveAudit(
                date=archive_date,
                path=str(archive_path),
                episode_json_files=len(json_members),
                manifest_rows=len(rows),
                compressed_bytes=sum(member.compress_size for member in members),
                uncompressed_bytes=sum(member.file_size for member in members),
                mean_average_rating=fmean(ratings) if ratings else None,
                min_average_rating=min(ratings) if ratings else None,
                max_average_rating=max(ratings) if ratings else None,
                error=None,
            )
    except (BadZipFile, OSError, UnicodeError, ValueError, csv.Error) as error:
        return DailyArchiveAudit(
            date=archive_date,
            path=str(archive_path),
            episode_json_files=0,
            manifest_rows=0,
            compressed_bytes=0,
            uncompressed_bytes=0,
            mean_average_rating=None,
            min_average_rating=None,
            max_average_rating=None,
            error=f"{type(error).__name__}: {error}",
        )


def audit_archives(episodes_root: Path) -> list[DailyArchiveAudit]:
    """episodes_root配下の完成済みZIPを日付順に監査する。"""

    return [audit_daily_archive(path) for path in sorted(episodes_root.rglob("*.zip"))]


def _summarize_deck_record(record: dict[str, object], database_index: int) -> dict[str, object]:
    """デッキDB一行を比較に必要な小さな辞書へ正規化する。"""

    wins = int(record.get("wins", 0))
    losses = int(record.get("losses", 0))
    draws = int(record.get("draws", 0))
    games = int(record.get("games", wins + losses + draws))
    score = (wins + 0.5 * draws) / games if games else 0.0
    lower, upper = wilson_interval(wins + 0.5 * draws, games)
    deck = [int(card_id) for card_id in record.get("deck", [])]
    return {
        "database_index": database_index,
        "deck": deck,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "cluster_id": int(record.get("cluster_id", -1)),
        "cluster_members": int(record.get("cluster_members", 0)),
        "cluster_games": int(record.get("cluster_games", 0)),
        "cluster_score": float(record.get("cluster_win_rate", 0.0)),
        "first_seen_date": str((record.get("first_seen") or {}).get("date", "")),
    }


def audit_deck_database(database_path: Path) -> dict[str, object]:
    """候補DBを読み、信頼下限、クラスタ、頻出カードを集計する。"""

    records: list[dict[str, object]] = []
    cluster_records: dict[int, list[dict[str, object]]] = defaultdict(list)
    card_frequency: Counter[int] = Counter()
    with database_path.open("r", encoding="utf-8") as source:
        for database_index, line in enumerate(source):
            if not line.strip():
                continue
            record = _summarize_deck_record(json.loads(line), database_index)
            if len(record["deck"]) != 60:
                continue
            records.append(record)
            cluster_records[int(record["cluster_id"])].append(record)
            card_frequency.update(set(record["deck"]))

    reliable_records = [record for record in records if int(record["games"]) >= 100]
    reliable_records.sort(
        key=lambda record: (
            float(record["wilson_lower_95"]),
            int(record["games"]),
        ),
        reverse=True,
    )
    cluster_summary = []
    for cluster_id, members in cluster_records.items():
        representative = max(members, key=lambda record: int(record["games"]))
        cluster_summary.append(
            {
                "cluster_id": cluster_id,
                "candidate_decks": len(members),
                "reported_members": int(representative["cluster_members"]),
                "reported_games": int(representative["cluster_games"]),
                "reported_score": float(representative["cluster_score"]),
                "representative_database_index": int(representative["database_index"]),
                "representative_deck": representative["deck"],
            }
        )
    cluster_summary.sort(
        key=lambda cluster: (float(cluster["reported_score"]), int(cluster["reported_games"])),
        reverse=True,
    )
    return {
        "path": str(database_path),
        "valid_decks": len(records),
        "reliable_decks_games_ge_100": len(reliable_records),
        "top_reliable_by_wilson": reliable_records[:20],
        "clusters": cluster_summary,
        "frequent_cards": [
            {"card_id": card_id, "candidate_decks": count}
            for card_id, count in card_frequency.most_common(30)
        ],
    }


def choose_date_split(audits: Iterable[DailyArchiveAudit]) -> dict[str, list[str]]:
    """時系列漏洩を避ける17日/3日/3日の固定splitを返す。"""

    valid_dates = sorted(audit.date for audit in audits if audit.error is None)
    if len(valid_dates) < 6:
        raise ValueError("train/validation/testを分離するには完成済み日次ZIPが6個以上必要です")
    return {
        "train": valid_dates[:-6],
        "validation": valid_dates[-6:-3],
        "test": valid_dates[-3:],
    }


def build_report(episodes_root: Path, deck_database: Path) -> dict[str, object]:
    """データ監査とデッキ監査を一つの再現可能なレポートにまとめる。"""

    archive_audits = audit_archives(episodes_root)
    return {
        "archives": [asdict(audit) for audit in archive_audits],
        "archive_totals": {
            "valid_days": sum(audit.error is None for audit in archive_audits),
            "invalid_days": sum(audit.error is not None for audit in archive_audits),
            "episode_json_files": sum(audit.episode_json_files for audit in archive_audits),
            "manifest_rows": sum(audit.manifest_rows for audit in archive_audits),
            "compressed_bytes": sum(audit.compressed_bytes for audit in archive_audits),
            "uncompressed_bytes": sum(audit.uncompressed_bytes for audit in archive_audits),
        },
        "date_split": choose_date_split(archive_audits),
        "deck_database": audit_deck_database(deck_database),
    }


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--deck-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """監査を実行し、標準出力とJSONファイルへ結果を保存する。"""

    args = parse_args()
    report = build_report(args.episodes_root, args.deck_database)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = report["archive_totals"]
    print(
        f"days={totals['valid_days']} episodes={totals['episode_json_files']} "
        f"invalid_days={totals['invalid_days']} output={args.output}"
    )


if __name__ == "__main__":
    main()
