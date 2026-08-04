"""外部worktreeの候補群とcodex-autoのagentを固定条件で総当たり評価する。

各カードは ``evaluate_fixed_league.py`` を子プロセスで実行する。これにより、
同名の ``cg`` / ``rl_mcts`` packageを含む別worktreeのagentどうしも、Pythonの
module cacheを共有せずに評価できる。各カードのJSONを残すため、中断後は
``--resume`` で完了済みカードを再利用できる。

実行例:
    python tools/run_cross_worktree_league.py \
        --agent /path/to/belief_puct/src \
        --agent /path/to/cluster_00/src \
        --agent /path/to/cluster_01/src \
        --games-per-pairing 20 \
        --output-dir results/upsize1_vs_belief_20260804 \
        --resume

``--agent`` には参加させる ``src`` ディレクトリを一つずつ指定する。
各ディレクトリは ``main.py``、``deck.csv``、``model.pth`` を持つことを想定する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from evaluate_fixed_league import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "evaluate_fixed_league.py"


@dataclass(frozen=True)
class Candidate:
    """総当たりへ登録するagentの実行に必要なファイル群。"""

    name: str
    src: Path
    deck: Path
    model: Path


def validate_deck(deck_path: Path) -> None:
    """deck.csvの60枚制約を開始前に検証する。"""

    cards = [line for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck.csvが60枚ではありません: {deck_path} ({len(cards)}枚)")


def require_file(path: Path) -> Path:
    """必須ファイルの存在を検証し、解決済みpathを返す。"""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"必須ファイルがありません: {resolved}")
    return resolved


def load_candidates(agent_src_paths: list[Path]) -> list[Candidate]:
    """指定されたsrcディレクトリを、そのまま総当たり候補として読み込む。"""

    if len(agent_src_paths) < 2:
        raise ValueError("--agent は対戦のため2個以上指定してください")

    candidates = []
    seen_names: set[str] = set()
    for raw_src in agent_src_paths:
        src = raw_src.resolve()
        name = src.parent.name
        if name in seen_names:
            raise ValueError(
                f"参加者名が重複しています: {name}。"
                "srcの親ディレクトリ名が一意になるよう指定してください"
            )
        candidate = Candidate(
            name=name,
            src=src,
            deck=require_file(src / "deck.csv"),
            model=require_file(src / "model.pth"),
        )
        require_file(src / "main.py")
        validate_deck(candidate.deck)
        candidates.append(candidate)
        seen_names.add(name)
    return candidates


def report_path(output_dir: Path, candidate_a: Candidate, candidate_b: Candidate) -> Path:
    """候補名から衝突しない対戦カードのJSON保存先を返す。"""

    return output_dir / "pairings" / f"{candidate_a.name}_vs_{candidate_b.name}.json"


def is_completed_report(path: Path, expected_games: int) -> bool:
    """再開時に再利用できる、完走済みの同条件reportかを判定する。"""

    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report["summary"]
        return (
            report["format"] == "pokemon-tcg-agent/fixed-league-evaluation-v1"
            and int(summary["games_requested"]) == expected_games
            and int(summary["games_completed"]) == expected_games
            and int(summary["errors"]) == 0
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_pairing(
    candidate_a: Candidate,
    candidate_b: Candidate,
    games: int,
    seed_start: int,
    output_path: Path,
    action_timeout_seconds: float,
    game_timeout_seconds: float,
) -> None:
    """既存の隔離評価器へ1対戦カードを渡し、結果JSONを出力する。"""

    command = [
        sys.executable,
        str(EVALUATOR),
        "--agent-a",
        str(candidate_a.src),
        "--deck-a",
        str(candidate_a.deck),
        "--model-a",
        str(candidate_a.model),
        "--name-a",
        candidate_a.name,
        "--agent-b",
        str(candidate_b.src),
        "--deck-b",
        str(candidate_b.deck),
        "--model-b",
        str(candidate_b.model),
        "--name-b",
        candidate_b.name,
        "--games",
        str(games),
        "--seed-start",
        str(seed_start),
        "--action-timeout-seconds",
        str(action_timeout_seconds),
        "--game-timeout-seconds",
        str(game_timeout_seconds),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, cwd=ROOT)


def aggregate_ranking(candidates: list[Candidate], output_dir: Path) -> dict[str, Any]:
    """カード別JSONから候補ごとの勝敗、score、Wilson区間を再集計する。"""

    records = {
        candidate.name: {"wins": 0, "losses": 0, "draws": 0, "errors": 0}
        for candidate in candidates
    }
    pairing_reports: list[str] = []
    for index, candidate_a in enumerate(candidates):
        for candidate_b in candidates[index + 1 :]:
            path = report_path(output_dir, candidate_a, candidate_b)
            if not path.is_file():
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            summary = report["summary"]
            records[candidate_a.name]["wins"] += int(summary["wins_a"])
            records[candidate_a.name]["losses"] += int(summary["wins_b"])
            records[candidate_a.name]["draws"] += int(summary["draws"])
            records[candidate_a.name]["errors"] += int(summary["errors"])
            records[candidate_b.name]["wins"] += int(summary["wins_b"])
            records[candidate_b.name]["losses"] += int(summary["wins_a"])
            records[candidate_b.name]["draws"] += int(summary["draws"])
            records[candidate_b.name]["errors"] += int(summary["errors"])
            pairing_reports.append(str(path.relative_to(output_dir)))

    ranking = []
    for name, record in records.items():
        completed = record["wins"] + record["losses"] + record["draws"]
        successes = record["wins"] + 0.5 * record["draws"]
        score = successes / completed if completed else 0.0
        lower, upper = wilson_interval(successes, completed)
        ranking.append(
            {
                "name": name,
                **record,
                "games_completed": completed,
                "score": score,
                "win_rate": record["wins"] / completed if completed else 0.0,
                "wilson_lower_95": lower,
                "wilson_upper_95": upper,
            }
        )
    ranking.sort(key=lambda record: (-record["score"], -record["wilson_lower_95"], record["name"]))
    return {
        "format": "pokemon-tcg-agent/cross-worktree-league-v1",
        "candidates": [candidate.name for candidate in candidates],
        "pairing_reports": pairing_reports,
        "ranking": ranking,
    }


def parse_args() -> argparse.Namespace:
    """参加src、試合条件、出力と再開方法を受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        type=Path,
        action="append",
        required=True,
        metavar="AGENT_SRC",
        help="参加させるagentのsrcディレクトリ。参加者ごとに繰り返し指定する",
    )
    parser.add_argument("--games-per-pairing", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=81000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    # 既存の固定リーグ評価器と同じ上限にし、初回の重み読込やPUCT探索を
    # 総当たりツールだけが短いtimeoutで失敗させない。
    parser.add_argument("--action-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--game-timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    """全136対戦カードを順番に実行し、途中経過を含む順位表を保存する。"""

    args = parse_args()
    if args.games_per_pairing < 2 or args.games_per_pairing % 2:
        raise SystemExit("--games-per-pairing は先後を均等にするため2以上の偶数にしてください")
    candidates = load_candidates(args.agent)
    total_pairings = len(candidates) * (len(candidates) - 1) // 2
    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    pairing_index = 0
    for index, candidate_a in enumerate(candidates):
        for candidate_b in candidates[index + 1 :]:
            pairing_index += 1
            output_path = report_path(output_dir, candidate_a, candidate_b)
            if args.resume and is_completed_report(output_path, args.games_per_pairing):
                completed += 1
                print(f"[{pairing_index}/{total_pairings}] reuse {candidate_a.name} vs {candidate_b.name}")
                continue
            print(f"[{pairing_index}/{total_pairings}] run {candidate_a.name} vs {candidate_b.name}", flush=True)
            if args.dry_run:
                continue
            # カードごとにseed範囲を分離し、再開しても既存カードの条件を変えない。
            pairing_seed = args.seed_start + (pairing_index - 1) * (args.games_per_pairing // 2)
            run_pairing(
                candidate_a,
                candidate_b,
                args.games_per_pairing,
                pairing_seed,
                output_path,
                args.action_timeout_seconds,
                args.game_timeout_seconds,
            )
            completed += 1
            aggregate = aggregate_ranking(candidates, output_dir)
            aggregate["configuration"] = {
                "agents": [str(agent_src.resolve()) for agent_src in args.agent],
                "games_per_pairing": args.games_per_pairing,
                "seed_start": args.seed_start,
                "seed_policy": "各カードで固有の連番seedを使用し、同一seedを先後入替の2試合へ使用",
                "pairings_completed": completed,
                "pairings_total": total_pairings,
            }
            (output_dir / "aggregate.json").write_text(
                json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if args.dry_run:
        print(f"candidates={len(candidates)}, pairings={total_pairings}, games={total_pairings * args.games_per_pairing}")
        return
    print(f"completed={completed}/{total_pairings}")
    print(f"saved={output_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()
