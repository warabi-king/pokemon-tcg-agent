"""固定した評価リーグでデッキ候補を比較し、再開可能な順位表を作る。

候補間で変えるのは ``--candidate-deck`` だけである。自分側の agent、model、
探索設定は固定し、相手側は ``--opponents`` JSON であらかじめ凍結する。各組合せの
実対戦は ``tools/evaluate_fixed_league.py`` に委譲する。

実行例:
    python tools/search_deck_fixed_league.py \\
        --agent-src agents/belief_puct/src --model agents/belief_puct/src/model.pth \\
        --candidate-deck agents/belief_puct/train/runs/trial/candidates/candidate_000/deck.csv \\
        --opponents configs/fixed_league_opponents.json --games-per-opponent 40 \\
        --output-dir results/deck_search_trial --resume

相手JSONは ``{\"opponents\": [{\"name\": \"baseline\", \"agent_src\": \"...\",
\"deck\": \"...\", \"model\": \"...\", \"weight\": 1.0}]}`` の形式である。\n
``model`` は省略でき、各 agent の既定checkpointを用いる。パスはJSONファイルからの
相対パスとして解決する。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "evaluate_fixed_league.py"


@dataclass(frozen=True)
class Opponent:
    """固定リーグに含める相手の実行条件を表す。"""

    name: str
    agent_src: Path
    deck: Path
    model: Path | None
    weight: float


def write_json(path: Path, content: Any) -> None:
    """JSON成果物をUTF-8で保存し、親ディレクトリを作成する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_config_path(raw_path: str, config_path: Path) -> Path:
    """JSON内の相対パスを設定ファイルの親から解決する。"""

    path = Path(raw_path)
    return (path if path.is_absolute() else config_path.parent / path).resolve()


def load_opponents(config_path: Path) -> list[Opponent]:
    """相手リーグJSONを読み、重複名・必須項目・重みを検証する。"""

    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"相手リーグJSONを解釈できません: {config_path}: {error}") from error
    raw_opponents = content.get("opponents") if isinstance(content, dict) else None
    if not isinstance(raw_opponents, list) or not raw_opponents:
        raise ValueError("相手リーグJSONには空でない opponents 配列が必要です")

    opponents: list[Opponent] = []
    names: set[str] = set()
    for index, raw_opponent in enumerate(raw_opponents):
        if not isinstance(raw_opponent, dict):
            raise ValueError(f"opponents[{index}] はobjectである必要があります")
        try:
            name = str(raw_opponent["name"])
            agent_src = resolve_config_path(str(raw_opponent["agent_src"]), config_path)
            deck = resolve_config_path(str(raw_opponent["deck"]), config_path)
        except KeyError as error:
            raise ValueError(f"opponents[{index}] に必須項目 {error.args[0]} がありません") from error
        if not name or name in names:
            raise ValueError(f"相手名は空または重複できません: {name!r}")
        weight = float(raw_opponent.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"opponents[{index}].weight は正の有限値である必要があります")
        raw_model = raw_opponent.get("model")
        model = resolve_config_path(str(raw_model), config_path) if raw_model else None
        for label, path in (("agent_src", agent_src), ("deck", deck), ("model", model)):
            if path is not None and not path.exists():
                raise FileNotFoundError(f"opponents[{index}] の {label} がありません: {path}")
        names.add(name)
        opponents.append(Opponent(name, agent_src, deck, model, weight))
    return opponents


def collect_candidate_decks(explicit_decks: list[Path], candidate_directories: list[Path]) -> list[Path]:
    """明示指定とrun配下の候補ディレクトリから重複しないdeck.csv群を集める。"""

    decks = [deck.resolve() for deck in explicit_decks]
    for directory in candidate_directories:
        resolved_directory = directory.resolve()
        if not resolved_directory.is_dir():
            raise FileNotFoundError(f"候補ディレクトリがありません: {resolved_directory}")
        decks.extend(sorted(resolved_directory.glob("candidate_*/deck.csv")))
    unique_decks: list[Path] = []
    seen_decks: set[Path] = set()
    for deck in decks:
        resolved_deck = deck.resolve()
        if resolved_deck not in seen_decks:
            unique_decks.append(resolved_deck)
            seen_decks.add(resolved_deck)
    if not unique_decks:
        raise ValueError("--candidate-deck または --candidate-dir から候補deck.csvを1個以上指定してください")
    return unique_decks


def wilson_lower(successes: float, trials: int) -> float:
    """draw=0.5勝として集計した95% Wilson下限を返す。"""

    if trials <= 0:
        return 0.0
    z_score = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z_score**2 / trials
    center = (probability + z_score**2 / (2.0 * trials)) / denominator
    margin = z_score * math.sqrt(
        probability * (1.0 - probability) / trials + z_score**2 / (4.0 * trials**2)
    ) / denominator
    return max(0.0, center - margin)


def pairing_path(output_dir: Path, candidate_index: int, opponent_name: str) -> Path:
    """候補と相手の組に一意な評価JSONパスを割り当てる。"""

    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in opponent_name)
    return output_dir / "pairings" / f"candidate_{candidate_index:03d}__{safe_name}.json"


def read_completed_summary(report_path: Path, games_per_opponent: int, allow_errors: bool) -> dict[str, Any] | None:
    """再開可能な評価JSONを返す。許可時は未判定試合を含むreportも使う。"""

    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        summary = report["summary"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    games_completed = summary.get("games_completed")
    errors = summary.get("errors")
    if not isinstance(games_completed, int) or not isinstance(errors, int):
        return None
    if games_completed + errors != games_per_opponent:
        return None
    if not allow_errors and (games_completed != games_per_opponent or errors != 0):
        return None
    return summary


def aggregate_candidate(candidate_index: int, candidate_deck: Path, opponents: list[Opponent], summaries: list[dict[str, Any]], errors_as_losses: bool = False) -> dict[str, Any]:
    """候補の相手別成績から平均、最苦手相手、Wilson下限を計算する。"""

    if len(opponents) != len(summaries):
        raise ValueError("相手数と評価summary数が一致しません")
    total_weight = sum(opponent.weight for opponent in opponents)
    def score(summary: dict[str, Any]) -> float:
        """短時間評価では未判定試合も候補側の敗戦としてscoreへ含める。"""

        if not errors_as_losses:
            return float(summary["score_a"])
        games_requested = int(summary["games_requested"])
        successes = int(summary["wins_a"]) + 0.5 * int(summary["draws"])
        return successes / games_requested if games_requested else 0.0

    opponent_scores = [score(summary) for summary in summaries]
    weighted_score = sum(opponent_score * opponent.weight for opponent, opponent_score in zip(opponents, opponent_scores)) / total_weight
    weakest_index = min(range(len(opponents)), key=lambda index: opponent_scores[index])
    successes = sum(float(summary["wins_a"]) + 0.5 * float(summary["draws"]) for summary in summaries)
    trials = sum(int(summary["games_requested"] if errors_as_losses else summary["games_completed"]) for summary in summaries)
    per_opponent = [
        {
            "name": opponent.name,
            "weight": opponent.weight,
            "score": float(summary["score_a"]),
            "wins": int(summary["wins_a"]),
            "losses": int(summary["wins_b"]),
            "draws": int(summary["draws"]),
            "games": int(summary["games_completed"]),
            "errors": int(summary.get("errors", 0)),
        }
        for opponent, summary in zip(opponents, summaries)
    ]
    return {
        "candidate_index": candidate_index,
        "deck": str(candidate_deck),
        "weighted_score": weighted_score,
        "worst_opponent_score": opponent_scores[weakest_index],
        "weakest_opponent": opponents[weakest_index].name,
        "wilson_lower_95": wilson_lower(successes, trials),
        "games_scored": trials,
        "per_opponent": per_opponent,
    }


def ranking_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    """広い相手への平均を主、最苦手相手と不確実性を副基準に順位付けする。"""

    return (
        -float(candidate["weighted_score"]),
        -float(candidate["worst_opponent_score"]),
        -float(candidate["wilson_lower_95"]),
        int(candidate["candidate_index"]),
    )


def build_evaluator_command(args: argparse.Namespace, candidate_deck: Path, opponent: Opponent, output_path: Path, seed_start: int) -> list[str]:
    """既存の隔離評価器を呼び出す引数列を構築する。"""

    command = [
        sys.executable, str(EVALUATOR),
        "--agent-a", str(args.agent_src),
        "--agent-b", str(opponent.agent_src),
        "--deck-a", str(candidate_deck),
        "--deck-b", str(opponent.deck),
        "--name-a", args.name_a,
        "--name-b", opponent.name,
        "--games", str(args.games_per_opponent),
        "--seed-start", str(seed_start),
        "--action-timeout-seconds", str(args.action_timeout_seconds),
        "--game-timeout-seconds", str(args.game_timeout_seconds),
        "--output", str(output_path),
    ]
    if args.model is not None:
        command.extend(["--model-a", str(args.model)])
    if opponent.model is not None:
        command.extend(["--model-b", str(opponent.model)])
    if args.determinizations is not None:
        command.extend(["--determinizations-a", str(args.determinizations)])
    if args.search_count is not None:
        command.extend(["--search-count-a", str(args.search_count)])
    return command


def parse_args() -> argparse.Namespace:
    """固定評価条件、候補、相手リーグ、再開オプションを受け取る。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-src", type=Path, required=True, help="候補側の固定agent src")
    parser.add_argument("--model", type=Path, default=None, help="候補側の固定checkpoint")
    parser.add_argument("--candidate-deck", type=Path, action="append", default=[], help="比較する60枚deck.csv。複数指定可")
    parser.add_argument("--candidate-dir", type=Path, action="append", default=[], help="run内の candidates など、candidate_*/deck.csvを含むディレクトリ。複数指定可")
    parser.add_argument("--opponents", type=Path, required=True, help="固定相手リーグJSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-a", default="deck_candidate")
    parser.add_argument("--games-per-opponent", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=51000)
    parser.add_argument("--action-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--game-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--determinizations", type=int, default=None)
    parser.add_argument("--search-count", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="正常終了済みの組合せを再実行しない")
    parser.add_argument("--errors-as-losses", action="store_true", help="timeout等の未判定試合を候補側の敗戦として採点し、完走reportを再利用する")
    parser.add_argument("--dry-run", action="store_true", help="予定のみ表示し、ファイルを作成しない")
    return parser.parse_args()


def main() -> None:
    """候補×固定相手リーグを実行し、順位表を保存する。"""

    args = parse_args()
    args.agent_src = args.agent_src.resolve()
    args.model = args.model.resolve() if args.model else None
    try:
        candidate_decks = collect_candidate_decks(args.candidate_deck, args.candidate_dir)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    opponents_path = args.opponents.resolve()
    output_dir = args.output_dir.resolve()
    if args.games_per_opponent <= 0 or args.games_per_opponent % 2:
        raise SystemExit("--games-per-opponent は先後均衡のため正の偶数で指定してください")
    for label, path in (("agent-src", args.agent_src), ("model", args.model)):
        if path is not None and not path.exists():
            raise SystemExit(f"--{label} がありません: {path}")
    for deck in candidate_decks:
        if not deck.is_file():
            raise SystemExit(f"候補deck.csvがありません: {deck}")
    opponents = load_opponents(opponents_path)

    planned = [
        {"candidate_index": candidate_index, "deck": str(deck), "opponent": opponent.name,
         "output": str(pairing_path(output_dir, candidate_index, opponent.name))}
        for candidate_index, deck in enumerate(candidate_decks)
        for opponent in opponents
    ]
    if args.dry_run:
        print(json.dumps({"format": "pokemon-tcg-agent/deck-search-fixed-league-plan-v1", "pairings": planned}, ensure_ascii=False, indent=2))
        return

    candidate_results: list[dict[str, Any]] = []
    for candidate_index, deck in enumerate(candidate_decks):
        summaries: list[dict[str, Any]] = []
        for opponent_index, opponent in enumerate(opponents):
            output_path = pairing_path(output_dir, candidate_index, opponent.name)
            summary = read_completed_summary(output_path, args.games_per_opponent, args.errors_as_losses) if args.resume else None
            if summary is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                seed_start = args.seed_start + candidate_index * 100000 + opponent_index * 1000
                command = build_evaluator_command(args, deck, opponent, output_path, seed_start)
                print("running=" + " ".join(command))
                subprocess.run(command, cwd=ROOT, check=True)
                summary = read_completed_summary(output_path, args.games_per_opponent, args.errors_as_losses)
            if summary is None:
                raise RuntimeError(f"評価が採点可能な試合数に達していません: {output_path}")
            summaries.append(summary)
        candidate_results.append(aggregate_candidate(candidate_index, deck, opponents, summaries, args.errors_as_losses))

    ranked = sorted(candidate_results, key=ranking_key)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    report = {
        "format": "pokemon-tcg-agent/deck-search-fixed-league-v1",
        "configuration": {
            "agent_src": str(args.agent_src), "model": str(args.model) if args.model else None,
            "candidate_decks": [str(deck) for deck in candidate_decks],
            "opponents_config": str(opponents_path), "games_per_opponent": args.games_per_opponent,
            "seed_start": args.seed_start, "seed_policy": "候補・相手組ごとに固定し、評価器内で先後を均衡化",
            "ranking_policy": "weighted_score, worst_opponent_score, wilson_lower_95 の降順",
            "errors_as_losses": args.errors_as_losses,
        },
        "ranking": ranked,
    }
    write_json(output_dir / "ranking.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
