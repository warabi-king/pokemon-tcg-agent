"""異なるエージェント同士でローカル対戦を複数回実行し、勝率などの統計を出す。

`tools/run_local_match.py` は main.py のエージェント同士（同一実装）を1回だけ
対戦させるものですが、こちらは

- 任意の2つのエージェントファイル（`agent(obs_dict)` を定義したPythonファイル）
- 任意の対戦回数
- （任意で）先手/後手を1試合ごとに入れ替え
- （任意で）エージェントごとに別のdeck.csvを使用

を指定して、勝率・引き分け率・エラー数を集計します。

使用例:
    # main.py（現行の提出物）と agent_v1.py を20回対戦させる
    python3 tools/run_matches.py  --agent1 src/main.py --agent2 src/main.py --games 20

    # 名前を指定しつつ、先手/後手を固定して50回対戦
    python tools/run_matches.py \\
        --agent1 src/main.py --name1 baseline \\
        --agent2 src/agent_v1.py --name2 mcts \\
        --games 50 --no-alternate

    # エージェントごとに別デッキを使わせる
    python tools/run_matches.py \\
        --agent1 src/main.py --deck1 src/deck.csv \\
        --agent2 src/agent_v1.py --deck2 src/deck_v2.csv \\
        --games 30

    # 詳細ログをresults/にJSONで保存する
    python tools/run_matches.py --agent1 src/main.py --agent2 src/agent_v1.py \\
        --games 20 --save-json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
RESULTS_ROOT = ROOT / "results"

AgentFunc = Callable[[dict], list]


def _load_module(path: Path, module_name: str) -> ModuleType:
    """任意のパスのPythonファイルを、指定した名前のモジュールとして読み込む。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} を読み込めませんでした。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_deck(path: Path) -> list[int]:
    """deck.csv（カードIDを1行1枚で60行）を読み込む。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} が存在しません。")
    deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(
            f"deck.csvはカードIDを60枚分だけ含める必要があります。"
            f"現在: {len(deck)}枚 ({path})"
        )
    return deck


def load_agent(path: Path, module_name: str, deck: list[int]) -> AgentFunc:
    """指定ファイルから agent(obs_dict) を読み込み、使用デッキを固定する。

    main.py 互換の実装（`read_deck_csv()` を持つ）であれば、それを差し替えて
    常に `deck` を返すようにする。これにより env に渡すデッキと、
    エージェントが対戦開始時に申告するデッキを一致させる。
    """
    module = _load_module(path, module_name)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} に agent(obs_dict) が定義されていません。")

    if hasattr(module, "read_deck_csv"):
        module.read_deck_csv = lambda: list(deck)  # type: ignore[assignment]
    else:
        print(
            f"警告: {path} に read_deck_csv が見つからないため、"
            f"このエージェントが対戦開始時に返すデッキは指定した --deckN と"
            f"一致しない可能性があります。",
            file=sys.stderr,
        )

    return module.agent


@dataclass
class Stats:
    name0: str
    name1: str
    wins: dict[str, int] = field(default_factory=dict)
    draws: int = 0
    unresolved: int = 0
    total: int = 0
    game_log: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wins.setdefault(self.name0, 0)
        self.wins.setdefault(self.name1, 0)

    def record(
        self,
        game_index: int,
        first_player_name: str,
        result: Optional[int],
        turns: int,
    ) -> None:
        """result は常に (name0, name1) の勝敗を表す 0/1/2/None で渡す。"""
        self.total += 1
        winner_name = None
        if result == 0:
            self.wins[self.name0] += 1
            winner_name = self.name0
        elif result == 1:
            self.wins[self.name1] += 1
            winner_name = self.name1
        elif result == 2:
            self.draws += 1
        else:
            self.unresolved += 1

        self.game_log.append(
            {
                "game": game_index,
                "first_player": first_player_name,
                "winner": winner_name,
                "turns": turns,
            }
        )

    def summary(self) -> str:
        lines = [f"総試合数: {self.total}"]
        for name in (self.name0, self.name1):
            wins = self.wins.get(name, 0)
            pct = (wins / self.total * 100) if self.total else 0.0
            lines.append(f"  {name} の勝利: {wins} ({pct:.1f}%)")
        draw_pct = (self.draws / self.total * 100) if self.total else 0.0
        lines.append(f"  引き分け: {self.draws} ({draw_pct:.1f}%)")
        if self.unresolved:
            lines.append(f"  未判定/エラー: {self.unresolved}")
        return "\n".join(lines)


def extract_result(steps: list) -> tuple[Optional[int], int]:
    """kaggle_environmentsのstepsから (result, turns) を推定する。

    result: 0 なら env.run に渡した1体目の勝ち、1 なら2体目の勝ち、
    2 なら引き分け、None なら判定不能（エラー等）。
    """
    if not steps:
        return None, 0

    final_step = steps[-1]
    turns = len(steps)

    if len(final_step) != 2:
        return None, turns

    rewards = [player_state.get("reward") for player_state in final_step]
    r0, r1 = rewards

    if r0 == 1 and r1 == -1:
        return 0, turns
    if r1 == 1 and r0 == -1:
        return 1, turns
    if r0 == r1 and r0 is not None:
        return 2, turns
    return None, turns


def run_matches(
    agent0: AgentFunc,
    agent1: AgentFunc,
    name0: str,
    name1: str,
    num_games: int,
    alternate_sides: bool = True,
    debug: bool = False,
    verbose: bool = True,
) -> Stats:
    stats = Stats(name0=name0, name1=name1)

    for i in range(num_games):
        # swap=True の試合では、cabt環境上の player0/player1 を入れ替えて
        # (name1側が先手) 実行する。集計は常に name0/name1 基準に揃え直す。
        swap = alternate_sides and (i % 2 == 1)
        agents = [agent1, agent0] if swap else [agent0, agent1]
        first_player_name = name1 if swap else name0

        env = make("cabt", debug=debug)
        try:
            env.run(agents)
            raw_result, turns = extract_result(env.steps)
        except Exception as exc:  # noqa: BLE001 - 1試合の異常終了で全体を止めない
            raw_result, turns = None, 0
            if verbose:
                print(f"[{i + 1}/{num_games}] エラー: {exc}", file=sys.stderr)

        # raw_result は「env.runに渡した順番」基準なので、swap時はname0/name1基準に戻す
        if raw_result in (0, 1):
            result = (1 - raw_result) if swap else raw_result
        else:
            result = raw_result

        stats.record(i + 1, first_player_name, result, turns)

        if verbose:
            if result == 0:
                outcome = f"{name0} の勝ち"
            elif result == 1:
                outcome = f"{name1} の勝ち"
            elif result == 2:
                outcome = "引き分け"
            else:
                outcome = "不明（エラー）"
            print(f"[{i + 1}/{num_games}] {outcome}  (先手={first_player_name}, turns={turns})")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent1", type=Path, default=SRC_ROOT / "main.py", help="1体目のエージェントファイル"
    )
    parser.add_argument(
        "--agent2", type=Path, default=SRC_ROOT / "main.py", help="2体目のエージェントファイル"
    )
    parser.add_argument("--name1", type=str, default=None, help="1体目の表示名（省略時はファイル名）")
    parser.add_argument("--name2", type=str, default=None, help="2体目の表示名（省略時はファイル名）")
    parser.add_argument(
        "--deck1",
        type=Path,
        default=SRC_ROOT / "deck.csv",
        help="1体目に使わせるdeck.csv（省略時は src/deck.csv）",
    )
    parser.add_argument(
        "--deck2",
        type=Path,
        default=SRC_ROOT / "deck.csv",
        help="2体目に使わせるdeck.csv（省略時は src/deck.csv）",
    )
    parser.add_argument("--games", type=int, default=10, help="対戦回数（デフォルト: 10）")
    parser.add_argument(
        "--no-alternate",
        action="store_true",
        help="先手/後手を固定する（デフォルトは1試合ごとに入れ替えて先手有利を均す）",
    )
    parser.add_argument("--debug", action="store_true", help="kaggle_environmentsのdebugログを出す")
    parser.add_argument("--quiet", action="store_true", help="各試合ごとの結果表示を省略する")
    parser.add_argument(
        "--save-json", action="store_true", help="results/に対戦ごとの詳細ログをJSONで保存する"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    name1 = args.name1 or args.agent1.stem
    name2 = args.name2 or args.agent2.stem
    if name1 == name2:
        name1 += "#1"
        name2 += "#2"

    agent1_path = args.agent1.resolve()
    agent2_path = args.agent2.resolve()
    deck1_path = args.deck1.resolve()
    deck2_path = args.deck2.resolve()

    # cg パッケージ（cg.api など）をどのエージェントからも import できるようにする
    sys.path.insert(0, str(SRC_ROOT))
    os.chdir(SRC_ROOT)

    deck1 = load_deck(deck1_path)
    deck2 = load_deck(deck2_path)

    agent1 = load_agent(agent1_path, "agent_module_1", deck1)
    agent2 = load_agent(agent2_path, "agent_module_2", deck2)

    print(f"{name1}: {agent1_path} (deck: {deck1_path})")
    print(f"{name2}: {agent2_path} (deck: {deck2_path})")
    print(f"対戦回数: {args.games}  先手/後手入れ替え: {not args.no_alternate}")
    print()

    started = time.time()
    stats = run_matches(
        agent0=agent1,
        agent1=agent2,
        name0=name1,
        name1=name2,
        num_games=args.games,
        alternate_sides=not args.no_alternate,
        debug=args.debug,
        verbose=not args.quiet,
    )
    elapsed = time.time() - started

    print()
    print(stats.summary())
    print(f"\n所要時間: {elapsed:.1f}秒 ({elapsed / max(stats.total, 1):.2f}秒/試合)")

    if args.save_json:
        RESULTS_ROOT.mkdir(exist_ok=True)
        out_path = RESULTS_ROOT / f"matches_{int(time.time())}.json"
        out_path.write_text(
            json.dumps(
                {
                    "name0": stats.name0,
                    "name1": stats.name1,
                    "agent1_path": str(agent1_path),
                    "agent2_path": str(agent2_path),
                    "deck1_path": str(deck1_path),
                    "deck2_path": str(deck2_path),
                    "total": stats.total,
                    "wins": stats.wins,
                    "draws": stats.draws,
                    "unresolved": stats.unresolved,
                    "elapsed_seconds": elapsed,
                    "games": stats.game_log,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"詳細ログを保存しました: {out_path}")


if __name__ == "__main__":
    main()
