"""3体以上のエージェント（それぞれ専用デッキ付き）を総当たりで対戦させ、
対戦カードごとの結果と各エージェントの総合勝率を集計する。

デフォルトでは自己対戦（同じエージェント同士の対戦）も対戦カードに含めます。
除外したい場合は `--no-self` を指定してください。

`tools/run_matches.py` は2体だけの対戦でしたが、こちらは
「エージェント名=main.pyのパス[:deck.csvのパス]」を複数指定して、
全ペア（自己対戦含む）で指定回数ずつ対戦させ、リーグ戦のような結果を出します。

使用例:
    # src/main.py（デフォルトデッキ）と、別実装2つを総当たりで各20試合
    # （baseline vs baseline / mcts vs mcts / ... の自己対戦も含む）
    python tools/run_matches_round_robin.py \\
        --agent baseline=src/main.py \\
        --agent mcts=agents/rl_mcts_sample/src/main.py:agents/rl_mcts_sample/src/deck.csv \\
        --agent rule_based=agents/rule_based/src/main.py \\
        --games 20

    # 自己対戦を除外したい場合
    python tools/run_matches_round_robin.py \\
        --agent baseline=src/main.py --agent mcts=agents/rl_mcts_sample/src/main.py \\
        --games 20 --no-self

    # JSON設定ファイルから読み込む場合
    python tools/run_matches_round_robin.py --config tournament.json --games 20

    tournament.json の例:
    [
      {"name": "baseline", "agent": "src/main.py"},
      {"name": "mcts", "agent": "agents/rl_mcts_sample/src/main.py",
       "deck": "agents/rl_mcts_sample/src/deck.csv"}
    ]

    # 詳細ログをresults/にJSONで保存する
    python tools/run_matches_round_robin.py --agent a=src/main.py --agent b=src/main.py \\
        --games 20 --save-json
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"

AgentFunc = Callable[[dict], list]

# スクリプト起動時点のsys.pathを保持しておき、agent読み込みのたびにこれへ戻す。
_BASE_SYS_PATH = list(sys.path)


@dataclass
class AgentSpec:
    name: str
    agent_path: Path
    deck_path: Path


# Edit this list to control which agents are used when no --agent/--config
# is provided on the command line. This lets you run the tournament by
# changing variables instead of passing CLI args.
DEFAULT_AGENTS: list[AgentSpec] = [
    AgentSpec(
        name="baseline",
        agent_path=Path("agents/random/src/main.py"),
        deck_path=Path("agents/random/src/deck.csv"),
    ),
    AgentSpec(
        name="mcts",
        agent_path=Path("agents/rl_mcts_sample/src/main.py"),
        deck_path=Path("agents/rl_mcts_sample/src/deck.csv"),
    ),
    AgentSpec(
        name="rl_mcts_sample",
        agent_path=Path("agents/rl_mcts_sample/src/main.py"),
        deck_path=Path("agents/rl_mcts_sample/src/deck.csv"),
    ),
]


@dataclass
class LoadedAgent:
    spec: AgentSpec
    func: AgentFunc
    deck: list[int]


def parse_agent_arg(raw: str) -> AgentSpec:
    """`name=path/to/main.py[:path/to/deck.csv]` をAgentSpecへ変換する。"""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            f"--agent は name=path[:deck] の形式で指定してください: {raw!r}"
        )
    name, rest = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"エージェント名が空です: {raw!r}")

    if ":" in rest:
        agent_part, deck_part = rest.rsplit(":", 1)
    else:
        agent_part, deck_part = rest, None

    agent_path = Path(agent_part.strip())
    if deck_part:
        deck_path = Path(deck_part.strip())
    else:
        deck_path = agent_path.parent / "deck.csv"

    return AgentSpec(name=name, agent_path=agent_path, deck_path=deck_path)


def load_agent_specs_from_config(config_path: Path) -> list[AgentSpec]:
    """JSON設定ファイル（[{"name":..,"agent":..,"deck":..}, ...]）を読む。"""
    data = json.loads(config_path.read_text(encoding="utf-8"))
    specs = []
    for entry in data:
        name = entry["name"]
        agent_path = Path(entry["agent"])
        deck_path = Path(entry["deck"]) if entry.get("deck") else agent_path.parent / "deck.csv"
        specs.append(AgentSpec(name=name, agent_path=agent_path, deck_path=deck_path))
    return specs


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


def _reset_sys_path_for(src_dir: Path) -> None:
    """agent固有のsrcディレクトリだけをsys.pathの先頭に置く。

    複数のagentがそれぞれ独自の `cg/` などの同名パッケージを持つ場合、
    先に読み込んだagentのモジュールキャッシュを次のagentが誤って
    再利用しないよう、そのagentが自分のsrc直下に持つトップレベルの
    パッケージ/モジュール名をsys.modulesから外してから読み込む。
    """
    sys.path[:] = [str(src_dir)] + _BASE_SYS_PATH

    if not src_dir.is_dir():
        return
    for entry in src_dir.iterdir():
        name: Optional[str] = None
        if entry.is_dir() and (entry / "__init__.py").exists():
            name = entry.name
        elif entry.suffix == ".py":
            name = entry.stem
        if name:
            sys.modules.pop(name, None)


def load_agent(spec: AgentSpec, module_name: str) -> LoadedAgent:
    """AgentSpecからagent(obs_dict)とデッキを読み込む。

    main.py 互換の実装（`read_deck_csv()` を持つ）であれば、それを差し替えて
    常に指定デッキを返すようにする。これにより env に渡すデッキと、
    エージェントが対戦開始時に申告するデッキを一致させる。
    """
    agent_path = spec.agent_path.resolve()
    deck_path = spec.deck_path.resolve()
    if not agent_path.exists():
        raise FileNotFoundError(f"{spec.name}: {agent_path} が存在しません。")

    deck = load_deck(deck_path)

    src_dir = agent_path.parent
    _reset_sys_path_for(src_dir)

    spec_obj = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec_obj is None or spec_obj.loader is None:
        raise ImportError(f"{agent_path} を読み込めませんでした。")
    module: ModuleType = importlib.util.module_from_spec(spec_obj)
    sys.modules[module_name] = module
    spec_obj.loader.exec_module(module)

    if not hasattr(module, "agent"):
        raise AttributeError(f"{agent_path} に agent(obs_dict) が定義されていません。")

    if hasattr(module, "read_deck_csv"):
        module.read_deck_csv = lambda: list(deck)  # type: ignore[assignment]
    else:
        print(
            f"警告: {spec.name} ({agent_path}) に read_deck_csv が見つからないため、"
            f"対戦開始時に申告されるデッキは {deck_path} と一致しない可能性があります。",
            file=sys.stderr,
        )

    return LoadedAgent(spec=spec, func=module.agent, deck=deck)


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


@dataclass
class HeadToHead:
    """1つの対戦カードの結果。name0==name1なら自己対戦。

    name0_wins / name1_wins は「name0側(player0)として何回勝ったか」
    「name1側(player1)として何回勝ったか」を表す。自己対戦の場合、
    どちらも同じエージェントのインスタンスだが、先手/後手による差を
    見えるようにするため別々にカウントする。
    """

    name0: str
    name1: str
    name0_wins: int = 0
    name1_wins: int = 0
    draws: int = 0
    unresolved: int = 0
    total: int = 0

    @property
    def is_self_match(self) -> bool:
        return self.name0 == self.name1


@dataclass
class OverallRecord:
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    unresolved: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws + self.unresolved

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games * 100) if self.games else 0.0

    @property
    def decided_win_rate(self) -> float:
        decided = self.wins + self.losses
        return (self.wins / decided * 100) if decided else 0.0


def run_pairing(
    agent0: LoadedAgent,
    agent1: LoadedAgent,
    num_games: int,
    alternate_sides: bool,
    debug: bool,
    verbose: bool,
) -> tuple[HeadToHead, list[dict]]:
    name0, name1 = agent0.spec.name, agent1.spec.name
    is_self = name0 == name1
    h2h = HeadToHead(name0=name0, name1=name1)
    game_log: list[dict] = []
    label = f"{name0} vs {name1}" + ("(自己対戦)" if is_self else "")

    for i in range(num_games):
        # 自己対戦の場合はagent0/agent1が同じ実装なのでswapに意味はないが、
        # ログ上の先手/後手表示は一貫させる。
        swap = alternate_sides and (i % 2 == 1)
        agents = [agent1.func, agent0.func] if swap else [agent0.func, agent1.func]
        first_player_label = "player1側" if swap else "player0側"

        env = make("cabt", debug=debug)
        try:
            env.run(agents)
            raw_result, turns = extract_result(env.steps)
        except Exception as exc:  # noqa: BLE001 - 1試合の異常終了で全体を止めない
            raw_result, turns = None, 0
            if verbose:
                print(f"  [{label} #{i + 1}] エラー: {exc}", file=sys.stderr)

        # raw_result は env.run に渡した順番（swap後）基準。
        # h2h.name0_wins / name1_wins は常に (agent0, agent1) 基準に揃え直す。
        if raw_result in (0, 1):
            result = (1 - raw_result) if swap else raw_result
        else:
            result = raw_result

        h2h.total += 1
        if result == 0:
            h2h.name0_wins += 1
            outcome_label = f"{name0}(player0側)の勝ち"
        elif result == 1:
            h2h.name1_wins += 1
            outcome_label = f"{name1}(player1側)の勝ち"
        elif result == 2:
            h2h.draws += 1
            outcome_label = "引き分け"
        else:
            h2h.unresolved += 1
            outcome_label = "不明（エラー）"

        game_log.append(
            {
                "matchup": label,
                "game": i + 1,
                "first_player": first_player_label,
                "result": outcome_label,
                "turns": turns,
            }
        )

        if verbose:
            print(
                f"  [{label} #{i + 1}/{num_games}] {outcome_label} "
                f"(先手={first_player_label}, turns={turns})"
            )

    return h2h, game_log


def print_head_to_head_table(names: list[str], h2h_map: dict[tuple[str, str], HeadToHead]) -> None:
    """縦=自分, 横=相手 の勝ち数マトリクスを表示する。対角成分は自己対戦。"""
    col_width = max(10, *(len(n) for n in names)) + 2
    header = "自分\\相手".ljust(col_width) + "".join(n.ljust(col_width) for n in names)
    print(header)
    for row_name in names:
        cells = []
        for col_name in names:
            key = (row_name, col_name) if (row_name, col_name) in h2h_map else (col_name, row_name)
            h2h = h2h_map.get(key)
            if h2h is None:
                cells.append("n/a".ljust(col_width))
                continue
            if h2h.is_self_match:
                text = f"p0:{h2h.name0_wins}/p1:{h2h.name1_wins}/分:{h2h.draws}"
            elif row_name == h2h.name0:
                text = f"{h2h.name0_wins}/{h2h.total}"
            else:
                text = f"{h2h.name1_wins}/{h2h.total}"
            cells.append(text.ljust(col_width))
        print(row_name.ljust(col_width) + "".join(cells))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        action="append",
        type=parse_agent_arg,
        default=None,
        metavar="NAME=PATH[:DECK]",
        help="対戦させるエージェント。複数回指定できる。例: mcts=agents/rl_mcts_sample/src/main.py:agents/rl_mcts_sample/src/deck.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="エージェント一覧を書いたJSON設定ファイル（--agentの代わりに使用可）",
    )
    parser.add_argument("--games", type=int, default=10, help="1対戦カードあたりの対戦回数")
    parser.add_argument(
        "--no-self",
        action="store_true",
        help="自己対戦（同じエージェント同士）を対戦カードから除外する",
    )
    parser.add_argument(
        "--no-alternate",
        action="store_true",
        help="先手/後手を固定する（デフォルトは1試合ごとに入れ替えて先手有利を均す）",
    )
    parser.add_argument("--debug", action="store_true", help="kaggle_environmentsのdebugログを出す")
    parser.add_argument("--quiet", action="store_true", help="各試合ごとの結果表示を省略する")
    parser.add_argument(
        "--save-json", action="store_true", help="results/に大会全体の詳細ログをJSONで保存する"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    specs: list[AgentSpec] = []
    if args.config:
        specs.extend(load_agent_specs_from_config(args.config))
    if args.agent:
        specs.extend(args.agent)

    # If no agents were passed via CLI/config, fall back to the in-file
    # DEFAULT_AGENTS. Edit DEFAULT_AGENTS above to control which agents
    # are used when running the script without command-line args.
    if not specs:
        print("No agents specified via CLI/config — using in-file DEFAULT_AGENTS.")
        specs.extend(DEFAULT_AGENTS)

    if len(specs) < 2:
        raise SystemExit(
            "少なくとも2体のエージェントを指定してください（--agent を複数回、または --config）。"
        )

    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"エージェント名が重複しています: {names}")

    print("=== エージェント読み込み ===")
    loaded: dict[str, LoadedAgent] = {}
    for index, spec in enumerate(specs):
        agent = load_agent(spec, f"tournament_agent_{index}")
        loaded[spec.name] = agent
        print(f"  {spec.name}: {spec.agent_path.resolve()} (deck: {spec.deck_path.resolve()})")

    if args.no_self:
        pairings = list(itertools.combinations(names, 2))
    else:
        pairings = list(itertools.combinations_with_replacement(names, 2))

    print(
        f"\n=== 総当たり戦開始 (対戦カード数={len(pairings)}, "
        f"カードあたり{args.games}試合, 自己対戦={'除外' if args.no_self else '含む'}) ===\n"
    )

    overall: dict[str, OverallRecord] = {name: OverallRecord(name=name) for name in names}
    h2h_map: dict[tuple[str, str], HeadToHead] = {}
    all_game_logs: list[dict] = []

    started = time.time()
    for name0, name1 in pairings:
        h2h, game_log = run_pairing(
            loaded[name0],
            loaded[name1],
            num_games=args.games,
            alternate_sides=not args.no_alternate,
            debug=args.debug,
            verbose=not args.quiet,
        )
        h2h_map[(name0, name1)] = h2h
        all_game_logs.extend(game_log)

        if h2h.is_self_match:
            # 自己対戦: 片側だけをカウントする。
            # 以前は name0_wins + name1_wins を両方勝ち・負けに積んでいたため
            # 実績が倍数になっていました。ここでは先手側の勝ちを `wins` に、
            # 後手側の勝ちを `losses` にそれぞれ加算して一度だけカウントします。
            overall[name0].wins += h2h.name0_wins
            overall[name0].losses += h2h.name1_wins
            overall[name0].draws += h2h.draws
            overall[name0].unresolved += h2h.unresolved
        else:
            overall[name0].wins += h2h.name0_wins
            overall[name0].losses += h2h.name1_wins
            overall[name0].draws += h2h.draws
            overall[name0].unresolved += h2h.unresolved

            overall[name1].wins += h2h.name1_wins
            overall[name1].losses += h2h.name0_wins
            overall[name1].draws += h2h.draws
            overall[name1].unresolved += h2h.unresolved

        if not args.quiet:
            if h2h.is_self_match:
                print(
                    f"  -> {name0} 自己対戦: player0側 {h2h.name0_wins}勝 / "
                    f"player1側 {h2h.name1_wins}勝 / 引き分け {h2h.draws}\n"
                )
            else:
                print(
                    f"  -> {name0} {h2h.name0_wins}勝 / {name1} {h2h.name1_wins}勝 "
                    f"/ 引き分け {h2h.draws}\n"
                )

    elapsed = time.time() - started

    print("=== 対戦カード別 勝ち数（行 vs 列） ===")
    print_head_to_head_table(names, h2h_map)
    print("  ※対角成分は自己対戦: p0=player0側勝数, p1=player1側勝数, 分=引き分け")

    print("\n=== 総合成績（勝率順） ===")
    ranking = sorted(overall.values(), key=lambda r: r.win_rate, reverse=True)
    header = f"{'順位':<4}{'エージェント':<20}{'試合数':>6}{'勝':>5}{'負':>5}{'分':>5}{'勝率':>8}{'決着勝率':>10}"
    print(header)
    for rank, record in enumerate(ranking, start=1):
        print(
            f"{rank:<4}{record.name:<20}{record.games:>6}{record.wins:>5}"
            f"{record.losses:>5}{record.draws:>5}{record.win_rate:>7.1f}%"
            f"{record.decided_win_rate:>9.1f}%"
        )

    print(f"\n所要時間: {elapsed:.1f}秒")

    if args.save_json:
        RESULTS_ROOT.mkdir(exist_ok=True)
        out_path = RESULTS_ROOT / f"tournament_{int(time.time())}.json"
        out_path.write_text(
            json.dumps(
                {
                    "agents": [
                        {
                            "name": spec.name,
                            "agent_path": str(spec.agent_path.resolve()),
                            "deck_path": str(spec.deck_path.resolve()),
                        }
                        for spec in specs
                    ],
                    "games_per_matchup": args.games,
                    "include_self_matches": not args.no_self,
                    "overall": {
                        record.name: {
                            "games": record.games,
                            "wins": record.wins,
                            "losses": record.losses,
                            "draws": record.draws,
                            "unresolved": record.unresolved,
                            "win_rate": record.win_rate,
                            "decided_win_rate": record.decided_win_rate,
                        }
                        for record in overall.values()
                    },
                    "head_to_head": {
                        f"{name0}_vs_{name1}": {
                            "is_self_match": h2h.is_self_match,
                            "name0_wins": h2h.name0_wins,
                            "name1_wins": h2h.name1_wins,
                            "draws": h2h.draws,
                            "unresolved": h2h.unresolved,
                            "total": h2h.total,
                        }
                        for (name0, name1), h2h in h2h_map.items()
                    },
                    "elapsed_seconds": elapsed,
                    "games": all_game_logs,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"詳細ログを保存しました: {out_path}")


if __name__ == "__main__":
    main()