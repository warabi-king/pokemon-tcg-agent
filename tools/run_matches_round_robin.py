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
        --games 20 --workers 4

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

    # 1つのlibcgで全試合を保持し、MCTSのNN評価をMPSへまとめる
    python tools/run_matches_round_robin.py \\
        --agent a=agents/rl_mcts_r_robin1/src/main.py \\
        --agent b=agents/rl_mcts_r_robin2/src/main.py \\
        --backend batched --device mps --lanes 128 --batch-size 128
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import importlib.util
import itertools
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

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


@dataclass(frozen=True)
class TournamentGameRequest:
    """並列ワーカーへ渡す1試合分の情報。"""

    name0: str
    name1: str
    game_index: int
    swap: bool
    debug: bool


@dataclass(frozen=True)
class TournamentGameResult:
    """1試合の実行結果。resultは常にname0/name1基準。"""

    name0: str
    name1: str
    game_index: int
    swap: bool
    result: Optional[int]
    turns: int
    error: str | None = None


_WORKER_LOADED_AGENTS: dict[str, LoadedAgent] = {}
_KAGGLE_MAKE: Callable[..., object] | None = None
_SILENT_KAGGLE_OUTPUT = False


@contextmanager
def _suppress_process_output():
    """Pythonだけでなくネイティブライブラリのstdout/stderrも一時的に抑える。"""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _load_kaggle_make(*, silent: bool = False) -> Callable[..., object]:
    """重いkaggle_environments importを必要になるまで遅延する。"""
    global _KAGGLE_MAKE
    if _KAGGLE_MAKE is not None:
        return _KAGGLE_MAKE

    if silent:
        with _suppress_process_output():
            from kaggle_environments import make as kaggle_make
    else:
        from kaggle_environments import make as kaggle_make

    _KAGGLE_MAKE = kaggle_make
    return kaggle_make


def _make_cabt(*, decks: list[list[int]] | None = None, debug: bool = False):
    configuration = {"decks": decks} if decks is not None else None
    kaggle_make = _load_kaggle_make()
    if _SILENT_KAGGLE_OUTPUT:
        with _suppress_process_output():
            return kaggle_make("cabt", configuration=configuration, debug=debug)
    return kaggle_make("cabt", configuration=configuration, debug=debug)


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


def resolve_worker_count(requested: int, total_games: int) -> int:
    """0ならCPU数と試合数から自動決定し、モデル複製を考慮して最大4に抑える。"""
    if requested < 0:
        raise ValueError("--workers は0以上で指定してください。")
    if requested == 0:
        # 5試合ほどを1ワーカーの目安にし、短いバッチではspawnコストを避ける。
        requested = min(os.cpu_count() or 1, 4, max(1, (total_games + 4) // 5))
    return max(1, min(requested, total_games))


def _configure_worker_threads() -> None:
    """複数プロセス×内部スレッドによるCPU過剰使用を防ぐ。"""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"


def _init_tournament_worker(specs: list[AgentSpec]) -> None:
    """各ワーカーで全エージェントを一度だけ読み込む。"""
    global _WORKER_LOADED_AGENTS, _SILENT_KAGGLE_OUTPUT

    _configure_worker_threads()
    _load_kaggle_make(silent=True)
    _SILENT_KAGGLE_OUTPUT = True
    _WORKER_LOADED_AGENTS = {
        spec.name: load_agent(spec, f"tournament_worker_{os.getpid()}_{index}")
        for index, spec in enumerate(specs)
    }

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _run_tournament_worker_game(request: TournamentGameRequest) -> TournamentGameResult:
    """初期化済みワーカーで総当たり戦の1試合を実行する。"""
    agent0 = _WORKER_LOADED_AGENTS.get(request.name0)
    agent1 = _WORKER_LOADED_AGENTS.get(request.name1)
    if agent0 is None or agent1 is None:
        raise RuntimeError(
            f"対戦ワーカーにagentがありません: {request.name0}, {request.name1}"
        )

    agents = [agent1.func, agent0.func] if request.swap else [agent0.func, agent1.func]
    decks = [agent1.deck, agent0.deck] if request.swap else [agent0.deck, agent1.deck]
    env = _make_cabt(decks=decks, debug=request.debug)
    try:
        if _SILENT_KAGGLE_OUTPUT and not request.debug:
            with _suppress_process_output():
                env.run(agents)
        else:
            env.run(agents)
        raw_result, turns = extract_result(env.steps)
        error = None
    except Exception as exc:  # noqa: BLE001 - 1試合の異常終了で全体を止めない
        raw_result, turns = None, 0
        error = f"{type(exc).__name__}: {exc}"

    if raw_result in (0, 1):
        result = (1 - raw_result) if request.swap else raw_result
    else:
        result = raw_result

    return TournamentGameResult(
        name0=request.name0,
        name1=request.name1,
        game_index=request.game_index,
        swap=request.swap,
        result=result,
        turns=turns,
        error=error,
    )


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

        env = _make_cabt(debug=debug)
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


def run_tournament_parallel(
    specs: list[AgentSpec],
    pairings: list[tuple[str, str]],
    num_games: int,
    workers: int,
    alternate_sides: bool,
    debug: bool,
    verbose: bool,
) -> tuple[dict[tuple[str, str], HeadToHead], list[dict]]:
    """総当たり戦の全試合を独立プロセスへ分配する。"""
    h2h_map = {
        (name0, name1): HeadToHead(name0=name0, name1=name1)
        for name0, name1 in pairings
    }
    requests = [
        TournamentGameRequest(
            name0=name0,
            name1=name1,
            game_index=i + 1,
            swap=alternate_sides and (i % 2 == 1),
            debug=debug,
        )
        for name0, name1 in pairings
        for i in range(num_games)
    ]
    all_game_logs: list[dict] = []

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_init_tournament_worker,
        initargs=(specs,),
    ) as executor:
        future_to_request = {
            executor.submit(_run_tournament_worker_game, request): request
            for request in requests
        }
        completed = 0
        for future in as_completed(future_to_request):
            request = future_to_request[future]
            try:
                game_result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"{request.name0} vs {request.name1} "
                    f"試合{request.game_index}のワーカー実行に失敗しました: {exc}"
                ) from exc

            completed += 1
            key = (game_result.name0, game_result.name1)
            h2h = h2h_map[key]
            h2h.total += 1
            if game_result.result == 0:
                h2h.name0_wins += 1
                outcome_label = f"{game_result.name0}(player0側)の勝ち"
            elif game_result.result == 1:
                h2h.name1_wins += 1
                outcome_label = f"{game_result.name1}(player1側)の勝ち"
            elif game_result.result == 2:
                h2h.draws += 1
                outcome_label = "引き分け"
            else:
                h2h.unresolved += 1
                outcome_label = "不明（エラー）"

            is_self = game_result.name0 == game_result.name1
            label = f"{game_result.name0} vs {game_result.name1}"
            if is_self:
                label += "(自己対戦)"
            first_player_label = "player1側" if game_result.swap else "player0側"
            all_game_logs.append(
                {
                    "matchup": label,
                    "game": game_result.game_index,
                    "first_player": first_player_label,
                    "result": outcome_label,
                    "turns": game_result.turns,
                }
            )

            if game_result.error and verbose:
                print(
                    f"  [{label} #{game_result.game_index}] エラー: {game_result.error}",
                    file=sys.stderr,
                )
            if verbose:
                print(
                    f"  [{completed}/{len(requests)}, {label} "
                    f"#{game_result.game_index}/{num_games}] {outcome_label} "
                    f"(先手={first_player_label}, turns={game_result.turns})"
                )

    # 完了順に依存しない安定したJSONにする。
    all_game_logs.sort(key=lambda game: (game["matchup"], game["game"]))
    return h2h_map, all_game_logs


def aggregate_tournament_results(
    pairings: list[tuple[str, str]],
    game_results: list[object],
    num_games: int,
    verbose: bool,
) -> tuple[dict[tuple[str, str], HeadToHead], list[dict]]:
    """共通フィールドを持つ試合結果列を総当たり集計へ変換する。"""
    h2h_map = {
        (name0, name1): HeadToHead(name0=name0, name1=name1)
        for name0, name1 in pairings
    }
    all_game_logs: list[dict] = []

    for completed, game_result in enumerate(game_results, start=1):
        name0 = getattr(game_result, "name0")
        name1 = getattr(game_result, "name1")
        game_index = getattr(game_result, "game_index")
        swap = getattr(game_result, "swap")
        result = getattr(game_result, "result")
        turns = getattr(game_result, "turns")
        error = getattr(game_result, "error")

        h2h = h2h_map[(name0, name1)]
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

        label = f"{name0} vs {name1}"
        if name0 == name1:
            label += "(自己対戦)"
        first_player_label = "player1側" if swap else "player0側"
        all_game_logs.append(
            {
                "matchup": label,
                "game": game_index,
                "first_player": first_player_label,
                "result": outcome_label,
                "turns": turns,
            }
        )

        if error and verbose:
            print(f"  [{label} #{game_index}] エラー: {error}", file=sys.stderr)
        if verbose:
            print(
                f"  [{completed}/{len(game_results)}, {label} "
                f"#{game_index}/{num_games}] {outcome_label} "
                f"(先手={first_player_label}, selections={turns})"
            )

    all_game_logs.sort(key=lambda game: (game["matchup"], game["game"]))
    return h2h_map, all_game_logs


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
        "--backend",
        choices=(
            "legacy",
            "batched",
            "worker-batched",
            "cuda-streams",
            "cuda-ensemble",
            "gpu-tree",
        ),
        default="legacy",
        help=(
            "legacyは1試合ずつkaggle環境で実行。batchedは1つのlibcgで複数試合を保持し、"
            "MCTSのNN評価を試合横断でGPUバッチ化する。worker-batchedは複数CPU processで"
            "libcgと特徴量を並列生成し、NN要求だけを中央GPUへ集約する。cuda-streamsは異なるモデルを"
            "専用CUDA streamへ並行投入する。cuda-ensembleは小batch時に全モデルの重みを"
            "stackしたvmap演算を使い、大batch時はper-model CUDAへ自動切替する。"
            "gpu-treeは探索木もdevice "
            "Tensorへ常駐させる（デフォルト: legacy）"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "CPUワーカー数。0はlegacyでは最大4を自動選択し、"
            "worker-batchedではlaneごとに1 worker（デフォルト: 0）"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="batched backendのPyTorch device。auto/cpu/mps/cudaなど（デフォルト: auto）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "モデル別NN最大batch size。省略時はworker-batched=256、"
            "その他のbackend=128"
        ),
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=0,
        help=(
            "batched backendで同時に保持する最大試合数。"
            "0は総試合数（全組み合わせを一度に実行）（デフォルト: 0）"
        ),
    )
    parser.add_argument(
        "--search-count",
        type=int,
        default=10,
        help="batched backendの1手あたりMCTS simulation数（デフォルト: 10）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="batched backendの乱数seed（デフォルト: 0）",
    )
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
    parser.add_argument(
        "--training-json-dir",
        type=Path,
        default=None,
        help=(
            "batched/worker-batched対戦中に、学習に必要な局面だけを"
            "1試合1JSONで直接保存するディレクトリ"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.games < 1:
        raise SystemExit("--games は1以上で指定してください。")
    if args.batch_size is None:
        args.batch_size = 256 if args.backend == "worker-batched" else 128
    if args.batch_size < 1:
        raise SystemExit("--batch-size は1以上で指定してください。")
    if args.training_json_dir is not None:
        if args.backend not in ("batched", "worker-batched", "cuda-streams", "cuda-ensemble"):
            raise SystemExit(
                "--training-json-dirはbatched/worker-batched/cuda-streams/"
                "cuda-ensemble backendで使用してください。"
            )
        args.training_json_dir = args.training_json_dir.resolve()
        args.training_json_dir.mkdir(parents=True, exist_ok=True)
        existing_training_json = next(
            args.training_json_dir.glob("episode_*.json"),
            None,
        )
        if existing_training_json is not None:
            raise SystemExit(
                "--training-json-dirに既存の学習JSONがあります。"
                f"別ディレクトリを指定してください: {existing_training_json}"
            )

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

    # spawnした子プロセスでも同じ場所を参照できるよう絶対パスへ正規化する。
    specs = [
        AgentSpec(
            name=spec.name,
            agent_path=spec.agent_path.resolve(),
            deck_path=spec.deck_path.resolve(),
        )
        for spec in specs
    ]

    print("=== エージェント確認 ===")
    for spec in specs:
        if not spec.agent_path.exists():
            raise FileNotFoundError(f"{spec.name}: {spec.agent_path} が存在しません。")
        load_deck(spec.deck_path)
        print(f"  {spec.name}: {spec.agent_path} (deck: {spec.deck_path})")

    if args.no_self:
        pairings = list(itertools.combinations(names, 2))
    else:
        pairings = list(itertools.combinations_with_replacement(names, 2))

    total_games = len(pairings) * args.games
    if args.lanes < 0:
        raise SystemExit("--lanes は0以上で指定してください。")
    # 0は全試合をlaneへ載せる。8エージェント・自己対戦込みなら
    # 1ラウンド36試合なので、--games kに対して36*k laneになる。
    lanes = total_games if args.lanes == 0 else min(args.lanes, total_games)
    workers = 0
    if args.backend == "worker-batched" and args.workers == 0:
        workers = lanes
    elif args.backend in ("legacy", "worker-batched"):
        try:
            workers = resolve_worker_count(args.workers, total_games)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    execution_label = (
        f"legacy, 並列ワーカー={workers}"
        if args.backend == "legacy"
        else (
            f"worker-batched, CPUワーカー={workers}, device={args.device}, "
            f"lanes={lanes}, batch-size={args.batch_size}"
            if args.backend == "worker-batched"
            else (
                f"{args.backend}, device={args.device}, lanes={lanes}, "
                f"batch-size={args.batch_size}"
                if args.backend in ("batched", "cuda-streams", "cuda-ensemble")
                else (
                    f"gpu-tree, device={args.device}, lanes={lanes}, "
                    f"batch-size={args.batch_size}"
                )
            )
        )
    )

    print(
        f"\n=== 総当たり戦開始 (対戦カード数={len(pairings)}, "
        f"カードあたり{args.games}試合, 自己対戦={'除外' if args.no_self else '含む'}, "
        f"backend={execution_label}) ===\n"
    )

    overall: dict[str, OverallRecord] = {name: OverallRecord(name=name) for name in names}
    started = time.time()
    batched_output = None
    if args.backend == "worker-batched":
        from batched_tournament import run_worker_batched_tournament

        batched_output = run_worker_batched_tournament(
            specs=specs,
            pairings=pairings,
            num_games=args.games,
            alternate_sides=not args.no_alternate,
            device_name=args.device,
            batch_size=args.batch_size,
            lanes=lanes,
            search_count=args.search_count,
            seed=args.seed,
            cpu_workers=workers,
            training_json_dir=args.training_json_dir,
        )
        h2h_map, all_game_logs = aggregate_tournament_results(
            pairings,
            batched_output.results,
            num_games=args.games,
            verbose=not args.quiet,
        )
    elif args.backend == "gpu-tree":
        from gpu_tree_tournament import run_gpu_tree_tournament

        batched_output = run_gpu_tree_tournament(
            specs=specs,
            pairings=pairings,
            num_games=args.games,
            alternate_sides=not args.no_alternate,
            device_name=args.device,
            batch_size=args.batch_size,
            lanes=lanes,
            search_count=args.search_count,
            seed=args.seed,
        )
        h2h_map, all_game_logs = aggregate_tournament_results(
            pairings,
            batched_output.results,
            num_games=args.games,
            verbose=not args.quiet,
        )
    elif args.backend in ("batched", "cuda-streams", "cuda-ensemble"):
        from batched_tournament import run_batched_tournament

        if args.backend in ("cuda-streams", "cuda-ensemble") and args.device not in (
            "auto",
            "cuda",
        ):
            raise SystemExit(
                f"{args.backend} backendの--deviceはcudaまたはautoにしてください。"
            )

        batched_output = run_batched_tournament(
            specs=specs,
            pairings=pairings,
            num_games=args.games,
            alternate_sides=not args.no_alternate,
            device_name=args.device,
            batch_size=args.batch_size,
            lanes=lanes,
            search_count=args.search_count,
            seed=args.seed,
            parallel_cuda_models=args.backend == "cuda-streams",
            cuda_ensemble_models=args.backend == "cuda-ensemble",
            training_json_dir=args.training_json_dir,
        )
        h2h_map, all_game_logs = aggregate_tournament_results(
            pairings,
            batched_output.results,
            num_games=args.games,
            verbose=not args.quiet,
        )
    elif workers == 1:
        loaded: dict[str, LoadedAgent] = {}
        for index, spec in enumerate(specs):
            loaded[spec.name] = load_agent(spec, f"tournament_agent_{index}")

        h2h_map: dict[tuple[str, str], HeadToHead] = {}
        all_game_logs: list[dict] = []
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
    else:
        h2h_map, all_game_logs = run_tournament_parallel(
            specs=specs,
            pairings=pairings,
            num_games=args.games,
            workers=workers,
            alternate_sides=not args.no_alternate,
            debug=args.debug,
            verbose=not args.quiet,
        )

    for name0, name1 in pairings:
        h2h = h2h_map[(name0, name1)]
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

    if args.training_json_dir is not None:
        training_json_count = sum(
            1 for _ in args.training_json_dir.glob("episode_*.json")
        )
        print(
            f"学習JSON: {training_json_count}試合を保存しました: "
            f"{args.training_json_dir}"
        )

    if batched_output is not None:
        profile = batched_output.profile
        print(f"\n=== {args.backend} backend profile ===")
        print(f"device: {batched_output.device}")
        print(
            f"NN: {profile.nn_seconds:.3f}秒 / {profile.nn_evaluations}評価 / "
            f"{profile.nn_batches}batch "
            f"(平均batch={profile.mean_batch_size:.1f}, 最大={profile.max_batch_size})"
        )
        print(
            f"libcg Search: begin={profile.search_begin_seconds:.3f}秒, "
            f"step={profile.search_step_seconds:.3f}秒/{profile.search_steps}回, "
            f"cleanup={profile.search_finalize_seconds:.3f}秒"
        )
        print(
            "Search.step内訳: "
            f"C API+ctypes={profile.search_step_c_api_seconds:.3f}秒, "
            f"JSON decode+object={profile.search_step_json_seconds:.3f}秒, "
            f"dataclass={profile.search_step_dataclass_seconds:.3f}秒"
        )
        print(
            f"libcg Battle: start={profile.battle_start_seconds:.3f}秒, "
            f"step={profile.battle_step_seconds:.3f}秒/{profile.battle_steps}回, "
            f"特徴量生成={profile.feature_seconds:.3f}秒"
        )
        if args.backend == "cuda-ensemble":
            print(
                "CUDA wave: "
                f"model-axis={profile.cuda_ensemble_waves}, "
                f"per-model={profile.cuda_per_model_waves}"
            )
        if args.backend == "worker-batched":
            print(
                "中央NN内訳: "
                f"merge/pad={profile.nn_merge_seconds:.3f}秒, "
                f"from_numpy/H2D={profile.nn_input_seconds:.3f}秒, "
                f"forward投入={profile.nn_forward_submit_seconds:.3f}秒, "
                f"forward待ち+D2H={profile.nn_output_wait_seconds:.3f}秒, "
                f"tolist={profile.nn_tolist_seconds:.3f}秒, "
                f"応答分割={profile.nn_response_pack_seconds:.3f}秒, "
                f"response put={profile.response_put_seconds:.3f}秒"
            )
            print(
                "Decoder padding: "
                f"source={profile.nn_decoder_source_tokens} token, "
                f"padded={profile.nn_decoder_padded_tokens} token, "
                f"有効率={profile.decoder_token_efficiency:.1%}"
            )
            print(
                f"CPU workers: {profile.cpu_workers}, "
                f"worker NN待ち合計={profile.remote_wait_seconds:.3f}秒, "
                f"worker NumPy梱包={profile.remote_numpy_pack_seconds:.3f}秒, "
                f"中央batch収集={profile.batch_collect_seconds:.3f}秒, "
                f"IPC request={profile.ipc_messages}回"
            )
            print(
                "CUDA wave: "
                f"model-axis={profile.cuda_ensemble_waves}, "
                f"per-model={profile.cuda_per_model_waves}"
            )
        if hasattr(profile, "gpu_tree_seconds"):
            print(
                f"GPU tree: {profile.gpu_tree_seconds:.3f}秒, "
                f"GPU→CPU small transfer/sync={profile.gpu_to_cpu_seconds:.3f}秒, "
                f"packed input={profile.packed_input_bytes / 1024 / 1024:.1f}MiB"
            )

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
                    "backend": args.backend,
                    "workers": workers,
                    "device": batched_output.device if batched_output is not None else None,
                    "batched_profile": (
                        {
                            "nn_seconds": batched_output.profile.nn_seconds,
                            "nn_evaluations": batched_output.profile.nn_evaluations,
                            "nn_batches": batched_output.profile.nn_batches,
                            "mean_batch_size": batched_output.profile.mean_batch_size,
                            "max_batch_size": batched_output.profile.max_batch_size,
                            "nn_merge_seconds": batched_output.profile.nn_merge_seconds,
                            "nn_input_seconds": batched_output.profile.nn_input_seconds,
                            "nn_forward_submit_seconds": batched_output.profile.nn_forward_submit_seconds,
                            "nn_output_wait_seconds": batched_output.profile.nn_output_wait_seconds,
                            "nn_tolist_seconds": batched_output.profile.nn_tolist_seconds,
                            "nn_response_pack_seconds": batched_output.profile.nn_response_pack_seconds,
                            "nn_decoder_source_tokens": batched_output.profile.nn_decoder_source_tokens,
                            "nn_decoder_padded_tokens": batched_output.profile.nn_decoder_padded_tokens,
                            "decoder_token_efficiency": batched_output.profile.decoder_token_efficiency,
                            "response_put_seconds": batched_output.profile.response_put_seconds,
                            "remote_numpy_pack_seconds": batched_output.profile.remote_numpy_pack_seconds,
                            "search_begin_seconds": batched_output.profile.search_begin_seconds,
                            "search_step_seconds": batched_output.profile.search_step_seconds,
                            "search_step_c_api_seconds": batched_output.profile.search_step_c_api_seconds,
                            "search_step_json_seconds": batched_output.profile.search_step_json_seconds,
                            "search_step_dataclass_seconds": batched_output.profile.search_step_dataclass_seconds,
                            "search_steps": batched_output.profile.search_steps,
                            "battle_step_seconds": batched_output.profile.battle_step_seconds,
                            "battle_steps": batched_output.profile.battle_steps,
                            "feature_seconds": batched_output.profile.feature_seconds,
                        }
                        if batched_output is not None
                        else None
                    ),
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
