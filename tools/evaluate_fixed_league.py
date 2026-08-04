"""agentを別processに隔離し、固定seed・先後均衡で対戦評価する。

同名Python packageを持つagent同士を同一processへimportするとmodule cacheが衝突する。
この評価器はagentごとに永続workerを起動し、推論だけをPipe越しに呼び出す。

実行例:
    python tools/evaluate_fixed_league.py \
        --agent-a agents/belief_puct/src \
        --agent-b agents/rl_mcts/src \
        --games 100 --seed-start 41000 \
        --output results/stage2_belief_puct_vs_rl_mcts.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Callable

from kaggle_environments import make


@dataclass(frozen=True)
class GameResult:
    """1試合の固定条件と結果を保持する。"""

    game_index: int
    seed: int
    swapped: bool
    first_player: str
    winner: str | None
    raw_result: int | None
    turns: int
    elapsed_seconds: float
    error: str | None


@dataclass(frozen=True)
class EvaluationSummary:
    """評価全体の勝敗、score、Wilson区間、速度を保持する。"""

    games_requested: int
    games_completed: int
    wins_a: int
    wins_b: int
    draws: int
    errors: int
    score_a: float
    wilson_lower_95: float
    wilson_upper_95: float
    elapsed_seconds: float
    mean_seconds_per_game: float


def wilson_interval(successes: float, trials: int, z_score: float = 1.959963984540054) -> tuple[float, float]:
    """drawを0.5成功として許容する95% Wilson区間を返す。"""

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


def read_deck(deck_path: Path) -> list[int]:
    """1行1カードIDのdeck.csvを読み、60枚制約を検証する。"""

    deck = [int(line) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csvが60枚ではありません: {deck_path} ({len(deck)}枚)")
    return deck


def _load_main(agent_src: Path, worker_name: str) -> ModuleType:
    """worker固有process内で提出用main.pyを読み込む。"""

    main_path = agent_src / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.pyがありません: {main_path}")
    os.chdir(agent_src)
    sys.path.insert(0, str(agent_src))
    spec = importlib.util.spec_from_file_location(worker_name, main_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"main.pyをimportできません: {main_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[worker_name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "agent", None)):
        raise AttributeError(f"agent(obs_dict)がありません: {main_path}")
    return module


def _patch_deck_readers(module: ModuleType, deck: list[int]) -> None:
    """import済みのagent内read_deck_csvを評価指定deckへ固定する。"""

    fixed_reader = lambda: list(deck)
    if hasattr(module, "read_deck_csv"):
        module.read_deck_csv = fixed_reader  # type: ignore[attr-defined]
    for imported_module in list(sys.modules.values()):
        if imported_module is not None and hasattr(imported_module, "read_deck_csv"):
            module_file = getattr(imported_module, "__file__", None)
            if module_file is not None and str(Path(module_file).resolve()).startswith(str(Path.cwd())):
                imported_module.read_deck_csv = fixed_reader  # type: ignore[attr-defined]


def _checkpoint_model_kwargs(model_path: Path) -> dict[str, int] | None:
    """checkpoint形状から、既定値と異なる既知のTransformer設定を復元する。

    16model_pretrained_upsize1の重みは192次元・3/2層だが、配布srcは128次元の
    factory既定値を持つ。そのままではstate_dictを読めないため、評価時だけ
    重みに対応するfactory引数を返す。未知の構造は推測して比較しない。
    """

    import torch

    state = torch.load(model_path, map_location=torch.device("cpu"))
    if not isinstance(state, dict) or "encoder_bag.weight" not in state:
        raise ValueError(f"未対応のmodel state_dict形式です: {model_path}")

    d_model = int(state["encoder_bag.weight"].shape[1])
    d_feedforward = int(state["encoder.layers.0.linear1.weight"].shape[0])
    encoder_layers = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("encoder.layers.")
    }
    decoder_layers = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("decoder.")
    }
    architecture = (d_model, d_feedforward, len(encoder_layers), len(decoder_layers))
    if architecture == (128, 256, 1, 1):
        return None
    if architecture == (192, 768, 3, 2):
        return {
            "d_model": 192,
            "num_heads": 6,
            "d_feedforward": 768,
            "num_layers_encoder": 3,
            "num_layers_decoder": 2,
        }
    raise ValueError(
        "checkpointのTransformer構造が未登録です: "
        f"d_model={d_model}, d_feedforward={d_feedforward}, "
        f"encoder_layers={len(encoder_layers)}, decoder_layers={len(decoder_layers)} "
        f"({model_path})"
    )


def _patch_model_factory_for_checkpoint(agent_object: Any, model_path: Path) -> None:
    """agentが遅延読込時に、checkpointと同じ構造のモデルを生成するよう差し替える。"""

    model_kwargs = _checkpoint_model_kwargs(model_path)
    if model_kwargs is None:
        return
    agent_module = sys.modules.get(agent_object.__class__.__module__)
    if agent_module is None or not callable(getattr(agent_module, "create_model", None)):
        raise ValueError("checkpoint構造を適用できるcreate_modelがagentにありません")
    original_factory = agent_module.create_model

    def checkpoint_compatible_factory() -> Any:
        """評価対象checkpoint用の明示したTransformer構造を生成する。"""

        return original_factory(**model_kwargs)

    agent_module.create_model = checkpoint_compatible_factory


def _configure_agent(module: ModuleType, options: dict[str, Any]) -> None:
    """対応agentだけに決定化数と探索数の評価用overrideを適用する。"""

    agent_object = getattr(module, "_AGENT", None)
    model_path = options.get("model_path")
    if model_path is not None:
        if agent_object is None or not hasattr(agent_object, "model_path"):
            raise ValueError("model checkpoint overrideに対応しないagentです")
        agent_object.model_path = Path(model_path).resolve()
        agent_object.model = None
        _patch_model_factory_for_checkpoint(agent_object, agent_object.model_path)
    decision_engine = getattr(agent_object, "decision_engine", None)
    if decision_engine is None:
        if options.get("determinizations") is not None:
            raise ValueError("決定化数overrideに対応しないagentです")
        search_count = options.get("search_count")
        if search_count is not None:
            if search_count <= 0 or not hasattr(agent_object, "search_count"):
                raise ValueError("探索数overrideに対応しないagentです")
            agent_object.search_count = search_count
        return
    determinizations = options.get("determinizations")
    search_count = options.get("search_count")
    if determinizations is not None:
        if determinizations <= 0:
            raise ValueError("determinizations overrideは1以上である必要があります")
        decision_engine.determinizations = determinizations
    if search_count is not None:
        if search_count <= 0:
            raise ValueError("search_count overrideは1以上である必要があります")
        decision_engine.search_count_per_determinization = search_count


def _worker_loop(
    connection: Any,
    agent_src_raw: str,
    deck: list[int],
    worker_name: str,
    options: dict[str, Any],
) -> None:
    """agentを一度だけloadし、reset/act/close要求へ応答する。"""

    try:
        module = _load_main(Path(agent_src_raw).resolve(), worker_name)
        _patch_deck_readers(module, deck)
        _configure_agent(module, options)
        connection.send({"status": "ready"})
        while True:
            request = connection.recv()
            command = request.get("command")
            if command == "close":
                break
            if command == "reset":
                agent_object = getattr(module, "_AGENT", None)
                reset_match = getattr(agent_object, "reset_match", None)
                if callable(reset_match):
                    reset_match()
                connection.send({"status": "ok"})
                continue
            if command != "act":
                raise ValueError(f"未知のworker commandです: {command}")
            observation = request["observation"]
            if observation.get("select") is None:
                action = list(deck)
            else:
                action = module.agent(observation)
            connection.send({"status": "ok", "action": action})
    except BaseException as error:  # worker側の原因を親processへ保存するため捕捉する
        try:
            connection.send(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class IsolatedAgent:
    """別processのagentをKaggle callbackとして公開するproxy。"""

    def __init__(
        self,
        agent_src: Path,
        deck: list[int],
        worker_name: str,
        timeout_seconds: float,
        options: dict[str, Any] | None = None,
    ) -> None:
        """spawn workerを起動し、提出コードのload完了まで待つ。"""

        self.agent_src = agent_src
        self.deck = list(deck)
        self.worker_name = worker_name
        self.timeout_seconds = timeout_seconds
        self.options = dict(options or {})
        self.last_error: str | None = None
        self._start_worker()

    def _start_worker(self) -> None:
        """保存済み設定から新しい隔離workerを起動する。"""

        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe()
        self.connection = parent_connection
        self.process = context.Process(
            target=_worker_loop,
            args=(
                child_connection,
                str(self.agent_src.resolve()),
                self.deck,
                self.worker_name,
                self.options,
            ),
            daemon=True,
        )
        self.process.start()
        child_connection.close()
        ready = self._receive("agent初期化")
        if ready.get("status") != "ready":
            raise RuntimeError(self._format_worker_error(ready))

    @staticmethod
    def _format_worker_error(response: dict[str, Any]) -> str:
        """worker応答を診断可能な一つのエラー文字列へ変換する。"""

        return f"{response.get('error', 'worker error')}\n{response.get('traceback', '')}".rstrip()

    def _receive(self, operation: str) -> dict[str, Any]:
        """timeoutつきでworker応答を受け、異常終了を明示する。"""

        if not self.connection.poll(self.timeout_seconds):
            raise TimeoutError(f"{operation}が{self.timeout_seconds:.1f}秒でtimeoutしました")
        response = self.connection.recv()
        if response.get("status") == "error":
            self.last_error = self._format_worker_error(response)
            raise RuntimeError(self.last_error)
        return response

    def reset_match(self) -> None:
        """workerの試合依存状態を初期化する。"""

        self.last_error = None
        self.connection.send({"command": "reset"})
        self._receive("試合reset")

    def __call__(
        self,
        observation: dict[str, Any],
        configuration: dict[str, Any] | None = None,
    ) -> list[int]:
        """Kaggle環境からの観測をworkerへ送り、行動を返す。

        configurationはKaggle callback互換のため受け取るが、agentの判断には使わない。
        """

        self.connection.send({"command": "act", "observation": observation})
        response = self._receive("agent推論")
        return list(response["action"])

    def close(self) -> None:
        """workerへ終了を通知し、残ったprocessを安全に停止する。"""

        if self.process.is_alive():
            try:
                self.connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.0)
        self.connection.close()

    def restart(self) -> None:
        """timeoutや未判定後にworkerを破棄し、通信状態を初期化する。"""

        self.close()
        self._start_worker()


@contextmanager
def game_deadline(timeout_seconds: float):
    """main process内の1試合へwall-clock deadlineを適用する。"""

    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum: int, frame: Any) -> None:
        """SIGALRMを通常の評価エラーとして扱える例外へ変換する。"""

        del signum, frame
        raise TimeoutError(f"1試合が{timeout_seconds:.1f}秒でtimeoutしました")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def extract_result(steps: list[Any]) -> tuple[int | None, int]:
    """Kaggle stepsの最終rewardからplayer順の0/1/2結果を返す。"""

    if not steps or len(steps[-1]) != 2:
        return None, len(steps)
    rewards = [player_state.get("reward") for player_state in steps[-1]]
    if rewards == [1, -1]:
        return 0, len(steps)
    if rewards == [-1, 1]:
        return 1, len(steps)
    if rewards[0] is not None and rewards[0] == rewards[1]:
        return 2, len(steps)
    return None, len(steps)


def run_evaluation(
    proxy_a: IsolatedAgent,
    proxy_b: IsolatedAgent,
    name_a: str,
    name_b: str,
    games: int,
    seed_start: int,
    game_timeout_seconds: float,
) -> tuple[EvaluationSummary, list[GameResult]]:
    """固定連番seedと先後入替で指定試合数を実行する。"""

    results: list[GameResult] = []
    evaluation_started = time.perf_counter()
    for game_index in range(games):
        seed = seed_start + game_index // 2
        swapped = bool(game_index % 2)
        first_player = name_b if swapped else name_a
        agents: list[Callable[[dict[str, Any]], list[int]]] = (
            [proxy_b, proxy_a] if swapped else [proxy_a, proxy_b]
        )
        game_started = time.perf_counter()
        error_message: str | None = None
        raw_result: int | None = None
        turns = 0
        restart_workers = False
        try:
            proxy_a.reset_match()
            proxy_b.reset_match()
            with game_deadline(game_timeout_seconds):
                environment = make("cabt", configuration={"seed": seed}, debug=False)
                environment.run(agents)
                raw_result, turns = extract_result(environment.steps)
                if raw_result is None:
                    final_step = environment.steps[-1] if environment.steps else []
                    unresolved = [
                        {
                            "status": player_state.get("status"),
                            "reward": player_state.get("reward"),
                            "info": player_state.get("info"),
                        }
                        for player_state in final_step
                    ]
                    error_message = "未判定終了: " + json.dumps(
                        unresolved,
                        ensure_ascii=False,
                        default=str,
                    )
                    worker_errors = {
                        name: error
                        for name, error in ((name_a, proxy_a.last_error), (name_b, proxy_b.last_error))
                        if error is not None
                    }
                    if worker_errors:
                        error_message += "\nagent例外: " + json.dumps(
                            worker_errors,
                            ensure_ascii=False,
                        )
                    restart_workers = True
        except Exception as error:  # 失敗試合を勝敗から除外し、原因をJSONへ残す
            error_message = f"{type(error).__name__}: {error}"
            restart_workers = True

        normalized_result = raw_result
        if swapped and raw_result in (0, 1):
            normalized_result = 1 - raw_result
        winner = name_a if normalized_result == 0 else name_b if normalized_result == 1 else None
        results.append(
            GameResult(
                game_index=game_index,
                seed=seed,
                swapped=swapped,
                first_player=first_player,
                winner=winner,
                raw_result=normalized_result,
                turns=turns,
                elapsed_seconds=time.perf_counter() - game_started,
                error=error_message,
            )
        )
        print(
            f"[{game_index + 1}/{games}] seed={seed} first={first_player} "
            f"winner={winner or ('draw' if normalized_result == 2 else 'error')} turns={turns}",
            flush=True,
        )
        if error_message is not None:
            # 総当たり実行中にも失敗原因を即座に見えるようにする。従来は
            # JSONへだけ保存され、完走前に停止すると原因を確認できなかった。
            print(f"  error: {error_message}", flush=True)
        if restart_workers:
            proxy_a.restart()
            proxy_b.restart()

    wins_a = sum(result.raw_result == 0 for result in results)
    wins_b = sum(result.raw_result == 1 for result in results)
    draws = sum(result.raw_result == 2 for result in results)
    errors = sum(result.error is not None or result.raw_result is None for result in results)
    completed = wins_a + wins_b + draws
    score_a = (wins_a + 0.5 * draws) / completed if completed else 0.0
    lower, upper = wilson_interval(wins_a + 0.5 * draws, completed)
    elapsed = time.perf_counter() - evaluation_started
    return (
        EvaluationSummary(
            games_requested=games,
            games_completed=completed,
            wins_a=wins_a,
            wins_b=wins_b,
            draws=draws,
            errors=errors,
            score_a=score_a,
            wilson_lower_95=lower,
            wilson_upper_95=upper,
            elapsed_seconds=elapsed,
            mean_seconds_per_game=elapsed / games if games else 0.0,
        ),
        results,
    )


def parse_args() -> argparse.Namespace:
    """評価対象、deck、試合数、seed、出力先を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-a", type=Path, required=True, help="agent Aのsrcディレクトリ")
    parser.add_argument("--agent-b", type=Path, required=True, help="agent Bのsrcディレクトリ")
    parser.add_argument("--deck-a", type=Path, default=None)
    parser.add_argument("--deck-b", type=Path, default=None)
    parser.add_argument("--name-a", default=None)
    parser.add_argument("--name-b", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=41000)
    parser.add_argument("--action-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--game-timeout-seconds",
        type=float,
        default=600.0,
        help="1試合全体のwall-clock上限。timeout試合はerrorとして除外しworkerを再起動する",
    )
    parser.add_argument("--determinizations-a", type=int, default=None)
    parser.add_argument("--determinizations-b", type=int, default=None)
    parser.add_argument("--search-count-a", type=int, default=None)
    parser.add_argument("--search-count-b", type=int, default=None)
    parser.add_argument("--model-a", type=Path, default=None)
    parser.add_argument("--model-b", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """隔離workerを起動し、評価結果と完全な条件をJSONへ保存する。"""

    args = parse_args()
    agent_a = args.agent_a.resolve()
    agent_b = args.agent_b.resolve()
    deck_a_path = (args.deck_a or agent_a / "deck.csv").resolve()
    deck_b_path = (args.deck_b or agent_b / "deck.csv").resolve()
    deck_a = read_deck(deck_a_path)
    deck_b = read_deck(deck_b_path)
    name_a = args.name_a or agent_a.parent.name
    name_b = args.name_b or agent_b.parent.name

    options_a = {
        "determinizations": args.determinizations_a,
        "search_count": args.search_count_a,
        "model_path": str(args.model_a.resolve()) if args.model_a else None,
    }
    options_b = {
        "determinizations": args.determinizations_b,
        "search_count": args.search_count_b,
        "model_path": str(args.model_b.resolve()) if args.model_b else None,
    }
    proxy_a = IsolatedAgent(
        agent_a,
        deck_a,
        "isolated_agent_a",
        args.action_timeout_seconds,
        options_a,
    )
    proxy_b = IsolatedAgent(
        agent_b,
        deck_b,
        "isolated_agent_b",
        args.action_timeout_seconds,
        options_b,
    )
    try:
        summary, games = run_evaluation(
            proxy_a,
            proxy_b,
            name_a,
            name_b,
            args.games,
            args.seed_start,
            args.game_timeout_seconds,
        )
    finally:
        proxy_a.close()
        proxy_b.close()

    report = {
        "format": "pokemon-tcg-agent/fixed-league-evaluation-v1",
        "configuration": {
            "agent_a": str(agent_a),
            "agent_b": str(agent_b),
            "deck_a": str(deck_a_path),
            "deck_b": str(deck_b_path),
            "name_a": name_a,
            "name_b": name_b,
            "games": args.games,
            "seed_start": args.seed_start,
            "seed_policy": "同一seedを先後入替の2試合へ使用",
            "game_timeout_seconds": args.game_timeout_seconds,
            "agent_a_options": options_a,
            "agent_b_options": options_b,
        },
        "summary": asdict(summary),
        "games": [asdict(game) for game in games],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
