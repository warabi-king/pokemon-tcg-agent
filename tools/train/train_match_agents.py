"""Train multiple rl_mcts agents by playing them against each other.

Each subdirectory under agents/match_agents is treated as one learner. A learner
owns one deck.csv and one model.pth. The model architecture and MCTS code are
loaded from agents/rl_mcts/src.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass, field
import itertools
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_MCTS_SRC_ROOT = REPO_ROOT / "agents" / "rl_mcts" / "src"
sys.path.insert(0, str(RL_MCTS_SRC_ROOT))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402
from rl_mcts.mcts import LearnSample, mcts_agent  # noqa: E402
from rl_mcts.model import create_model  # noqa: E402

from train import TrainStats, raise_for_deck_error, train_one_iteration  # noqa: E402

DECK_SIZE = 60


@dataclass
class AgentState:
    name: str
    root: Path
    deck: list[int]
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    model_path: Path
    samples: list[LearnSample] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return 100.0 * self.wins / decided if decided else 0.0

    def reset_iteration(self) -> None:
        self.samples = []
        self.wins = 0
        self.losses = 0
        self.draws = 0


@dataclass
class PairStats:
    iteration: int
    agent: str
    opponent: str
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return 100.0 * self.wins / decided if decided else 0.0


@dataclass(frozen=True)
class WorkerAgentSpec:
    name: str
    deck: list[int]
    model_path: Path


@dataclass(frozen=True)
class WorkerGameRequest:
    iteration: int
    game_index: int
    agent_a: WorkerAgentSpec
    agent_b: WorkerAgentSpec
    search_count: int
    lambda_value: float
    device: str


@dataclass
class WorkerGameResult:
    iteration: int
    game_index: int
    agent_a: str
    agent_b: str
    result_a: int
    result_b: int
    samples_a: list[LearnSample]
    samples_b: list[LearnSample]


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != DECK_SIZE:
        raise ValueError(f"{path} must contain {DECK_SIZE} card IDs, got {len(deck)}")
    return deck


def discover_agent_dirs(root: Path, names: list[str] | None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Agents root not found: {root}")
    if names:
        dirs = [root / name for name in names]
        missing = [path for path in dirs if not (path / "deck.csv").exists()]
        if missing:
            missing_text = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing deck.csv under: {missing_text}")
    else:
        dirs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "deck.csv").exists())
    if len(dirs) < 2:
        raise ValueError("At least two match agents are required.")
    return dirs


def load_state_if_available(
    model: torch.nn.Module,
    local_model: Path,
    initial_model: Path | None,
    device: torch.device,
    fresh: bool,
) -> str:
    if not fresh and local_model.exists():
        state = torch.load(local_model, map_location=device)
        model.load_state_dict(state)
        return str(local_model)
    if initial_model is not None and initial_model.exists():
        state = torch.load(initial_model, map_location=device)
        model.load_state_dict(state)
        return str(initial_model)
    return "random-init"


def load_agents(args: argparse.Namespace, device: torch.device) -> list[AgentState]:
    agent_dirs = discover_agent_dirs(args.agents_root, args.agent)
    agents: list[AgentState] = []
    initial_model = None if args.no_initial_model else args.initial_model
    for agent_dir in agent_dirs:
        model_path = agent_dir / args.model_name
        model = create_model().to(device)
        loaded_from = load_state_if_available(
            model,
            model_path,
            initial_model,
            device,
            args.fresh,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        state = AgentState(
            name=agent_dir.name,
            root=agent_dir,
            deck=read_deck(agent_dir / "deck.csv"),
            model=model,
            optimizer=optimizer,
            model_path=model_path,
        )
        agents.append(state)
        print(f"Loaded {state.name}: deck={agent_dir / 'deck.csv'} model={loaded_from}")
    return agents


def add_labeled_samples(
    destination: list[LearnSample],
    samples: list[LearnSample],
    result: int,
    player_index: int,
    lambda_value: float,
) -> None:
    if result == 2:
        value = 0.0
    else:
        value = 1.0 if player_index == result else -1.0

    for sample in reversed(samples):
        label = (value + sample.value) * 0.5
        value = value * lambda_value + sample.value * (1.0 - lambda_value)
        sample.value = label
        destination.append(sample)


def record_result(agent: AgentState, result_for_agent: int) -> None:
    if result_for_agent == 0:
        agent.wins += 1
    elif result_for_agent == 1:
        agent.losses += 1
    else:
        agent.draws += 1


def update_pair_stats(pair_stats: dict[tuple[str, str], PairStats], agent: str, opponent: str, result: int) -> None:
    stats = pair_stats[(agent, opponent)]
    stats.games += 1
    if result == 0:
        stats.wins += 1
    elif result == 1:
        stats.losses += 1
    else:
        stats.draws += 1


def result_for_slots(result: int, first_is_a: bool) -> tuple[int, int]:
    if result == 2:
        return 2, 2
    if (result == 0 and first_is_a) or (result == 1 and not first_is_a):
        return 0, 1
    return 1, 0


def play_training_game(
    first: AgentState,
    second: AgentState,
    search_count: int,
    lambda_value: float,
) -> int:
    slots = [first, second]
    decks = [first.deck, second.deck]
    per_slot_samples: list[list[LearnSample]] = [[], []]

    obs, start_data = battle_start(decks[0], decks[1])
    try:
        raise_for_deck_error(start_data)
        while obs["current"]["result"] < 0:
            player_index = obs["current"]["yourIndex"]
            current = slots[player_index]
            selected, sample = mcts_agent(
                obs,
                decks[player_index],
                current.model,
                search_count=search_count,
            )
            if sample is not None:
                per_slot_samples[player_index].append(sample)
            obs = battle_select(selected)
    finally:
        battle_finish()

    result = obs["current"]["result"]
    for player_index, agent in enumerate(slots):
        add_labeled_samples(
            agent.samples,
            per_slot_samples[player_index],
            result,
            player_index,
            lambda_value,
        )
    return result


def train_pair(
    iteration: int,
    agent_a: AgentState,
    agent_b: AgentState,
    games: int,
    search_count: int,
    lambda_value: float,
    pair_stats: dict[tuple[str, str], PairStats],
) -> None:
    for game_index in range(games):
        if agent_a is agent_b:
            result = play_training_game(agent_a, agent_b, search_count, lambda_value)
            agent_a.draws += 1
            update_pair_stats(pair_stats, agent_a.name, agent_a.name, 2)
            print(
                f"iteration={iteration} pair={agent_a.name}-{agent_a.name} "
                f"game={game_index + 1}/{games} seat_result={result}"
            )
            continue

        if game_index % 2 == 0:
            first, second = agent_a, agent_b
            first_is_a = True
        else:
            first, second = agent_b, agent_a
            first_is_a = False

        result = play_training_game(first, second, search_count, lambda_value)
        result_a, result_b = result_for_slots(result, first_is_a)

        record_result(agent_a, result_a)
        record_result(agent_b, result_b)
        update_pair_stats(pair_stats, agent_a.name, agent_b.name, result_a)
        update_pair_stats(pair_stats, agent_b.name, agent_a.name, result_b)
        print(
            f"iteration={iteration} pair={agent_a.name}-{agent_b.name} "
            f"game={game_index + 1}/{games} result={agent_a.name}:{result_a} {agent_b.name}:{result_b}"
        )


def _load_worker_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = create_model().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _play_worker_training_game(request: WorkerGameRequest) -> WorkerGameResult:
    device = torch.device(request.device)
    model_a = _load_worker_model(request.agent_a.model_path, device)
    if request.agent_a.name == request.agent_b.name:
        model_b = model_a
    else:
        model_b = _load_worker_model(request.agent_b.model_path, device)

    first_is_a = request.game_index % 2 == 0
    if request.agent_a.name == request.agent_b.name:
        first_is_a = True

    slots = [
        (request.agent_a, model_a),
        (request.agent_b, model_b),
    ]
    if not first_is_a:
        slots.reverse()

    decks = [slots[0][0].deck, slots[1][0].deck]
    models = [slots[0][1], slots[1][1]]
    per_slot_samples: list[list[LearnSample]] = [[], []]

    with torch.inference_mode():
        obs, start_data = battle_start(decks[0], decks[1])
        try:
            raise_for_deck_error(start_data)
            while obs["current"]["result"] < 0:
                player_index = obs["current"]["yourIndex"]
                selected, sample = mcts_agent(
                    obs,
                    decks[player_index],
                    models[player_index],
                    search_count=request.search_count,
                )
                if sample is not None:
                    per_slot_samples[player_index].append(sample)
                obs = battle_select(selected)
        finally:
            battle_finish()

    result = obs["current"]["result"]
    result_a, result_b = result_for_slots(result, first_is_a)
    samples_a: list[LearnSample] = []
    samples_b: list[LearnSample] = []

    for player_index, samples in enumerate(per_slot_samples):
        destination = samples_a
        agent_name = slots[player_index][0].name
        if agent_name == request.agent_b.name and request.agent_a.name != request.agent_b.name:
            destination = samples_b
        add_labeled_samples(
            destination,
            samples,
            result,
            player_index,
            request.lambda_value,
        )

    if request.agent_a.name == request.agent_b.name:
        samples_a.extend(samples_b)
        samples_b = []

    return WorkerGameResult(
        iteration=request.iteration,
        game_index=request.game_index,
        agent_a=request.agent_a.name,
        agent_b=request.agent_b.name,
        result_a=result_a,
        result_b=result_b,
        samples_a=samples_a,
        samples_b=samples_b,
    )


def apply_worker_result(
    agents_by_name: dict[str, AgentState],
    pair_stats: dict[tuple[str, str], PairStats],
    result: WorkerGameResult,
) -> None:
    agent_a = agents_by_name[result.agent_a]
    agent_b = agents_by_name[result.agent_b]
    agent_a.samples.extend(result.samples_a)
    agent_b.samples.extend(result.samples_b)

    if agent_a is agent_b:
        agent_a.draws += 1
        update_pair_stats(pair_stats, agent_a.name, agent_a.name, 2)
        print(
            f"iteration={result.iteration} pair={agent_a.name}-{agent_a.name} "
            f"game={result.game_index + 1} result=self"
        )
        return

    record_result(agent_a, result.result_a)
    record_result(agent_b, result.result_b)
    update_pair_stats(pair_stats, agent_a.name, agent_b.name, result.result_a)
    update_pair_stats(pair_stats, agent_b.name, agent_a.name, result.result_b)
    print(
        f"iteration={result.iteration} pair={agent_a.name}-{agent_b.name} "
        f"game={result.game_index + 1} result={agent_a.name}:{result.result_a} {agent_b.name}:{result.result_b}"
    )


def collect_training_games_parallel(
    iteration: int,
    agents: list[AgentState],
    scheduled_pairs: list[tuple[int, int]],
    checkpoint_paths: dict[str, Path],
    games_per_pair: int,
    search_count: int,
    lambda_value: float,
    pair_stats: dict[tuple[str, str], PairStats],
    workers: int,
    worker_device: str,
) -> None:
    agents_by_name = {agent.name: agent for agent in agents}
    specs = {
        agent.name: WorkerAgentSpec(
            name=agent.name,
            deck=agent.deck,
            model_path=checkpoint_paths[agent.name],
        )
        for agent in agents
    }
    requests = [
        WorkerGameRequest(
            iteration=iteration,
            game_index=game_index,
            agent_a=specs[agents[i].name],
            agent_b=specs[agents[j].name],
            search_count=search_count,
            lambda_value=lambda_value,
            device=worker_device,
        )
        for i, j in scheduled_pairs
        for game_index in range(games_per_pair)
    ]
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
        future_map = {executor.submit(_play_worker_training_game, request): request for request in requests}
        for future in as_completed(future_map):
            apply_worker_result(agents_by_name, pair_stats, future.result())


def append_agent_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "agent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "samples",
        "batches",
        "loss",
        "loss_value",
        "loss_policy",
        "elapsed_seconds",
        "checkpoint_path",
        "model_path",
    ]
    write_header = not metrics_path.exists()
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_pair_metrics(pair_metrics_path: Path, rows: list[PairStats]) -> None:
    pair_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "agent",
        "opponent",
        "games",
        "wins",
        "losses",
        "draws",
        "win_rate",
    ]
    write_header = not pair_metrics_path.exists()
    with pair_metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "iteration": row.iteration,
                    "agent": row.agent,
                    "opponent": row.opponent,
                    "games": row.games,
                    "wins": row.wins,
                    "losses": row.losses,
                    "draws": row.draws,
                    "win_rate": row.win_rate,
                }
            )


def make_pair_stats(iteration: int, agents: list[AgentState], include_self: bool) -> dict[tuple[str, str], PairStats]:
    stats: dict[tuple[str, str], PairStats] = {}
    for agent in agents:
        for opponent in agents:
            if not include_self and agent.name == opponent.name:
                continue
            stats[(agent.name, opponent.name)] = PairStats(iteration, agent.name, opponent.name)
    return stats


def save_state_dict_atomic(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    torch.save(state_dict, temporary_path)
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            if path.exists():
                path.chmod(0o666)
            temporary_path.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.2 * (attempt + 1))

    try:
        if path.exists():
            path.unlink()
        temporary_path.replace(path)
        return
    except PermissionError as error:
        last_error = error

    raise PermissionError(
        f"Could not replace model file after retries: {temporary_path} -> {path}. "
        "Another process may still be holding the target file open."
    ) from last_error


def save_checkpoint(agent: AgentState, checkpoint_dir: Path, run_name: str, iteration: int) -> Path:
    path = checkpoint_dir / agent.name / f"model_{iteration}.pth"
    if run_name:
        path = checkpoint_dir / run_name / agent.name / f"model_{iteration}.pth"
    save_state_dict_atomic(agent.model.state_dict(), path)
    return path


def save_final_model(agent: AgentState) -> None:
    save_state_dict_atomic(agent.model.state_dict(), agent.model_path)


def resolve_worker_count(workers: int, total_games: int) -> int:
    if workers < 0:
        raise ValueError("--workers must be 0 or greater")
    if total_games <= 1:
        return 1
    if workers == 0:
        return min(os.cpu_count() or 1, total_games)
    return min(workers, total_games)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents-root",
        type=Path,
        default=REPO_ROOT / "agents" / "match_agents",
        help="Directory containing one subdirectory per match agent.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=None,
        help="Train only this subdirectory name. Can be passed multiple times.",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-pair", type=int, default=2)
    parser.add_argument("--search-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument("--include-self", action="store_true", help="Also train each agent against itself.")
    parser.add_argument("--shuffle-pairs", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-name", default="model.pth")
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=REPO_ROOT / "agents" / "rl_mcts" / "src" / "model.pth",
        help="Initial weights used when an agent directory has no model.pth.",
    )
    parser.add_argument("--no-initial-model", action="store_true", help="Use random weights for missing models.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing per-agent model.pth files.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "agents" / "match_agents" / "train" / "checkpoints",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Checkpoint run directory name. Defaults to a timestamp to avoid overwriting old checkpoints.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO_ROOT / "agents" / "match_agents" / "train" / "logs",
    )
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--pair-metrics-file", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for game collection. Use 0 for auto.",
    )
    parser.add_argument(
        "--worker-device",
        default="cpu",
        help="Torch device used by worker processes during game collection.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print the schedule without training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.games_per_pair < 1:
        raise ValueError("--games-per-pair must be at least 1")
    if args.workers < 0:
        raise ValueError("--workers must be 0 or greater")

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = args.metrics_file or args.log_dir / "train_metrics.csv"
    pair_metrics_path = args.pair_metrics_file or args.log_dir / "pair_metrics.csv"
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")

    agents = load_agents(args, device)
    pair_indices = list(itertools.combinations(range(len(agents)), 2))
    if args.include_self:
        pair_indices.extend((i, i) for i in range(len(agents)))
    print(
        f"Training {len(agents)} agents for {args.iterations} iterations, "
        f"{len(pair_indices)} pairs, {args.games_per_pair} games per pair."
    )
    total_games = len(pair_indices) * args.games_per_pair
    workers = resolve_worker_count(args.workers, total_games)
    print(f"Game collection workers: {workers} (worker_device={args.worker_device})")

    if args.dry_run:
        for i, j in pair_indices:
            print(f"DRY RUN pair: {agents[i].name} vs {agents[j].name}")
        return

    for iteration in range(args.iterations):
        started = time.time()
        for agent in agents:
            agent.reset_iteration()
            agent.model.eval()

        checkpoint_paths = {
            agent.name: save_checkpoint(agent, args.checkpoint_dir, run_name, iteration)
            for agent in agents
        }

        pair_stats = make_pair_stats(iteration, agents, args.include_self)
        scheduled_pairs = pair_indices.copy()
        if args.shuffle_pairs:
            rng.shuffle(scheduled_pairs)

        if workers == 1:
            with torch.inference_mode():
                for i, j in scheduled_pairs:
                    train_pair(
                        iteration,
                        agents[i],
                        agents[j],
                        args.games_per_pair,
                        args.search_count,
                        args.lambda_value,
                        pair_stats,
                    )
        else:
            collect_training_games_parallel(
                iteration=iteration,
                agents=agents,
                scheduled_pairs=scheduled_pairs,
                checkpoint_paths=checkpoint_paths,
                games_per_pair=args.games_per_pair,
                search_count=args.search_count,
                lambda_value=args.lambda_value,
                pair_stats=pair_stats,
                workers=workers,
                worker_device=args.worker_device,
            )

        for agent in agents:
            print(f"Training {agent.name}: samples={len(agent.samples)} games={agent.games}")
            stats: TrainStats = train_one_iteration(
                agent.model,
                agent.optimizer,
                agent.samples,
                args.batch_size,
                device,
            )
            elapsed = time.time() - started
            append_agent_metrics(
                metrics_path,
                {
                    "iteration": iteration,
                    "agent": agent.name,
                    "games": agent.games,
                    "wins": agent.wins,
                    "losses": agent.losses,
                    "draws": agent.draws,
                    "win_rate": agent.win_rate,
                    "samples": len(agent.samples),
                    "batches": stats.batches,
                    "loss": stats.loss,
                    "loss_value": stats.loss_value,
                    "loss_policy": stats.loss_policy,
                    "elapsed_seconds": elapsed,
                    "checkpoint_path": checkpoint_paths[agent.name],
                    "model_path": agent.model_path,
                },
            )
            print(
                f"Finished {agent.name}: win_rate={agent.win_rate:.1f}% "
                f"loss={stats.loss:.6f} batches={stats.batches}"
            )

        append_pair_metrics(pair_metrics_path, list(pair_stats.values()))
        for agent in agents:
            save_final_model(agent)
        print(f"Iteration {iteration} saved metrics: {metrics_path}")

    if args.plot:
        from plot_match_metrics import plot_match_metrics

        plot_match_metrics(metrics_path, pair_metrics_path, args.log_dir)


if __name__ == "__main__":
    main()
