"""Continuously train match agents by random two-agent generations.

Each generation:

1. Re-scan agents/match_agents for subdirectories containing deck.csv.
2. Pick two distinct agents at random.
3. Play two training games with first/second order swapped.
4. Train both selected agents from that generation's samples.
5. Save updated model.pth and per-agent CSV logs.

This is intentionally a long-running process. Use --max-generations for local
smoke tests or scheduled short runs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import random
from pathlib import Path
import signal
import sys
import time

import torch

from train import TrainStats, train_one_iteration
from train_match_agents import (
    AgentState,
    REPO_ROOT,
    load_state_if_available,
    play_training_game,
    read_deck,
    result_for_slots,
    save_checkpoint,
    save_final_model,
)


@dataclass(frozen=True)
class GenerationResult:
    generation: int
    game_index: int
    agent: str
    opponent: str
    first_player: str
    result: int
    elapsed_seconds: float


class StopRequested:
    value = False


def request_stop(_signum, _frame) -> None:
    StopRequested.value = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents-root",
        type=Path,
        default=REPO_ROOT / "agents" / "match_agents",
        help="Directory containing one subdirectory per match agent.",
    )
    parser.add_argument("--model-name", default="model.pth")
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=REPO_ROOT / "agents" / "rl_mcts" / "src" / "model.pth",
        help="Initial weights used when a new agent has no model.pth.",
    )
    parser.add_argument("--no-initial-model", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing per-agent model.pth files.")
    parser.add_argument("--search-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-value", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=30.0,
        help="Sleep time when fewer than two valid agents are available.",
    )
    parser.add_argument(
        "--generation-sleep",
        type=float,
        default=0.0,
        help="Optional sleep after each completed generation.",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Stop after this many generations. Omit to run forever.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save checkpoints every N generations. 0 disables extra checkpoints.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Checkpoint run directory. Defaults to server_YYYYmmdd_HHMMSS.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "agents" / "match_agents" / "train" / "checkpoints",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_agent_dirs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Agents root not found: {root}")
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "deck.csv").exists())


def load_agent_state(
    agent_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> AgentState:
    from rl_mcts.model import create_model

    model_path = agent_dir / args.model_name
    model = create_model().to(device)
    initial_model = None if args.no_initial_model else args.initial_model
    loaded_from = load_state_if_available(model, model_path, initial_model, device, args.fresh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    state = AgentState(
        name=agent_dir.name,
        root=agent_dir,
        deck=read_deck(agent_dir / "deck.csv"),
        model=model,
        optimizer=optimizer,
        model_path=model_path,
    )
    print(f"Loaded agent {state.name}: deck={agent_dir / 'deck.csv'} model={loaded_from}", flush=True)
    return state


def sync_agents(
    agents: dict[str, AgentState],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, AgentState]:
    discovered_dirs = discover_agent_dirs(args.agents_root)
    discovered_names = {path.name for path in discovered_dirs}

    for removed_name in sorted(set(agents) - discovered_names):
        print(f"Agent removed from active set: {removed_name}", flush=True)
        agents.pop(removed_name)

    for agent_dir in discovered_dirs:
        if agent_dir.name not in agents:
            try:
                agents[agent_dir.name] = load_agent_state(agent_dir, args, device)
            except Exception as exc:  # noqa: BLE001 - partial agent copies should not stop the server
                print(
                    f"Skipping agent {agent_dir.name}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    return agents


def reset_generation(agent: AgentState) -> None:
    agent.samples = []
    agent.wins = 0
    agent.losses = 0
    agent.draws = 0


def result_label(result_for_agent: int) -> str:
    if result_for_agent == 0:
        return "win"
    if result_for_agent == 1:
        return "loss"
    if result_for_agent == 2:
        return "draw"
    return "unknown"


def record_agent_result(agent: AgentState, result_for_agent: int) -> None:
    if result_for_agent == 0:
        agent.wins += 1
    elif result_for_agent == 1:
        agent.losses += 1
    else:
        agent.draws += 1


def append_csv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_match_result(row: GenerationResult) -> None:
    agent_dir = row_path_agent_dir(row.agent)
    path = agent_dir / "train" / "logs" / "match_results.csv"
    append_csv(
        path,
        [
            "generation",
            "game_index",
            "agent",
            "opponent",
            "first_player",
            "result",
            "elapsed_seconds",
        ],
        {
            "generation": row.generation,
            "game_index": row.game_index,
            "agent": row.agent,
            "opponent": row.opponent,
            "first_player": row.first_player,
            "result": result_label(row.result),
            "elapsed_seconds": row.elapsed_seconds,
        },
    )


_ACTIVE_AGENT_DIRS: dict[str, Path] = {}


def row_path_agent_dir(agent_name: str) -> Path:
    return _ACTIVE_AGENT_DIRS[agent_name]


def append_train_metrics(
    agent: AgentState,
    generation: int,
    opponent: str,
    stats: TrainStats,
    elapsed_seconds: float,
    checkpoint_path: Path | None,
) -> None:
    append_csv(
        agent.root / "train" / "logs" / "train_metrics.csv",
        [
            "generation",
            "agent",
            "opponent",
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
        ],
        {
            "generation": generation,
            "agent": agent.name,
            "opponent": opponent,
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
            "elapsed_seconds": elapsed_seconds,
            "checkpoint_path": checkpoint_path or "",
            "model_path": agent.model_path,
        },
    )


def play_generation(
    generation: int,
    agent_a: AgentState,
    agent_b: AgentState,
    args: argparse.Namespace,
    device: torch.device,
    run_name: str,
) -> None:
    started = time.time()
    reset_generation(agent_a)
    reset_generation(agent_b)
    agent_a.model.eval()
    agent_b.model.eval()

    print(f"Generation {generation}: {agent_a.name} vs {agent_b.name}", flush=True)
    with torch.inference_mode():
        for game_index, first_is_a in enumerate((True, False), start=1):
            first = agent_a if first_is_a else agent_b
            second = agent_b if first_is_a else agent_a
            game_started = time.time()
            result = play_training_game(first, second, args.search_count, args.lambda_value)
            elapsed = time.time() - game_started
            result_a, result_b = result_for_slots(result, first_is_a)
            record_agent_result(agent_a, result_a)
            record_agent_result(agent_b, result_b)
            first_player = first.name

            append_match_result(
                GenerationResult(
                    generation=generation,
                    game_index=game_index,
                    agent=agent_a.name,
                    opponent=agent_b.name,
                    first_player=first_player,
                    result=result_a,
                    elapsed_seconds=elapsed,
                )
            )
            append_match_result(
                GenerationResult(
                    generation=generation,
                    game_index=game_index,
                    agent=agent_b.name,
                    opponent=agent_a.name,
                    first_player=first_player,
                    result=result_b,
                    elapsed_seconds=elapsed,
                )
            )
            print(
                f"  game {game_index}/2 first={first_player} "
                f"result={agent_a.name}:{result_label(result_a)} {agent_b.name}:{result_label(result_b)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    checkpoint_a = checkpoint_b = None
    if args.checkpoint_every > 0 and generation % args.checkpoint_every == 0:
        checkpoint_a = save_checkpoint(agent_a, args.checkpoint_dir, run_name, generation)
        checkpoint_b = save_checkpoint(agent_b, args.checkpoint_dir, run_name, generation)

    train_started = time.time()
    stats_a = train_one_iteration(agent_a.model, agent_a.optimizer, agent_a.samples, args.batch_size, device)
    stats_b = train_one_iteration(agent_b.model, agent_b.optimizer, agent_b.samples, args.batch_size, device)
    save_final_model(agent_a)
    save_final_model(agent_b)
    elapsed_total = time.time() - started
    elapsed_train = time.time() - train_started

    append_train_metrics(agent_a, generation, agent_b.name, stats_a, elapsed_total, checkpoint_a)
    append_train_metrics(agent_b, generation, agent_a.name, stats_b, elapsed_total, checkpoint_b)
    print(
        f"  trained {agent_a.name}: samples={len(agent_a.samples)} loss={stats_a.loss:.6f} "
        f"batches={stats_a.batches}",
        flush=True,
    )
    print(
        f"  trained {agent_b.name}: samples={len(agent_b.samples)} loss={stats_b.loss:.6f} "
        f"batches={stats_b.batches}",
        flush=True,
    )
    print(f"Generation {generation} complete in {elapsed_total:.1f}s (train={elapsed_train:.1f}s)", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_generations is not None and args.max_generations < 1:
        raise SystemExit("--max-generations must be at least 1 when specified.")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be 0 or greater.")
    if args.idle_seconds < 0 or args.generation_sleep < 0:
        raise SystemExit("sleep values must be 0 or greater.")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or time.strftime("server_%Y%m%d_%H%M%S")
    rng = random.Random(args.seed)
    agents: dict[str, AgentState] = {}
    generation = 0

    print(f"Match training server started. agents_root={args.agents_root} device={device}", flush=True)
    print("Press Ctrl+C to stop after the current generation.", flush=True)

    while not StopRequested.value:
        agents = sync_agents(agents, args, device)
        _ACTIVE_AGENT_DIRS.clear()
        _ACTIVE_AGENT_DIRS.update({name: agent.root for name, agent in agents.items()})

        active_agents = sorted(agents.values(), key=lambda agent: agent.name)
        if len(active_agents) < 2:
            print(
                f"Waiting for at least two agents under {args.agents_root}. "
                f"Currently found {len(active_agents)}.",
                flush=True,
            )
            if args.dry_run:
                return
            time.sleep(args.idle_seconds)
            continue

        if args.dry_run:
            names = ", ".join(agent.name for agent in active_agents)
            print(f"Dry run agents: {names}", flush=True)
            agent_a, agent_b = rng.sample(active_agents, 2)
            print(f"Dry run next generation: {agent_a.name} vs {agent_b.name}", flush=True)
            return

        generation += 1
        agent_a, agent_b = rng.sample(active_agents, 2)
        play_generation(generation, agent_a, agent_b, args, device, run_name)

        if args.max_generations is not None and generation >= args.max_generations:
            break
        if args.generation_sleep > 0:
            time.sleep(args.generation_sleep)

    print("Match training server stopped.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
