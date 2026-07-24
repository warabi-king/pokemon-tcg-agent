"""中央管理のラウンドロビン学習実行スクリプト。

このスクリプトは各エージェントを一つのプロセス内で扱い、
各iterationごとに全組合せ（自分との対戦含む）で対戦を回して
サンプルを収集し、その後全エージェントをまとめて学習（update）します。

注意: 各エージェントの実装は `agent_dir/src/rl_mcts/*` と `agent_dir/src/cg/*` を
個別にロードして利用します。各エージェントの `create_model()` と
`mcts_agent()` のインターフェースは train.py と互換であることを前提とします。

使い方例:
  python tools/run_train_round_robin.py --agent agents/rl_mcts_r_robin1 --agent agents/rl_mcts_r_robin2 --iterations 3 --games 50
"""

from __future__ import annotations

import argparse
import itertools
import importlib
import importlib.util
import json
import random
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    import torch
except Exception:
    print("Missing dependency: 'torch' is not installed in this Python environment.")
    print("Activate your virtualenv or install dependencies, e.g.:")
    print("  python3 -m pip install -r requirements.txt")
    raise

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "train_round_robin_central"


@dataclass
class TrainSpec:
    name: str
    agent_dir: Path


@dataclass
class TrainStats:
    batches: int
    loss: float
    loss_value: float
    loss_policy: float


def resolve_agent_dir(p: Path) -> Path | None:
    p = p.expanduser()
    if p.exists():
        if p.is_file():
            if p.name == "main.py":
                return p.parent.parent if (p.parent / "deck.csv").exists() or (p.parent / "cg").exists() else p.parent
            if p.name == "src":
                return p.parent if (p / "main.py").exists() else None
            return p.parent
        if p.is_dir():
            if (p / "src" / "main.py").exists():
                return p
            if (p / "train" / "train.py").exists():
                return p
            if (p / "deck.csv").exists() and (p / "cg").exists():
                return p
    for parent in p.parents:
        if (parent / "src" / "main.py").exists() or (parent / "train" / "train.py").exists():
            return parent
    return None


def parse_agent_arg(raw: str) -> TrainSpec:
    if "=" in raw:
        name, path = raw.split("=", 1)
        name = name.strip()
        p = Path(path.strip())
    else:
        p = Path(raw.strip())
        name = p.name
    agent_dir = resolve_agent_dir(p)
    if agent_dir is None:
        raise argparse.ArgumentTypeError(f"--agent {raw}: 有効な agent ルートが見つかりません。src/main.py を含む agent ルートを指定してください。")
    return TrainSpec(name=name, agent_dir=agent_dir.resolve())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", action="append", type=parse_agent_arg, default=None, help="name=path か path の形式でエージェント指定")
    p.add_argument("--config", type=Path, default=None, help="JSON config file with agent entries")
    p.add_argument("--dry-run", action="store_true", help="DRY RUN: show planned actions only")
    p.add_argument("--iterations", type=int, default=5, help="学習iteration数")
    p.add_argument("--games", type=int, default=100, help="各iterationの対戦（self/cross）試合数")
    p.add_argument("--batch-size", type=int, default=128, help="学習バッチサイズ")
    p.add_argument("--search-count", type=int, default=10, help="MCTS探索回数")
    p.add_argument("--lr", type=float, default=3e-4, help="学習率")
    p.add_argument("--lambda-value", type=float, default=0.9, help="終局価値の逆向き更新率")
    p.add_argument("--eval-games", type=int, default=50, help="各iterationの評価試合数")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    return p.parse_args()


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure agent's src/ directory is on sys.path so imports like `import cg` resolve
    agent_src = path.parent.parent
    inserted = False
    try:
        if str(agent_src) not in sys.path:
            sys.path.insert(0, str(agent_src))
            inserted = True
        spec.loader.exec_module(mod)
    finally:
        if inserted:
            try:
                sys.path.remove(str(agent_src))
            except ValueError:
                pass
    return mod


def load_agent_api(spec: TrainSpec) -> dict:
    """Import agent packages by temporarily inserting agent/src into sys.path.

    This allows package-local relative imports (e.g., `from .sim import lib`) to work.
    """
    src = spec.agent_dir / "src"
    if not src.exists():
        raise ImportError(f"Agent src directory not found: {src}")

    inserted = False
    try:
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
            inserted = True

        model_mod = importlib.import_module("rl_mcts.model")
        mcts_mod = importlib.import_module("rl_mcts.mcts")
        deck_mod = importlib.import_module("rl_mcts.deck")
        cg_api_mod = importlib.import_module("cg.api")
        cg_game_mod = importlib.import_module("cg.game")

        api: dict = {}
        api["model_mod"] = model_mod
        api["mcts_mod"] = mcts_mod
        api["deck_mod"] = deck_mod
        api["cg_api_mod"] = cg_api_mod
        api["cg_game_mod"] = cg_game_mod

        api["create_model"] = getattr(model_mod, "create_model")
        api["mcts_agent"] = getattr(mcts_mod, "mcts_agent")
        api["LearnSample"] = getattr(mcts_mod, "LearnSample")
        api["LearnInput"] = getattr(mcts_mod, "LearnInput")
        api["MAX_ACTIONS"] = getattr(mcts_mod, "MAX_ACTIONS")
        api["read_deck_csv"] = getattr(deck_mod, "read_deck_csv")
        api["to_observation_class"] = getattr(cg_api_mod, "to_observation_class")
        api["battle_start"] = getattr(cg_game_mod, "battle_start")
        api["battle_select"] = getattr(cg_game_mod, "battle_select")
        api["battle_finish"] = getattr(cg_game_mod, "battle_finish")
        return api
    finally:
        if inserted:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass


def collect_self_play_samples_agent(api: dict, model, deck: list[int], games: int, search_count: int, lambda_value: float):
    samples = []
    for _ in range(games):
        obs, start_data = api["battle_start"](deck, deck)
        per_player = [[], []]
        while True:
            if obs["current"]["result"] >= 0:
                break
            selected, sample = api["mcts_agent"](obs, deck, model, search_count=search_count)
            if sample is not None:
                per_player[obs["current"]["yourIndex"]].append(sample)
            obs = api["battle_select"](selected)
        api["battle_finish"]()

        for player_index in range(2):
            if obs["current"]["result"] == 2:
                value = 0.0
            else:
                value = 1.0 if player_index == obs["current"]["result"] else -1.0
            for sample in reversed(per_player[player_index]):
                label = (value + sample.value) * 0.5
                value = value * lambda_value + sample.value * (1.0 - lambda_value)
                sample.value = label
                samples.append(sample)
    return samples


def collect_cross_play_samples_agent(api_a: dict, api_b: dict, model_a, model_b, deck_a: list[int], deck_b: list[int], games: int, search_count: int, lambda_value: float):
    samples_a = []
    samples_b = []
    for _ in range(games):
        obs, start_data = api_a["battle_start"](deck_a, deck_b)
        per_player = [[], []]
        while True:
            if obs["current"]["result"] >= 0:
                break
            if obs["current"]["yourIndex"] == 0:
                selected, sample = api_a["mcts_agent"](obs, deck_a, model_a, search_count=search_count)
            else:
                selected, sample = api_b["mcts_agent"](obs, deck_b, model_b, search_count=search_count)
            if sample is not None:
                per_player[obs["current"]["yourIndex"]].append(sample)
            obs = api_a["battle_select"](selected)
        api_a["battle_finish"]()

        for player_index in range(2):
            if obs["current"]["result"] == 2:
                value = 0.0
            else:
                value = 1.0 if player_index == obs["current"]["result"] else -1.0
            for sample in reversed(per_player[player_index]):
                label = (value + sample.value) * 0.5
                value = value * lambda_value + sample.value * (1.0 - lambda_value)
                sample.value = label
                if player_index == 0:
                    samples_a.append(sample)
                else:
                    samples_b.append(sample)
    return samples_a, samples_b


def train_one_iteration_local(model, optimizer, samples: list, batch_size: int, device: torch.device, api_mod: dict) -> TrainStats:
    if len(samples) < batch_size:
        return TrainStats(batches=0, loss=0.0, loss_value=0.0, loss_policy=0.0)

    model.train()
    random.shuffle(samples)
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    batch_count = len(samples) // batch_size
    loss_total = 0.0
    loss_value_total = 0.0
    loss_policy_total = 0.0
    LearnInput = api_mod["LearnInput"]
    MAX_ACTIONS = api_mod["MAX_ACTIONS"]

    for i in range(batch_count):
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask: list[float] = []
        label_enc: list[float] = []
        label_dec: list[float] = []
        start = batch_size * i

        for sample in samples[start : start + batch_size]:
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            mask.extend([1.0] * len(sample.policy))
            for _ in range(MAX_ACTIONS - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        label_tensor_enc = torch.tensor(label_enc, dtype=torch.float32, device=device).view(batch_size, -1)
        label_tensor_dec = torch.tensor(label_dec, dtype=torch.float32, device=device).view(batch_size, -1)

        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        )

        loss_enc = loss_fn_enc(out_enc, label_tensor_enc)
        loss_dec = loss_fn_dec(out_dec, label_tensor_dec)
        loss_dec = (loss_dec * mask_tensor).sum() / float(batch_size)
        loss = loss_enc + loss_dec
        loss.backward()
        optimizer.step()

        loss_total += float(loss.item())
        loss_value_total += float(loss_enc.item())
        loss_policy_total += float(loss_dec.item())

    return TrainStats(batches=batch_count, loss=loss_total / batch_count, loss_value=loss_value_total / batch_count, loss_policy=loss_policy_total / batch_count)


def append_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "eval_games",
        "eval_win",
        "eval_lose",
        "eval_draw",
        "eval_win_rate",
        "games",
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
        import csv

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_agent(api_mod: dict, model, deck: list[int], games: int, search_count: int) -> tuple[int, int, int]:
    results = [0, 0, 0]
    for i in range(games):
        obs, start_data = api_mod["battle_start"](deck, deck)
        your_index = i % 2
        while True:
            if obs["current"]["result"] >= 0:
                break
            if obs["current"]["yourIndex"] == your_index:
                selected, _ = api_mod["mcts_agent"](obs, deck, model, search_count=search_count)
            else:
                obs_cls = api_mod["to_observation_class"](obs)
                selected = random.sample(list(range(len(obs_cls.select.option))), obs_cls.select.maxCount)
            obs = api_mod["battle_select"](selected)
        api_mod["battle_finish"]()

        if obs["current"]["result"] == 2:
            results[2] += 1
        elif obs["current"]["result"] == your_index:
            results[0] += 1
        else:
            results[1] += 1
    return results[0], results[1], results[2]


def main() -> None:
    args = parse_args()

    specs: List[TrainSpec] = []
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))
        for entry in data:
            name = entry["name"]
            raw_path = Path(entry.get("train") or entry.get("path"))
            agent_dir = resolve_agent_dir(raw_path)
            if agent_dir is None:
                raise SystemExit(f"config entry {name}: エージェントルートが見つかりません")
            specs.append(TrainSpec(name=name, agent_dir=agent_dir.resolve()))
    if args.agent:
        specs.extend(args.agent)

    if len(specs) < 1:
        raise SystemExit("少なくとも1体のエージェントを指定してください。")

    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"エージェント名が重複しています: {names}")

    pairings = list(itertools.combinations_with_replacement(names, 2))

    print("Agents:")
    for spec in specs:
        print(f"  {spec.name}: {spec.agent_dir}")
    print(f"Total pairings: {len(pairings)}")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run: will perform central training with the following iterations and pairings:")
        print(f" iterations={args.iterations}, games={args.games}, batch_size={args.batch_size}")
        for name0, name1 in pairings:
            print(f" pairing: {name0} vs {name1}")
        return

    # setup agents
    timestamp = int(time.time())
    run_dir = RESULTS_ROOT / f"central_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    agents: dict[str, dict] = {}
    for spec in specs:
        api = load_agent_api(spec)
        model = api["create_model"]().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        try:
            deck = api["read_deck_csv"]()
        except Exception:
            deck_path = spec.agent_dir / "deck.csv"
            deck = [int(line.strip()) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        persistent = spec.agent_dir / "src" / "model.pth"
        if persistent.exists():
            state = torch.load(persistent, map_location=device)
            model.load_state_dict(state)

        ckpt_dir = run_dir / spec.name / "checkpoints"
        log_dir = run_dir / spec.name / "logs"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        agents[spec.name] = {
            "spec": spec,
            "api": api,
            "model": model,
            "optimizer": optimizer,
            "deck": deck,
            "ckpt_dir": ckpt_dir,
            "log_dir": log_dir,
            "persistent": persistent,
        }

    # iterations
    for iteration in range(args.iterations):
        iter_start = time.time()
        samples_accum = {name: [] for name in agents.keys()}

        for name0, name1 in pairings:
            a = agents[name0]
            b = agents[name1]
            if name0 == name1:
                s = collect_self_play_samples_agent(a["api"], a["model"], a["deck"], args.games, args.search_count, args.lambda_value)
                samples_accum[name0].extend(s)
            else:
                sa, sb = collect_cross_play_samples_agent(a["api"], b["api"], a["model"], b["model"], a["deck"], b["deck"], args.games, args.search_count, args.lambda_value)
                samples_accum[name0].extend(sa)
                samples_accum[name1].extend(sb)

        # train each agent
        for name, info in agents.items():
            model = info["model"]
            optimizer = info["optimizer"]
            api = info["api"]
            samples = samples_accum[name]

            cp = info["ckpt_dir"] / f"model_{iteration}.pth"
            torch.save(model.state_dict(), cp)

            if args.eval_games > 0:
                wa, la, da = evaluate_agent(api, model, info["deck"], args.eval_games, args.search_count)
                decided = wa + la
                win_rate = 100.0 * wa / decided if decided else 0.0
            else:
                wa = la = da = 0
                win_rate = 0.0

            stats = train_one_iteration_local(model, optimizer, samples, args.batch_size, device, api)
            elapsed = time.time() - iter_start

            metrics_path = info["log_dir"] / "train_metrics.csv"
            append_metrics(
                metrics_path,
                {
                    "iteration": iteration,
                    "eval_games": args.eval_games,
                    "eval_win": wa,
                    "eval_lose": la,
                    "eval_draw": da,
                    "eval_win_rate": win_rate,
                    "games": args.games,
                    "samples": len(samples),
                    "batches": stats.batches if stats else 0,
                    "loss": stats.loss if stats else 0.0,
                    "loss_value": stats.loss_value if stats else 0.0,
                    "loss_policy": stats.loss_policy if stats else 0.0,
                    "elapsed_seconds": elapsed,
                    "checkpoint_path": cp,
                    "model_path": info["persistent"],
                },
            )

            info["persistent"].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), info["persistent"])

        print(f"Completed iteration {iteration}")

    print("Central round-robin training finished.")


if __name__ == "__main__":
    main()
