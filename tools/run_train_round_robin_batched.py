"""中央管理・GPUバッチ推論対応のラウンドロビン学習実行スクリプト。

`tools/run_train_round_robin.py`（複数train.pyをサブプロセスとして呼ぶ従来版）とは別物です。
このスクリプトは各エージェントを一つのプロセス内で扱い、
各iterationごとに全組合せ（自分との対戦含む）で対戦を回して
サンプルを収集し、その後全エージェントをまとめて学習（update）します。
`--backend shared-batch`/`shared-cpu-batch`では、複数試合のNN評価を
`tools/batched_training.py`経由でバッチ化してGPUへ渡すため、自己対戦の
データ収集自体をGPUで効率よく回せます。

注意: 各エージェントの実装は `agent_dir/src/rl_mcts/*` と `agent_dir/src/cg/*` を
個別にロードして利用します。各エージェントの `create_model()` と
`mcts_agent()` のインターフェースは train.py と互換であることを前提とします。

使い方例:
  python tools/run_train_round_robin_batched.py --agent agents/rl_mcts_r_robin1 --agent agents/rl_mcts_r_robin2 --iterations 3 --games 50
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
from typing import Callable, List

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


@dataclass(frozen=True)
class TrainBatchProgress:
    batch: int
    batches: int
    batch_loss: float
    running_loss: float
    value_loss: float
    policy_loss: float


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
    p.add_argument(
        "--backend",
        choices=("sequential", "shared-batch", "shared-cpu-batch"),
        default="shared-batch",
        help="対戦サンプル収集方式（shared-cpu-batchは旧名称）",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="学習iteration数（target-loss指定時は最大iteration数）",
    )
    p.add_argument("--games", type=int, default=100, help="各iterationの対戦（self/cross）試合数")
    p.add_argument("--batch-size", type=int, default=128, help="学習バッチサイズ")
    p.add_argument(
        "--inference-batch-size",
        type=int,
        default=128,
        help="shared-cpu-batchのNN推論バッチ上限",
    )
    p.add_argument(
        "--lanes",
        type=int,
        default=128,
        help="shared-cpu-batchで同時進行する対戦数",
    )
    p.add_argument("--search-count", type=int, default=10, help="MCTS探索回数")
    p.add_argument("--lr", type=float, default=3e-4, help="学習率")
    p.add_argument("--lambda-value", type=float, default=0.9, help="終局価値の逆向き更新率")
    p.add_argument("--eval-games", type=int, default=50, help="各iterationの評価試合数")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    p.add_argument(
        "--inference-device",
        type=str,
        default="cpu",
        help="shared-cpu-batchの推論device",
    )
    p.add_argument("--seed", type=int, default=0, help="乱数seed")
    p.add_argument(
        "--convergence",
        choices=("auto", "iterations", "target", "plateau"),
        default="auto",
        help="停止条件。autoはtarget-loss指定時だけtarget、それ以外はiterations",
    )
    p.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="全agentのLossがこの値以下になったら早期終了",
    )
    p.add_argument(
        "--min-iterations",
        type=int,
        default=1,
        help="早期終了を判定し始めるiteration数",
    )
    p.add_argument(
        "--loss-patience",
        type=int,
        default=1,
        help="target-lossを連続して満たす必要があるiteration数",
    )
    p.add_argument(
        "--plateau-window",
        type=int,
        default=3,
        help="plateau判定に使う直近iteration数",
    )
    p.add_argument(
        "--plateau-delta",
        type=float,
        default=0.0015,
        help="各agentの直近Loss変動幅がこの値以下ならplateau候補",
    )
    p.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="全agentのplateauを連続して満たす必要がある判定回数",
    )
    p.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run directoryへLoss推移PNGを保存",
    )
    p.add_argument(
        "--live-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Notebook監視用のバッチLoss CSVと状態JSONを逐次保存",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="出力run directoryを固定（Notebookからの監視用）",
    )
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="検証用: agent配下のsrc/model.pthを上書きしない",
    )
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


def train_one_iteration_local(
    model,
    optimizer,
    samples: list,
    batch_size: int,
    device: torch.device,
    api_mod: dict,
    progress_callback: Callable[[TrainBatchProgress], None] | None = None,
) -> TrainStats:
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
        if progress_callback is not None:
            progress_callback(
                TrainBatchProgress(
                    batch=i + 1,
                    batches=batch_count,
                    batch_loss=float(loss.item()),
                    running_loss=loss_total / (i + 1),
                    value_loss=float(loss_enc.item()),
                    policy_loss=float(loss_dec.item()),
                )
            )

    return TrainStats(batches=batch_count, loss=loss_total / batch_count, loss_value=loss_value_total / batch_count, loss_policy=loss_policy_total / batch_count)


def append_metrics(metrics_path: Path, row: dict[str, object]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "backend",
        "device",
        "inference_device",
        "search_count",
        "lanes",
        "inference_batch_size",
        "eval_games",
        "eval_win",
        "eval_lose",
        "eval_draw",
        "eval_win_rate",
        "games",
        "completed_games",
        "failed_games",
        "samples",
        "batches",
        "loss",
        "loss_value",
        "loss_policy",
        "elapsed_seconds",
        "collect_seconds",
        "nn_evaluations",
        "nn_batches",
        "nn_mean_batch_size",
        "nn_max_batch_size",
        "nn_seconds",
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


def save_loss_artifacts(
    run_dir: Path,
    loss_history: dict[str, list[float]],
    target_loss: float | None,
    *,
    save_plot: bool,
) -> Path | None:
    """全agentのLoss履歴をCSVとPNGへ保存する。"""
    import csv

    names = list(loss_history)
    iteration_count = len(loss_history[names[0]])
    csv_path = run_dir / "loss_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["iteration", *names, "mean_loss", "max_loss"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(iteration_count):
            losses = [loss_history[name][index] for name in names]
            writer.writerow(
                {
                    "iteration": index,
                    **{name: loss_history[name][index] for name in names},
                    "mean_loss": sum(losses) / len(losses),
                    "max_loss": max(losses),
                }
            )

    if not save_plot:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_values = list(range(iteration_count))
    figure, axis = plt.subplots(figsize=(12, 7))
    for name in names:
        axis.plot(x_values, loss_history[name], marker="o", alpha=0.65, label=name)
    means = [
        sum(loss_history[name][index] for name in names) / len(names)
        for index in x_values
    ]
    axis.plot(x_values, means, color="black", linewidth=3, marker="o", label="mean")
    if target_loss is not None:
        axis.axhline(
            target_loss,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"target={target_loss:g}",
        )
    axis.set_title("Round-robin training loss")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Loss")
    axis.set_xticks(x_values)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3, fontsize=9)
    figure.tight_layout()
    plot_path = run_dir / "loss.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return plot_path


def loss_plateau_status(
    loss_history: dict[str, list[float]],
    *,
    window: int,
    delta: float,
) -> tuple[bool, dict[str, float]]:
    """全agentの直近Loss変動幅がdelta以下かを返す。"""
    if any(len(history) < window for history in loss_history.values()):
        return False, {}
    ranges = {
        name: max(history[-window:]) - min(history[-window:])
        for name, history in loss_history.items()
    }
    return all(value <= delta for value in ranges.values()), ranges


def validate_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDAを利用できません。")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPSを利用できません。")
    return device


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


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
    main_started = time.perf_counter()
    args = parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterationsは1以上で指定してください。")
    if args.games < 1:
        raise SystemExit("--gamesは1以上で指定してください。")
    if args.batch_size < 1 or args.inference_batch_size < 1:
        raise SystemExit("バッチサイズは1以上で指定してください。")
    if args.lanes < 1:
        raise SystemExit("--lanesは1以上で指定してください。")
    if args.target_loss is not None and args.target_loss <= 0:
        raise SystemExit("--target-lossは0より大きい値で指定してください。")
    if args.min_iterations < 1 or args.min_iterations > args.iterations:
        raise SystemExit("--min-iterationsは1以上かつ--iterations以下で指定してください。")
    if args.loss_patience < 1:
        raise SystemExit("--loss-patienceは1以上で指定してください。")
    if args.plateau_window < 2:
        raise SystemExit("--plateau-windowは2以上で指定してください。")
    if args.plateau_delta <= 0:
        raise SystemExit("--plateau-deltaは0より大きい値で指定してください。")
    if args.plateau_patience < 1:
        raise SystemExit("--plateau-patienceは1以上で指定してください。")

    convergence_mode = args.convergence
    if convergence_mode == "auto":
        convergence_mode = "target" if args.target_loss is not None else "iterations"
    if convergence_mode == "target" and args.target_loss is None:
        raise SystemExit("--convergence targetには--target-lossが必要です。")
    if convergence_mode == "plateau" and args.min_iterations < args.plateau_window:
        raise SystemExit(
            "plateau判定では--min-iterationsを--plateau-window以上にしてください。"
        )

    device = validate_device(args.device)
    inference_device = validate_device(args.inference_device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        print(
            f" backend={args.backend}, iterations={args.iterations}, "
            f"games={args.games}, batch_size={args.batch_size}, "
            f"inference_batch_size={args.inference_batch_size}, lanes={args.lanes}, "
            f"device={device}, inference_device={inference_device}, "
            f"convergence={convergence_mode}"
        )
        for name0, name1 in pairings:
            print(f" pairing: {name0} vs {name1}")
        return

    # setup agents
    if args.run_dir is None:
        timestamp = int(time.time())
        run_dir = RESULTS_ROOT / f"central_{timestamp}"
    else:
        run_dir = args.run_dir.expanduser().resolve()
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"run directoryが既に存在します: {run_dir}") from exc

    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "resolved_convergence": convergence_mode,
        "resolved_device": str(device),
        "resolved_inference_device": str(inference_device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "agents": [
            {"name": spec.name, "agent_dir": str(spec.agent_dir)} for spec in specs
        ],
        "pairings": len(pairings),
    }
    write_json(run_dir / "run_config.json", run_config)
    print(f"Training device: {device}")
    print(f"Inference device: {inference_device}")
    print(f"Convergence: {convergence_mode}")

    live_recorder = None
    if args.live_loss:
        from live_loss_recorder import LiveLossRecorder

        live_recorder = LiveLossRecorder(
            run_dir,
            names,
            target_loss=args.target_loss,
        )
        print(f"Live Loss CSV: {live_recorder.csv_path}")
        print(f"Live status: {live_recorder.status_path}")

    agents: dict[str, dict] = {}
    for spec in specs:
        api = load_agent_api(spec)
        model = api["create_model"]().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        # Pythonのmodule cacheにより複数agentのdeck moduleが共有されても、
        # 必ず指定agent自身のデッキを読む。
        deck_path = spec.agent_dir / "src" / "deck.csv"
        deck = [
            int(line.strip())
            for line in deck_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(deck) != 60:
            raise ValueError(
                f"deck.csvは60枚である必要があります: {deck_path} ({len(deck)}枚)"
            )

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

    loss_history: dict[str, list[float]] = {name: [] for name in agents}
    convergence_streak = 0
    converged = False
    convergence_reason = "max_iterations"
    plateau_ranges: dict[str, float] = {}
    total_games = 0
    total_samples = 0
    completed_iterations = 0

    # iterations
    for iteration in range(args.iterations):
        iter_start = time.time()
        collect_start = time.time()
        collect_profile = None
        completed_games = 0
        failed_games = 0
        if live_recorder is not None:
            live_recorder.update_status(
                "collecting",
                f"iteration {iteration} の対戦サンプルを収集中",
                iteration=iteration,
            )

        if args.backend in ("shared-batch", "shared-cpu-batch"):
            from batched_training import (
                BatchedTrainingAgent,
                collect_batched_training_samples,
            )

            output = collect_batched_training_samples(
                [
                    BatchedTrainingAgent(
                        name=name,
                        model=info["model"],
                        deck=info["deck"],
                    )
                    for name, info in agents.items()
                ],
                pairings,
                args.games,
                canonical_src=specs[0].agent_dir / "src",
                device=inference_device,
                batch_size=args.inference_batch_size,
                lanes=args.lanes,
                search_count=args.search_count,
                lambda_value=args.lambda_value,
                seed=args.seed + iteration,
            )
            samples_accum = output.samples
            collect_profile = output.profile
            completed_games = sum(result.error is None for result in output.results)
            failed = [result for result in output.results if result.error is not None]
            failed_games = len(failed)
            print(
                f"iteration {iteration}: collected {sum(map(len, samples_accum.values()))} "
                f"samples from {completed_games} games; "
                f"NN batch mean={output.profile.mean_batch_size:.2f}, "
                f"max={output.profile.max_batch_size}"
            )
            if failed:
                first = failed[0]
                raise RuntimeError(
                    f"共有libcg収集中に{len(failed)}試合が失敗しました。"
                    f"最初の失敗: {first.name0} vs {first.name1}: {first.error}"
                )
        else:
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
            completed_games = len(pairings) * args.games

        collect_seconds = time.time() - collect_start
        total_games += completed_games
        total_samples += sum(len(samples) for samples in samples_accum.values())

        # train each agent
        iteration_losses: dict[str, float] = {}
        iteration_batches: dict[str, int] = {}
        for name, info in agents.items():
            model = info["model"]
            optimizer = info["optimizer"]
            api = info["api"]
            samples = samples_accum[name]
            model.to(device)
            if live_recorder is not None:
                live_recorder.update_status(
                    "training",
                    f"{name} の学習を開始",
                    iteration=iteration,
                    agent=name,
                )

            before_cp = info["ckpt_dir"] / f"model_{iteration}_before.pth"
            torch.save(model.state_dict(), before_cp)

            if args.eval_games > 0:
                wa, la, da = evaluate_agent(api, model, info["deck"], args.eval_games, args.search_count)
                decided = wa + la
                win_rate = 100.0 * wa / decided if decided else 0.0
            else:
                wa = la = da = 0
                win_rate = 0.0

            progress_callback = None
            if live_recorder is not None:
                def progress_callback(
                    progress: TrainBatchProgress,
                    *,
                    agent_name: str = name,
                    iteration_index: int = iteration,
                ) -> None:
                    live_recorder.record_batch(
                        iteration=iteration_index,
                        agent=agent_name,
                        batch=progress.batch,
                        batches=progress.batches,
                        batch_loss=progress.batch_loss,
                        running_loss=progress.running_loss,
                        value_loss=progress.value_loss,
                        policy_loss=progress.policy_loss,
                    )

            stats = train_one_iteration_local(
                model,
                optimizer,
                samples,
                args.batch_size,
                device,
                api,
                progress_callback=progress_callback,
            )
            iteration_losses[name] = stats.loss
            iteration_batches[name] = stats.batches
            cp = info["ckpt_dir"] / f"model_{iteration}.pth"
            torch.save(model.state_dict(), cp)
            elapsed = time.time() - iter_start

            metrics_path = info["log_dir"] / "train_metrics.csv"
            append_metrics(
                metrics_path,
                {
                    "iteration": iteration,
                    "backend": args.backend,
                    "device": str(device),
                    "inference_device": str(inference_device),
                    "search_count": args.search_count,
                    "lanes": args.lanes,
                    "inference_batch_size": args.inference_batch_size,
                    "eval_games": args.eval_games,
                    "eval_win": wa,
                    "eval_lose": la,
                    "eval_draw": da,
                    "eval_win_rate": win_rate,
                    "games": args.games,
                    "completed_games": completed_games,
                    "failed_games": failed_games,
                    "samples": len(samples),
                    "batches": stats.batches if stats else 0,
                    "loss": stats.loss if stats else 0.0,
                    "loss_value": stats.loss_value if stats else 0.0,
                    "loss_policy": stats.loss_policy if stats else 0.0,
                    "elapsed_seconds": elapsed,
                    "collect_seconds": collect_seconds,
                    "nn_evaluations": collect_profile.nn_evaluations if collect_profile else 0,
                    "nn_batches": collect_profile.nn_batches if collect_profile else 0,
                    "nn_mean_batch_size": collect_profile.mean_batch_size if collect_profile else 0.0,
                    "nn_max_batch_size": collect_profile.max_batch_size if collect_profile else 0,
                    "nn_seconds": collect_profile.nn_seconds if collect_profile else 0.0,
                    "checkpoint_path": cp,
                    "model_path": info["persistent"],
                },
            )

            if not args.no_persist:
                info["persistent"].parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), info["persistent"])

        for name in agents:
            loss_history[name].append(iteration_losses[name])
        plot_path = save_loss_artifacts(
            run_dir,
            loss_history,
            args.target_loss,
            save_plot=args.plot,
        )
        mean_loss = sum(iteration_losses.values()) / len(iteration_losses)
        max_loss = max(iteration_losses.values())
        completed_iterations = iteration + 1
        print(
            f"Completed iteration {iteration}: mean_loss={mean_loss:.6f}, "
            f"max_loss={max_loss:.6f}"
        )
        if plot_path is not None:
            print(f"Loss plot: {plot_path}")
        if live_recorder is not None:
            live_recorder.update_status(
                "iteration_complete",
                f"iteration {iteration} 完了: mean_loss={mean_loss:.6f}",
                iteration=iteration,
            )

        if convergence_mode == "target" and iteration + 1 >= args.min_iterations:
            target_converged = all(
                iteration_batches[name] > 0
                and iteration_losses[name] <= args.target_loss
                for name in agents
            )
            convergence_streak = convergence_streak + 1 if target_converged else 0
            print(
                f"Loss convergence: {convergence_streak}/{args.loss_patience} "
                f"iterations (target <= {args.target_loss:g} for every agent)"
            )
            if convergence_streak >= args.loss_patience:
                converged = True
                convergence_reason = "target_loss"
                print(f"Target loss reached at iteration {iteration}.")
                break

        if convergence_mode == "plateau" and iteration + 1 >= args.min_iterations:
            plateau, plateau_ranges = loss_plateau_status(
                loss_history,
                window=args.plateau_window,
                delta=args.plateau_delta,
            )
            convergence_streak = convergence_streak + 1 if plateau else 0
            max_range = max(plateau_ranges.values(), default=float("inf"))
            print(
                f"Loss plateau: {convergence_streak}/{args.plateau_patience} "
                f"checks (window={args.plateau_window}, max_range={max_range:.6f}, "
                f"delta<={args.plateau_delta:g})"
            )
            if convergence_streak >= args.plateau_patience:
                converged = True
                convergence_reason = "loss_plateau"
                print(f"Loss plateau reached at iteration {iteration}.")
                break

    elapsed_seconds = time.perf_counter() - main_started
    if convergence_mode == "plateau":
        converged_loss = {
            name: sum(history[-args.plateau_window :]) / args.plateau_window
            for name, history in loss_history.items()
        }
    else:
        converged_loss = {
            name: history[-1] if history else 0.0
            for name, history in loss_history.items()
        }
    final_loss = {
        name: history[-1] if history else 0.0 for name, history in loss_history.items()
    }
    summary = {
        "converged": converged,
        "convergence_mode": convergence_mode,
        "convergence_reason": convergence_reason,
        "elapsed_seconds": elapsed_seconds,
        "iterations": completed_iterations,
        "total_games": total_games,
        "total_samples": total_samples,
        "device": str(device),
        "inference_device": str(inference_device),
        "search_count": args.search_count,
        "lanes": args.lanes,
        "inference_batch_size": args.inference_batch_size,
        "plateau_window": args.plateau_window,
        "plateau_delta": args.plateau_delta,
        "plateau_patience": args.plateau_patience,
        "plateau_ranges": plateau_ranges,
        "converged_loss": converged_loss,
        "converged_mean_loss": (
            sum(converged_loss.values()) / len(converged_loss)
            if converged_loss
            else 0.0
        ),
        "final_loss": final_loss,
        "final_mean_loss": (
            sum(final_loss.values()) / len(final_loss) if final_loss else 0.0
        ),
        "final_max_loss": max(final_loss.values(), default=0.0),
    }
    write_json(run_dir / "convergence_summary.json", summary)
    if live_recorder is not None:
        live_recorder.finish_training(
            f"学習完了: {convergence_reason}, {elapsed_seconds:.2f}秒"
        )
    print(
        f"Central round-robin training finished in {elapsed_seconds:.2f}s "
        f"({convergence_reason})."
    )


if __name__ == "__main__":
    main()
