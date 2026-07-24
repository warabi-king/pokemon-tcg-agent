"""公式リプレイ(エピソードJSON)を使ったrl_mctsの模倣学習スクリプト。

各エピソードJSONの steps[i][player]["observation"]["select"] が提示された選択肢で、
それに対する実際の回答は steps[i+1][player]["action"] に入っている
(kaggle_environmentsの一般的な規約: action[i]はobservation[i-1]への回答)。

policyはHuberLoss回帰ではなく、実際に選ばれた手を正解クラスとした交差エントロピー、
valueはMCTS探索を使わず、そのエピソードの実際の勝敗(rewards)をそのまま使う。

使い方:
    事前にKaggleの日次エピソードデータセット(例: pokemon-tcg-ai-battle-episodes-2026-07-23)
    をダウンロードしておく。--episodesには展開済みディレクトリと.zipのどちらも渡せる
    (.zipならディスク展開せずそのまま読む)。

    python tools/train/train_imitation.py \
        --episodes path/to/pokemon-tcg-ai-battle-episodes-2026-07-23.zip \
        --epochs 5 \
        --batch-size 128

    少数だけで動作確認する場合:

    python tools/train/train_imitation.py \
        --episodes path/to/episodes.zip \
        --max-episodes 5 \
        --epochs 2 \
        --batch-size 32

    --output-modelのデフォルトは提出用のagents/rl_mcts/src/model.pthを誤って
    上書きしないよう、agents/rl_mcts/train/checkpoints/imitation_model.pthにしている。
    結果を提出物として使う場合は --output-model agents/rl_mcts/src/model.pth を指定する。
    詳細な引数一覧は tools/train/README.md の「模倣学習（公式リプレイ）」節を参照。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agents" / "rl_mcts"
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cg.api import to_observation_class  # noqa: E402
from rl_mcts.features import SparseVector, get_decoder_input, get_encoder_input  # noqa: E402
from rl_mcts.mcts import MAX_ACTIONS, LearnInput, enumerate_actions  # noqa: E402
from rl_mcts.model import create_model  # noqa: E402


@dataclass
class ImitationSample:
    sv_enc: SparseVector
    sv_dec: SparseVector
    chosen_index: int
    value: float


def iter_episode_files(episodes_path: Path) -> Iterator[tuple[str, bytes]]:
    """展開済みディレクトリ、またはKaggleからの.zipのどちらからでも(名前, 生データ)を読む。"""
    if episodes_path.is_dir():
        for p in sorted(episodes_path.glob("*.json")):
            yield p.name, p.read_bytes()
        return

    with zipfile.ZipFile(episodes_path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                yield name, zf.read(name)


def extract_samples_from_episode(data: bytes) -> list[ImitationSample]:
    """1エピソード分のJSONから学習サンプルを取り出す。"""
    j = json.loads(data)
    rewards = j.get("rewards")
    if not rewards or len(rewards) != 2 or any(r is None for r in rewards):
        return []

    steps = j["steps"]
    if len(steps) < 3:
        return []

    decks = [steps[1][0]["action"], steps[1][1]["action"]]
    if len(decks[0]) != 60 or len(decks[1]) != 60:
        return []

    samples: list[ImitationSample] = []
    for player in range(2):
        your_deck = decks[player]
        value = float(rewards[player])

        for i in range(1, len(steps) - 1):
            sel = steps[i][player]["observation"].get("select")
            if sel is None:
                continue

            actual_action = steps[i + 1][player]["action"]
            if actual_action is None:
                continue

            obs = to_observation_class(steps[i][player]["observation"])
            actions = enumerate_actions(len(obs.select.option), obs.select.maxCount)
            if not actions:
                continue

            target = tuple(sorted(actual_action))
            chosen_index = next(
                (idx for idx, candidate in enumerate(actions) if tuple(candidate) == target),
                None,
            )
            if chosen_index is None:
                continue

            sv_enc = get_encoder_input(obs, your_deck)
            sv_dec = get_decoder_input(obs, actions)
            samples.append(ImitationSample(sv_enc, sv_dec, chosen_index, value))

    return samples


def collect_samples(
    episodes_path: Path,
    max_episodes: int | None,
    max_samples: int | None,
) -> list[ImitationSample]:
    samples: list[ImitationSample] = []
    episode_count = 0
    error_count = 0
    t0 = time.time()

    for name, data in iter_episode_files(episodes_path):
        if name == "manifest.csv":
            continue
        if max_episodes is not None and episode_count >= max_episodes:
            break

        try:
            samples.extend(extract_samples_from_episode(data))
        except Exception:
            error_count += 1
            continue

        episode_count += 1
        if episode_count % 200 == 0:
            elapsed = time.time() - t0
            print(
                f"episodes={episode_count} samples={len(samples)} errors={error_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if max_samples is not None and len(samples) >= max_samples:
            break

    print(f"collected: episodes={episode_count} samples={len(samples)} errors={error_count}")
    return samples


def build_batch_tensors(batch: list[ImitationSample], device: torch.device):
    input_enc = LearnInput()
    input_dec = LearnInput()
    mask: list[float] = []
    label_value: list[float] = []
    chosen_indices: list[int] = []

    for sample in batch:
        input_enc.add(sample.sv_enc)
        input_dec.add(sample.sv_dec)
        label_value.append(sample.value)
        chosen_indices.append(sample.chosen_index)

        n_candidates = len(sample.sv_dec.offset)
        mask.extend([1.0] * n_candidates)
        for _ in range(MAX_ACTIONS - n_candidates):
            mask.append(0.0)
            input_dec.offset.append(len(input_dec.index))

    n = len(batch)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(n, -1)
    label_value_tensor = torch.tensor(label_value, dtype=torch.float32, device=device).view(n, -1)
    chosen_index_tensor = torch.tensor(chosen_indices, dtype=torch.long, device=device)

    tensors = (
        torch.tensor(input_enc.index, dtype=torch.int32, device=device),
        torch.tensor(input_enc.value, dtype=torch.float32, device=device),
        torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(input_dec.index, dtype=torch.int32, device=device),
        torch.tensor(input_dec.value, dtype=torch.float32, device=device),
        torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
    )
    return tensors, mask_tensor, label_value_tensor, chosen_index_tensor


def train_one_epoch(model, optimizer, samples: list[ImitationSample], batch_size: int, device: torch.device) -> dict:
    model.train()
    random.shuffle(samples)
    loss_fn_value = torch.nn.HuberLoss(delta=0.2)

    batch_count = len(samples) // batch_size
    total_loss = total_loss_value = total_loss_policy = 0.0
    total_correct = total_seen = 0

    for b in range(batch_count):
        batch = samples[b * batch_size : (b + 1) * batch_size]
        tensors, mask_tensor, label_value_tensor, chosen_index_tensor = build_batch_tensors(batch, device)

        optimizer.zero_grad()
        out_enc, out_dec = model(*tensors)

        loss_value = loss_fn_value(out_enc, label_value_tensor)
        masked_logits = out_dec.masked_fill(mask_tensor == 0, float("-inf"))
        loss_policy = F.cross_entropy(masked_logits, chosen_index_tensor)

        loss = loss_value + loss_policy
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = masked_logits.argmax(dim=1)
            total_correct += int((pred == chosen_index_tensor).sum().item())
            total_seen += len(batch)

        total_loss += float(loss.item())
        total_loss_value += float(loss_value.item())
        total_loss_policy += float(loss_policy.item())

    if batch_count == 0:
        return {"batches": 0, "loss": 0.0, "loss_value": 0.0, "loss_policy": 0.0, "accuracy": 0.0}

    return {
        "batches": batch_count,
        "loss": total_loss / batch_count,
        "loss_value": total_loss_value / batch_count,
        "loss_policy": total_loss_policy / batch_count,
        "accuracy": total_correct / total_seen,
    }


def evaluate(model, samples: list[ImitationSample], batch_size: int, device: torch.device) -> float:
    if not samples:
        return 0.0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            tensors, mask_tensor, _, chosen_index_tensor = build_batch_tensors(batch, device)
            _, out_dec = model(*tensors)
            masked_logits = out_dec.masked_fill(mask_tensor == 0, float("-inf"))
            pred = masked_logits.argmax(dim=1)
            correct += int((pred == chosen_index_tensor).sum().item())
            total += len(batch)

    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="展開済みエピソードJSONのディレクトリ、またはKaggleからダウンロードした.zip",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--initial-model", type=Path, default=None, help="続きから学習する場合の初期重み")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=AGENT_ROOT / "train" / "checkpoints" / "imitation_model.pth",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=AGENT_ROOT / "train" / "logs" / "imitation_metrics.csv",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    samples = collect_samples(args.episodes, args.max_episodes, args.max_samples)
    if len(samples) < args.batch_size:
        raise SystemExit(f"サンプル数が不足しています: {len(samples)} < batch_size={args.batch_size}")

    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    print(f"train samples={len(train_samples)} val samples={len(val_samples)}")

    model = create_model().to(device)
    if args.initial_model and args.initial_model.exists():
        model.load_state_dict(torch.load(args.initial_model, map_location=device))
        print(f"loaded initial weights: {args.initial_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "batches",
        "loss",
        "loss_value",
        "loss_policy",
        "train_accuracy",
        "val_accuracy",
        "elapsed_seconds",
    ]
    with open(args.metrics_file, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    t0 = time.time()
    for epoch in range(args.epochs):
        stats = train_one_epoch(model, optimizer, train_samples, args.batch_size, device)
        val_acc = evaluate(model, val_samples, args.batch_size, device)
        elapsed = time.time() - t0

        print(
            f"epoch={epoch} loss={stats['loss']:.4f} loss_value={stats['loss_value']:.4f} "
            f"loss_policy={stats['loss_policy']:.4f} train_acc={stats['accuracy']:.3f} "
            f"val_acc={val_acc:.3f} elapsed={elapsed:.1f}s"
        )

        with open(args.metrics_file, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {
                    "epoch": epoch,
                    "batches": stats["batches"],
                    "loss": stats["loss"],
                    "loss_value": stats["loss_value"],
                    "loss_policy": stats["loss_policy"],
                    "train_accuracy": stats["accuracy"],
                    "val_accuracy": val_acc,
                    "elapsed_seconds": elapsed,
                }
            )

        torch.save(model.state_dict(), args.output_model)

    print(f"saved model: {args.output_model}")


if __name__ == "__main__":
    main()
