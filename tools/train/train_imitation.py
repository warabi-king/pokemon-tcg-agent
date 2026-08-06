"""前処理済みシャード(tools/train/preprocess_episodes.pyの出力)を使ったrl_mctsの模倣学習スクリプト。

一度に全サンプルをメモリに載せず、シャード単位でストリーミング学習する
(常時メモリ上に持つのは1シャード分のサンプルだけ)。毎エポック、シャードの読み込み順と
シャード内のサンプル順をシャッフルする。

policyはHuberLoss回帰ではなく、実際に選ばれた手を正解クラスとした交差エントロピー、
valueはMCTS探索を使わず、そのエピソードの実際の勝敗(rewards)をそのまま使う。

使い方:
    事前に tools/train/preprocess_episodes.py でシャードを作っておく。

    python tools/train/train_imitation.py \
        --shards shards/all \
        --epochs 3 \
        --batch-size 128

    --output-modelのデフォルトは提出用のagents/rl_mcts/src/model.pthを誤って
    上書きしないよう、agents/rl_mcts/train/checkpoints/imitation_model.pthにしている。
    結果を提出物として使う場合は --output-model agents/rl_mcts/src/model.pth を指定する。
    詳細な引数一覧は tools/train/README.md の「模倣学習（公式リプレイ）」節を参照。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agents" / "rl_mcts"
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rl_mcts.mcts import MAX_ACTIONS, LearnInput  # noqa: E402
from rl_mcts.model import create_model  # noqa: E402


def load_shard(path: Path) -> list[tuple]:
    with open(path, "rb") as f:
        return pickle.load(f)


# SELFPLAY_COMPLETED_Q_TARGET_PATCH_V1
# search_count=10のself-playでは、MCTSのvisit分布がほぼ均等になる(2026-08-05実測:
# 305 real decisionsのうち80.1%でvisit最大が同数、tieの平均幅4.42)。そのため
# chosen_index(=visit argmax)は合法手の列挙順で決まってしまい、45.4%の局面では
# ネットワーク自身の最有力手と食い違うラベルになっていた。
# completed_q(合法手ごとのQ優位度 clamp(Q(a)-Q(root), -1, 1))が保存されている
# サンプルでは、1手のhard labelではなくsoftmax(Q/T)のsoft targetを教師にできる。
# 環境変数SELFPLAY_POLICY_TARGET未設定時は従来どおりhard labelのまま。
POLICY_TARGET_MODES = frozenset({"hard", "completed_q"})


def policy_target_mode() -> str:
    mode = os.environ.get("SELFPLAY_POLICY_TARGET", "hard")
    if mode not in POLICY_TARGET_MODES:
        raise SystemExit(
            f"SELFPLAY_POLICY_TARGETが不正です: {mode!r} (有効値: {sorted(POLICY_TARGET_MODES)})"
        )
    return mode


def policy_target_temperature() -> float:
    temperature = float(os.environ.get("SELFPLAY_POLICY_TARGET_TEMPERATURE", "0.25"))
    if temperature <= 0.0:
        raise SystemExit("SELFPLAY_POLICY_TARGET_TEMPERATUREは正の値が必要です。")
    return temperature


# SELFPLAY_SEARCH_VALUE_TARGET_PATCH_V1
# valueの教師は「その試合の最終勝敗(±1)」を全局面へ一律に付けたもの。60ターンの
# 試合なら序盤も終盤も同じ±1になり、局面ごとの優劣が入らない最大分散の教師になる。
# q tie-breakの採用でQ値が着手を決めるようになったため、valueヘッドの質は直接
# 着手品質に効く。探索rootの評価値と混ぜて分散を下げられるようにする。
# lambda=1.0(既定)なら従来どおり最終勝敗そのまま。
def value_target_lambda() -> float:
    lam = float(os.environ.get("SELFPLAY_VALUE_TARGET_LAMBDA", "1.0"))
    if not 0.0 <= lam <= 1.0:
        raise SystemExit(
            f"SELFPLAY_VALUE_TARGET_LAMBDAは0.0〜1.0にしてください: {lam}"
        )
    return lam


def blended_value(outcome: float, search_value, lam: float) -> float:
    if lam >= 1.0 or search_value is None:
        return outcome
    return lam * outcome + (1.0 - lam) * float(search_value)


def policy_target_zero_spread() -> str:
    """Q優位度が全合法手で同値だった行の扱い。

    実測(2026-08-05, 条件Iの実shard 107,997サンプル)では28.2%の行がこれに当たる。
    softmaxをそのまま適用すると一様分布が教師になり、学習データの28%が
    「方策を平坦にせよ」という勾配になってしまう。既定では従来のhard labelへ
    退避させ、探索が意見を持っている行だけをsoft targetに変える(条件Jで
    測りたいのはそこだけなので、交絡させない)。
    """
    mode = os.environ.get("SELFPLAY_POLICY_TARGET_ZERO_SPREAD", "hard")
    if mode not in {"hard", "uniform"}:
        raise SystemExit(
            f"SELFPLAY_POLICY_TARGET_ZERO_SPREADが不正です: {mode!r} (有効値: ['hard', 'uniform'])"
        )
    return mode


def soft_target_row(
    completed_q: list[float],
    chosen_index: int,
    n_candidates: int,
    temperature: float,
    zero_spread: str = "hard",
) -> list[float]:
    """1サンプル分のsoft target分布を返す(長さはn_candidates)。"""
    if len(completed_q) != n_candidates:
        # 探索が走らなかった局面(セットアップ中のpolicy-only判断)などは
        # completed_qが空になる。その行だけ従来のhard labelへ退避する。
        row = [0.0] * n_candidates
        row[chosen_index] = 1.0
        return row
    if zero_spread == "hard" and max(completed_q) == min(completed_q):
        row = [0.0] * n_candidates
        row[chosen_index] = 1.0
        return row
    scaled = [value / temperature for value in completed_q]
    highest = max(scaled)
    weights = [math.exp(value - highest) for value in scaled]
    total = sum(weights)
    if total <= 0.0:
        row = [0.0] * n_candidates
        row[chosen_index] = 1.0
        return row
    return [weight / total for weight in weights]


def build_batch_tensors(batch: list[tuple], device: torch.device):
    input_enc = LearnInput()
    input_dec = LearnInput()
    mask: list[float] = []
    label_value: list[float] = []
    chosen_indices: list[int] = []
    soft_targets: list[float] = []
    mode = policy_target_mode()
    temperature = policy_target_temperature()
    zero_spread = policy_target_zero_spread()
    value_lambda = value_target_lambda()

    for sample in batch:
        (
            enc_index,
            enc_value,
            enc_offset,
            dec_index,
            dec_value,
            dec_offset,
            chosen_index,
            value,
        ) = sample[:8]
        completed_q = list(sample[8]) if len(sample) > 8 else []
        search_value = sample[9] if len(sample) > 9 else None
        value = blended_value(value, search_value, value_lambda)
        enc_count = len(input_enc.index)
        input_enc.index.extend(enc_index)
        input_enc.value.extend(enc_value)
        input_enc.offset.extend(o + enc_count for o in enc_offset)

        dec_count = len(input_dec.index)
        input_dec.index.extend(dec_index)
        input_dec.value.extend(dec_value)
        input_dec.offset.extend(o + dec_count for o in dec_offset)

        label_value.append(value)
        chosen_indices.append(chosen_index)

        n_candidates = len(dec_offset)
        mask.extend([1.0] * n_candidates)
        if mode == "completed_q":
            soft_targets.extend(
                soft_target_row(
                    completed_q, chosen_index, n_candidates, temperature, zero_spread
                )
            )
        for _ in range(MAX_ACTIONS - n_candidates):
            mask.append(0.0)
            if mode == "completed_q":
                soft_targets.append(0.0)
            input_dec.offset.append(len(input_dec.index))

    n = len(batch)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).view(n, -1)
    label_value_tensor = torch.tensor(label_value, dtype=torch.float32, device=device).view(n, -1)
    chosen_index_tensor = torch.tensor(chosen_indices, dtype=torch.long, device=device)
    soft_target_tensor = (
        torch.tensor(soft_targets, dtype=torch.float32, device=device).view(n, -1)
        if mode == "completed_q"
        else None
    )

    tensors = (
        torch.tensor(input_enc.index, dtype=torch.int32, device=device),
        torch.tensor(input_enc.value, dtype=torch.float32, device=device),
        torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(input_dec.index, dtype=torch.int32, device=device),
        torch.tensor(input_dec.value, dtype=torch.float32, device=device),
        torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
    )
    return tensors, mask_tensor, label_value_tensor, chosen_index_tensor, soft_target_tensor


def iter_batches(shard_paths: list[Path], batch_size: int, shuffle: bool) -> Iterator[list[tuple]]:
    """シャードを1つずつ読み込み、シャード内シャッフルしてバッチを返す。

    常にメモリ上には1シャード分のサンプルしか保持しない。バッチはシャードをまたがない
    (端数はそのシャード内で切り捨てる)。
    """
    paths = list(shard_paths)
    if shuffle:
        random.shuffle(paths)

    for path in paths:
        samples = load_shard(path)
        if shuffle:
            random.shuffle(samples)
        for start in range(0, len(samples) - batch_size + 1, batch_size):
            yield samples[start : start + batch_size]


def train_one_epoch(model, optimizer, shard_paths: list[Path], batch_size: int, device: torch.device) -> dict:
    model.train()
    loss_fn_value = torch.nn.HuberLoss(delta=0.2)

    batch_count = 0
    total_loss = total_loss_value = total_loss_policy = 0.0
    total_correct = total_seen = 0

    for batch in iter_batches(shard_paths, batch_size, shuffle=True):
        (
            tensors,
            mask_tensor,
            label_value_tensor,
            chosen_index_tensor,
            soft_target_tensor,
        ) = build_batch_tensors(batch, device)

        optimizer.zero_grad()
        out_enc, out_dec = model(*tensors)

        loss_value = loss_fn_value(out_enc, label_value_tensor)
        masked_logits = out_dec.masked_fill(mask_tensor == 0, float("-inf"))
        if soft_target_tensor is None:
            loss_policy = F.cross_entropy(masked_logits, chosen_index_tensor)
        else:
            # maskされた位置のlog_probは-infなので、0 * -inf = nan を避けるため
            # 先に0で埋める(soft targetもその位置は0)。
            log_probabilities = torch.log_softmax(masked_logits, dim=1)
            log_probabilities = log_probabilities.masked_fill(mask_tensor == 0, 0.0)
            loss_policy = -(soft_target_tensor * log_probabilities).sum(dim=1).mean()

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
        batch_count += 1

    if batch_count == 0:
        return {"batches": 0, "loss": 0.0, "loss_value": 0.0, "loss_policy": 0.0, "accuracy": 0.0}

    return {
        "batches": batch_count,
        "loss": total_loss / batch_count,
        "loss_value": total_loss_value / batch_count,
        "loss_policy": total_loss_policy / batch_count,
        "accuracy": total_correct / total_seen,
    }


def evaluate(model, shard_paths: list[Path], batch_size: int, device: torch.device) -> float:
    if not shard_paths:
        return 0.0

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in iter_batches(shard_paths, batch_size, shuffle=False):
            tensors, mask_tensor, _, chosen_index_tensor, _ = build_batch_tensors(batch, device)
            _, out_dec = model(*tensors)
            masked_logits = out_dec.masked_fill(mask_tensor == 0, float("-inf"))
            pred = masked_logits.argmax(dim=1)
            correct += int((pred == chosen_index_tensor).sum().item())
            total += len(batch)

    return correct / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards", type=Path, required=True, help="preprocess_episodes.pyの--output-dir")
    parser.add_argument("--val-shards", type=int, default=1, help="検証用に取り分けるシャード数")
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto/cpu/mps/cuda（autoはCUDA、MPS、CPUの順）",
    )
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    normalized = name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif torch.backends.mps.is_available():
            normalized = "mps"
        else:
            normalized = "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDAを利用できません。")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPSを利用できません。")
    return device


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = select_device(args.device)
    print(f"device: {device}")

    shard_paths = sorted(args.shards.glob("shard_*.pkl"))
    if not shard_paths:
        raise SystemExit(f"シャードが見つかりません: {args.shards}（先にpreprocess_episodes.pyを実行してください）")

    manifest_path = args.shards / "manifest.json"
    if manifest_path.exists():
        print(f"manifest: {json.loads(manifest_path.read_text(encoding='utf-8'))}")

    random.shuffle(shard_paths)
    val_count = min(max(args.val_shards, 0), max(len(shard_paths) - 1, 0))
    val_shards = shard_paths[:val_count]
    train_shards = shard_paths[val_count:]
    print(f"train shards={len(train_shards)} val shards={len(val_shards)}")

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
        stats = train_one_epoch(model, optimizer, train_shards, args.batch_size, device)
        val_acc = evaluate(model, val_shards, args.batch_size, device)
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
