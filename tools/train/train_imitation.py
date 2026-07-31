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
import pickle
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agents" / "rl_mcts"
SRC_ROOT = AGENT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rl_mcts.mcts import MAX_ACTIONS  # noqa: E402
from rl_mcts.model import create_model  # noqa: E402


def load_shard(path: Path) -> list[tuple]:
    with open(path, "rb") as f:
        return pickle.load(f)


def build_batch_tensors_cpu(batch: list[tuple], pin: bool):
    """バッチをCPUテンソルに変換する（numpyでベクトル化、必要ならpinned memory化）。

    サンプルの各フィールドは Python list でも numpy 配列でも受け付ける
    （新しい preprocess はメモリ削減のため numpy int32/float32 で保存する）。
    GPU転送は呼び出し側が to_device() で行う。
    """
    n = len(batch)
    enc_idx, enc_val, enc_off = [], [], []
    dec_idx, dec_val, dec_off = [], [], []
    mask = np.zeros((n, MAX_ACTIONS), dtype=np.float32)
    label_value = np.empty((n, 1), dtype=np.float32)
    chosen = np.empty(n, dtype=np.int64)

    enc_base = 0
    dec_base = 0
    for i, (e_i, e_v, e_o, d_i, d_v, d_o, chosen_index, value) in enumerate(batch):
        enc_idx.append(np.asarray(e_i, dtype=np.int32))
        enc_val.append(np.asarray(e_v, dtype=np.float32))
        enc_off.append(np.asarray(e_o, dtype=np.int32) + enc_base)
        enc_base += len(e_i)

        dec_idx.append(np.asarray(d_i, dtype=np.int32))
        dec_val.append(np.asarray(d_v, dtype=np.float32))
        off = np.asarray(d_o, dtype=np.int32) + dec_base
        dec_base += len(d_i)
        n_candidates = len(d_o)
        if n_candidates < MAX_ACTIONS:
            # 旧実装と同じく、空バッグ（開始位置=現在の末尾）でMAX_ACTIONSまで埋める。
            off = np.concatenate([off, np.full(MAX_ACTIONS - n_candidates, dec_base, dtype=np.int32)])
        dec_off.append(off)

        mask[i, :n_candidates] = 1.0
        label_value[i, 0] = value
        chosen[i] = chosen_index

    tensors = (
        torch.from_numpy(np.concatenate(enc_idx)),
        torch.from_numpy(np.concatenate(enc_val)),
        torch.from_numpy(np.concatenate(enc_off)),
        torch.from_numpy(np.concatenate(dec_idx)),
        torch.from_numpy(np.concatenate(dec_val)),
        torch.from_numpy(np.concatenate(dec_off)),
    )
    mask_tensor = torch.from_numpy(mask)
    label_value_tensor = torch.from_numpy(label_value)
    chosen_index_tensor = torch.from_numpy(chosen)

    if pin:
        tensors = tuple(t.pin_memory() for t in tensors)
        mask_tensor = mask_tensor.pin_memory()
        label_value_tensor = label_value_tensor.pin_memory()
        chosen_index_tensor = chosen_index_tensor.pin_memory()
    return tensors, mask_tensor, label_value_tensor, chosen_index_tensor


def to_device(built, device: torch.device):
    """build_batch_tensors_cpu の結果をGPUへ非同期転送する（pinned前提でnon_blocking）。"""
    tensors, mask_tensor, label_value_tensor, chosen_index_tensor = built
    return (
        tuple(t.to(device, non_blocking=True) for t in tensors),
        mask_tensor.to(device, non_blocking=True),
        label_value_tensor.to(device, non_blocking=True),
        chosen_index_tensor.to(device, non_blocking=True),
    )


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


def iter_built_batches(
    shard_paths: list[Path],
    batch_size: int,
    shuffle: bool,
    pin: bool,
    prefetch: int,
) -> Iterator[tuple]:
    """バックグラウンドスレッドでシャード読み込み＋テンソル構築を先行実行する。

    GPUが現在のバッチを計算している間に、次バッチのディスクI/OとCPU側の
    テンソル構築を進める（先読み深さ=prefetch）。prefetch<=0 なら同期実行。
    """
    if prefetch <= 0:
        for batch in iter_batches(shard_paths, batch_size, shuffle):
            yield build_batch_tensors_cpu(batch, pin)
        return

    q: queue.Queue = queue.Queue(maxsize=prefetch)
    sentinel = object()

    def producer() -> None:
        try:
            for batch in iter_batches(shard_paths, batch_size, shuffle):
                q.put(build_batch_tensors_cpu(batch, pin))
        except BaseException as exc:  # 例外は消費側スレッドへ運ぶ
            q.put(exc)
            return
        q.put(sentinel)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    thread.join()


def _autocast(device: torch.device, enabled: bool):
    """CUDA時のみbf16 autocastを返す（未対応環境・CPUでは無効）。"""
    use = enabled and device.type == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use)


def train_one_epoch(
    model,
    optimizer,
    shard_paths: list[Path],
    batch_size: int,
    device: torch.device,
    amp: bool = False,
    prefetch: int = 4,
) -> dict:
    model.train()
    loss_fn_value = torch.nn.HuberLoss(delta=0.2)
    pin = device.type == "cuda"

    batch_count = 0
    total_seen = 0
    # GPU同期(.item())を毎バッチ呼ぶとCPU側の先読みが止まるため、
    # 集計はテンソルのまま持ち回してエポック末に1回だけ同期する。
    sum_loss = torch.zeros((), device=device)
    sum_loss_value = torch.zeros((), device=device)
    sum_loss_policy = torch.zeros((), device=device)
    sum_correct = torch.zeros((), dtype=torch.long, device=device)

    for built in iter_built_batches(shard_paths, batch_size, shuffle=True, pin=pin, prefetch=prefetch):
        tensors, mask_tensor, label_value_tensor, chosen_index_tensor = to_device(built, device)

        optimizer.zero_grad()
        with _autocast(device, amp):
            out_enc, out_dec = model(*tensors)
            loss_value = loss_fn_value(out_enc.float(), label_value_tensor)
            masked_logits = out_dec.float().masked_fill(mask_tensor == 0, float("-inf"))
            loss_policy = F.cross_entropy(masked_logits, chosen_index_tensor)
            loss = loss_value + loss_policy

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = masked_logits.argmax(dim=1)
            sum_correct += (pred == chosen_index_tensor).sum()
            sum_loss += loss.detach()
            sum_loss_value += loss_value.detach()
            sum_loss_policy += loss_policy.detach()
        total_seen += len(chosen_index_tensor)
        batch_count += 1

    if batch_count == 0:
        return {"batches": 0, "loss": 0.0, "loss_value": 0.0, "loss_policy": 0.0, "accuracy": 0.0}

    return {
        "batches": batch_count,
        "loss": float(sum_loss.item()) / batch_count,
        "loss_value": float(sum_loss_value.item()) / batch_count,
        "loss_policy": float(sum_loss_policy.item()) / batch_count,
        "accuracy": int(sum_correct.item()) / total_seen,
    }


def evaluate(
    model,
    shard_paths: list[Path],
    batch_size: int,
    device: torch.device,
    amp: bool = False,
    prefetch: int = 4,
) -> float:
    if not shard_paths:
        return 0.0

    model.eval()
    pin = device.type == "cuda"
    correct = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    with torch.no_grad():
        for built in iter_built_batches(shard_paths, batch_size, shuffle=False, pin=pin, prefetch=prefetch):
            tensors, mask_tensor, _, chosen_index_tensor = to_device(built, device)
            with _autocast(device, amp):
                _, out_dec = model(*tensors)
            masked_logits = out_dec.float().masked_fill(mask_tensor == 0, float("-inf"))
            pred = masked_logits.argmax(dim=1)
            correct += (pred == chosen_index_tensor).sum()
            total += len(chosen_index_tensor)

    return int(correct.item()) / total if total else 0.0


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
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA時にbf16 autocast + TF32行列演算を使う（--no-ampで旧来のfp32厳密計算）",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="バックグラウンドで先読み構築するバッチ数（0で同期実行）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if use_amp:
        # TF32(行列演算)とbf16 autocastを併用。--no-ampで従来のfp32厳密計算に戻せる。
        torch.set_float32_matmul_precision("high")
    print(f"device: {device} amp={'bf16+tf32' if use_amp else 'off'} prefetch={args.prefetch_batches}")

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
        stats = train_one_epoch(
            model, optimizer, train_shards, args.batch_size, device,
            amp=use_amp, prefetch=args.prefetch_batches,
        )
        val_acc = evaluate(
            model, val_shards, args.batch_size, device,
            amp=use_amp, prefetch=args.prefetch_batches,
        )
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
