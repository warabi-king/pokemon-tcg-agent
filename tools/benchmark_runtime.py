"""最終TransformerのCPU/MPS短時間推論benchmarkをJSONへ保存する。

実行例:
    python tools/benchmark_runtime.py \
        --model agents/belief_puct/src/model.pth \
        --shard agents/belief_puct/train/shards/validation_20260718_20/shard_00000.pkl \
        --output work/audits/hardware-benchmark.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agents" / "belief_puct"
sys.path.insert(0, str(AGENT_ROOT / "src"))
sys.path.insert(0, str(AGENT_ROOT / "train"))

from rl_mcts.model import create_model  # noqa: E402
from train_imitation import build_batch_tensors  # noqa: E402


def load_samples(shard_path: Path, batch_size: int) -> list[tuple]:
    """pickle shardの先頭から固定batchを読み込む。"""

    with shard_path.open("rb") as source:
        samples = pickle.load(source)
    if len(samples) < batch_size:
        raise ValueError(f"benchmark用sampleが不足しています: {len(samples)} < {batch_size}")
    return samples[:batch_size]


def percentile(values: list[float], proportion: float) -> float:
    """小標本benchmark用のnearest-rank percentileを返す。"""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * proportion)))
    return ordered[index]


def benchmark_device(
    model_path: Path,
    samples: list[tuple],
    device_name: str,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """指定deviceで同じbatchのforward latencyを測定する。"""

    device = torch.device(device_name)
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tensors, mask, labels, chosen = build_batch_tensors(samples, device)
    del mask, labels, chosen

    with torch.inference_mode():
        for _ in range(warmup):
            model(*tensors)
        if device.type == "mps":
            torch.mps.synchronize()
        latencies_ms: list[float] = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(*tensors)
            if device.type == "mps":
                torch.mps.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "device": device_name,
        "batch_size": len(samples),
        "warmup": warmup,
        "repeats": repeats,
        "median_batch_ms": statistics.median(latencies_ms),
        "p95_batch_ms": percentile(latencies_ms, 0.95),
        "median_sample_ms": statistics.median(latencies_ms) / len(samples),
        "min_batch_ms": min(latencies_ms),
        "max_batch_ms": max(latencies_ms),
    }


def parse_args() -> argparse.Namespace:
    """model、shard、batch、反復数、出力先を解析する。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """利用可能deviceを確認し、CPUと利用可能ならMPSを測定する。"""

    args = parse_args()
    torch.manual_seed(20260803)
    samples = load_samples(args.shard, args.batch_size)
    model = create_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model

    devices = ["cpu"]
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    if mps_available:
        devices.append("mps")
    results = [
        benchmark_device(
            args.model,
            samples,
            device,
            args.warmup,
            args.repeats,
        )
        for device in devices
    ]
    report = {
        "format": "pokemon-tcg-agent/runtime-benchmark-v1",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_status": "measured" if mps_available else "unavailable; not benchmarked",
        "model": str(args.model),
        "model_bytes": args.model.stat().st_size,
        "parameter_count": parameter_count,
        "sample_shard": str(args.shard),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
