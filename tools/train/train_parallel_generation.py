"""並列対戦JSONから複数agentのself/相手モデルを1回更新する。

全JSONを一度だけ前処理してagent別シャードへ振り分け、各agentについて
``model.pth``（自分の着手）と``opponent_model.pth``（そのagentと戦う側の着手）を
別々に継続学習する。全モデルの学習成功後にだけ実行agentへ一括反映する。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = TRAIN_ROOT.parent / "pipeline"
for path in (TRAIN_ROOT, PIPELINE_ROOT):
    sys.path.insert(0, str(path))

from preprocess_match_agents import preprocess_match_agents  # noqa: E402
from trainer import train_model  # noqa: E402


def parse_agent(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--agentはname=agent/src形式で指定してください。")
    name, path = raw.split("=", 1)
    src = Path(path).resolve()
    if not name.strip() or not src.is_dir():
        raise argparse.ArgumentTypeError(f"不正な--agentです: {raw}")
    for filename in ("deck.csv", "model.pth"):
        if not (src / filename).exists():
            raise argparse.ArgumentTypeError(f"{src / filename}がありません。")
    return name.strip(), src


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _default_checkpoint_dir(
    agents: list[tuple[str, Path]],
    work_dir: Path,
) -> Path:
    model_roots = {src.parent.parent.resolve() for _name, src in agents}
    if len(model_roots) != 1:
        raise ValueError(
            "agentが同じモデルルートにないため--checkpoint-dirを指定してください。"
        )
    episode_name = work_dir.parent.name
    if not episode_name.startswith("episode"):
        raise ValueError(
            "work-dirから累計episode数を判定できないため"
            "--checkpoint-dirを指定してください。"
        )
    return next(iter(model_roots)) / f"model_{episode_name}"


def train_parallel_generation(
    episodes: list[Path],
    agents: list[tuple[str, Path]],
    work_dir: Path,
    *,
    checkpoint_dir: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shard_size: int,
    device: str,
    seed: int,
) -> dict:
    if len({name for name, _ in agents}) != len(agents):
        raise ValueError("agent名が重複しています。")
    work_dir = work_dir.resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"学習work-dirが空ではありません: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is None:
        checkpoint_dir = _default_checkpoint_dir(agents, work_dir)
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.exists():
        raise FileExistsError(
            f"学習済みcheckpointの保存先が既に存在します: {checkpoint_dir}"
        )
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint_dir = work_dir / ".checkpoint_staging"
    if temporary_checkpoint_dir.exists():
        raise FileExistsError(
            f"一時checkpoint保存先が既に存在します: {temporary_checkpoint_dir}"
        )

    started = time.time()
    logs = work_dir / "logs"

    preprocess_started = time.time()
    shard_summary = preprocess_match_agents(
        episodes,
        [(name, src / "deck.csv") for name, src in agents],
        work_dir / "shards",
        shard_size=shard_size,
    )
    preprocess_seconds = time.time() - preprocess_started

    training_started = time.time()
    trained_models: list[dict] = []
    for agent_index, (name, src) in enumerate(agents):
        input_self = src / "model.pth"
        input_opponent = src / "opponent_model.pth"
        if not input_opponent.exists():
            input_opponent = input_self
        output_self = temporary_checkpoint_dir / name / "model.pth"
        output_opponent = temporary_checkpoint_dir / name / "opponent_model.pth"
        output_self.parent.mkdir(parents=True, exist_ok=True)

        self_trained = train_model(
            work_dir / "shards" / f"{name}_own",
            output_self,
            epochs,
            initial_model=input_self,
            metrics_file=logs / f"{name}_self.csv",
            batch_size=batch_size,
            lr=learning_rate,
            device=device,
            seed=seed + agent_index * 2,
            val_shards=0,
        )
        if not self_trained:
            shutil.copy2(input_self, output_self)

        opponent_trained = train_model(
            work_dir / "shards" / f"{name}_opponent",
            output_opponent,
            epochs,
            initial_model=input_opponent,
            metrics_file=logs / f"{name}_opponent.csv",
            batch_size=batch_size,
            lr=learning_rate,
            device=device,
            seed=seed + agent_index * 2 + 1,
            val_shards=0,
        )
        if not opponent_trained:
            shutil.copy2(input_opponent, output_opponent)

        trained_models.append(
            {
                "agent": name,
                "selfTrained": self_trained,
                "opponentTrained": opponent_trained,
                "selfSamples": shard_summary[f"{name}_own"]["samples"],
                "opponentSamples": shard_summary[f"{name}_opponent"]["samples"],
            }
        )

    training_seconds = time.time() - training_started

    # 途中で1モデルでも失敗した場合はここへ到達しない。全32出力が揃った
    # ディレクトリだけを累計episode checkpointとして公開する。
    os.replace(temporary_checkpoint_dir, checkpoint_dir)

    for name, src in agents:
        _copy_atomic(checkpoint_dir / name / "model.pth", src / "model.pth")
        _copy_atomic(
            checkpoint_dir / name / "opponent_model.pth",
            src / "opponent_model.pth",
        )

    summary = {
        "episodes": [str(Path(path).resolve()) for path in episodes],
        "agents": trained_models,
        "preprocessSeconds": preprocess_seconds,
        "trainingSeconds": training_seconds,
        "elapsedSeconds": time.time() - started,
        "epochs": epochs,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "device": device,
        "checkpointDir": str(checkpoint_dir),
    }
    (work_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"episode checkpoint training done: preprocess={preprocess_seconds:.1f}s "
        f"training={training_seconds:.1f}s total={summary['elapsedSeconds']:.1f}s"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--agent", type=parse_agent, action="append", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "更新後の全agentパラメータを保存するディレクトリ。省略時は"
            "agent共通ルート/model_episodeXXXXへ保存する"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--shard-size", type=int, default=20_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_parallel_generation(
        args.episodes,
        args.agent,
        args.work_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        shard_size=args.shard_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
