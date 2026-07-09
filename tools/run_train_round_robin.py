"""エージェント同士を総当たりで組合せ、各組合せごとに `train.py` を呼び出すスクリプト。

各エージェントは `name=path/to/train.py` の形式で指定します。`path` にディレクトリを
指定した場合は、その中の `train/train.py` または `train.py` を探します。

このスクリプトは各呼び出しで以下の環境変数を設定します（エージェントの `train.py` が
これらを参照できるようにするため）:
  - `TRAIN_ROUND_ROBIN_OPPONENT`: 対戦相手のエージェント名
  - `TRAIN_ROUND_ROBIN_OUTDIR`: その試行の出力先ディレクトリ（エージェント毎のサブディレクトリ）

動作例:
  python tools/run_train_round_robin.py \
    --agent a=agents/rl_mcts_sample \
    --agent b=agents/rl_mcts_sample \
    --no-self --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "train_round_robin"


@dataclass
class TrainSpec:
    name: str
    train_path: Path


def parse_agent_arg(raw: str) -> TrainSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("--agent は name=path の形式で指定してください")
    name, path = raw.split("=", 1)
    name = name.strip()
    p = Path(path.strip())
    if p.is_dir():
        if (p / "train" / "train.py").exists():
            p = p / "train" / "train.py"
        elif (p / "train.py").exists():
            p = p / "train.py"
    return TrainSpec(name=name, train_path=p)


def load_specs_from_config(config: Path) -> List[TrainSpec]:
    data = json.loads(config.read_text(encoding="utf-8"))
    specs: List[TrainSpec] = []
    for entry in data:
        name = entry["name"]
        train = Path(entry["train"]) if entry.get("train") else Path(entry.get("path"))
        specs.append(TrainSpec(name=name, train_path=train))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        action="append",
        type=parse_agent_arg,
        default=None,
        help="エージェント指定。name=path の形式。path は train.py かエージェントルート。",
    )
    parser.add_argument("--config", type=Path, default=None, help="JSON config file")
    parser.add_argument("--no-self", action="store_true", help="自己対戦を除外")
    parser.add_argument("--dry-run", action="store_true", help="実行コマンドを表示するだけ")
    parser.add_argument("--timeout", type=int, default=0, help="各 train.py のタイムアウト秒（0で無制限）")
    parser.add_argument("--parallel", action="store_true", help="各ペア内の2つを並列実行（default: 直列）")
    parser.add_argument(
        "--train-args",
        type=str,
        default="",
        help="train.py に追加で渡す引数（例: \"--iterations 3 --self-play-games 50\"）",
    )
    return parser.parse_args()


def run_train_script(
    spec: TrainSpec,
    opponent: str,
    outdir: Path,
    timeout: int,
    dry_run: bool,
    train_args: list[str],
) -> dict:
    if not spec.train_path.exists():
        raise FileNotFoundError(f"{spec.name}: {spec.train_path} が存在しません。")
    # ensure per-run output/checkpoint dirs
    agent_outdir = outdir / spec.name
    agent_outdir.mkdir(parents=True, exist_ok=True)

    # construct command: include any user-provided train args, and force
    # output model / checkpoint dir to be per-run so existing models aren't reused
    cmd = [sys.executable, str(spec.train_path)] + train_args
    cmd += ["--output-model", str(agent_outdir / "model.pth")]
    cmd += ["--checkpoint-dir", str(agent_outdir / "checkpoints")]
    env = os.environ.copy()
    env["TRAIN_ROUND_ROBIN_OPPONENT"] = opponent
    env["TRAIN_ROUND_ROBIN_OUTDIR"] = str(outdir / spec.name)

    result = {"agent": spec.name, "train_path": str(spec.train_path), "cmd": cmd}


    if dry_run:
        print(
            "DRY RUN:",
            cmd,
            "cwd=",
            spec.train_path.parent,
            "env vars:",
            {"TRAIN_ROUND_ROBIN_OPPONENT": opponent, "TRAIN_ROUND_ROBIN_OUTDIR": env["TRAIN_ROUND_ROBIN_OUTDIR"]},
        )
        result.update({"status": "dry-run"})
        return result

    print(f"Running: {spec.name} ({spec.train_path}) vs {opponent}")
    try:
        subprocess.run(cmd, cwd=spec.train_path.parent, env=env, check=True, timeout=timeout if timeout > 0 else None)
        result["status"] = "ok"
    except subprocess.CalledProcessError as exc:
        result["status"] = "error"
        result["returncode"] = exc.returncode
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"

    return result


def main() -> None:
    args = parse_args()

    # parse additional train args into list for subprocess command
    train_args = shlex.split(args.train_args) if args.train_args else []

    specs: List[TrainSpec] = []
    if args.config:
        specs.extend(load_specs_from_config(args.config))
    if args.agent:
        specs.extend(args.agent)

    if len(specs) < 1:
        raise SystemExit("少なくとも1体のエージェントを指定してください。")

    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"エージェント名が重複しています: {names}")

    spec_map = {s.name: s for s in specs}

    if args.no_self:
        pairings = list(itertools.combinations(names, 2))
    else:
        pairings = list(itertools.combinations_with_replacement(names, 2))

    print(f"Total pairings: {len(pairings)}")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_results = []
    for name0, name1 in pairings:
        timestamp = int(time.time())
        pairing_dir = RESULTS_ROOT / f"{name0}_vs_{name1}_{timestamp}"
        pairing_dir.mkdir(parents=True, exist_ok=True)

        # Run each agent's train.py for this pairing. By default run serially.
        if args.parallel:
            # simple parallel: start both subprocesses and wait
            import threading

            res_container = [None, None]

            def run_idx(i, spec, opp):
                res_container[i] = run_train_script(
                    spec, opp, pairing_dir, args.timeout, args.dry_run, train_args
                )

            t0 = threading.Thread(target=run_idx, args=(0, spec_map[name0], name1))
            t1 = threading.Thread(target=run_idx, args=(1, spec_map[name1], name0))
            t0.start()
            t1.start()
            t0.join()
            t1.join()
            all_results.extend(res_container)
        else:
            all_results.append(
                run_train_script(spec_map[name0], name1, pairing_dir, args.timeout, args.dry_run, train_args)
            )
            # if same agent (self), avoid running twice
            if name0 != name1:
                all_results.append(
                    run_train_script(spec_map[name1], name0, pairing_dir, args.timeout, args.dry_run, train_args)
                )

    # save summary
    out_path = RESULTS_ROOT / f"train_round_robin_{int(time.time())}.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
