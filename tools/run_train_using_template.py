"""Run tools/train.py inside a target agent directory to produce a model file.

This copies `tools/train.py` into the target agent directory (as a temporary
runner) and runs it there so that AGENT_ROOT/SRC_ROOT inside the template
resolve to the agent directory's layout. Use this when you want to apply the
common training script to a particular agent implementation directory.

Example:
  # run training with template inside agents/rl_mcts_r_robin and write model to agents/rl_mcts_sample/src/model.pth
  python tools/run_train_using_template.py \
    --agent-dir agents/rl_mcts_r_robin \
    --output-model agents/rl_mcts_sample/src/model.pth \
    --train-args "--iterations 1 --self-play-games 100"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools" / "train" / "train.py"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent-dir", type=Path, required=True, help="agent implementation dir (contains src/)")
    p.add_argument("--output-model", type=Path, required=True, help="path to write the final model.pth (will be overwritten)")
    p.add_argument("--train-args", type=str, default="", help="extra args passed to train.py")
    p.add_argument("--timeout", type=int, default=0, help="timeout seconds for the train process (0 = none)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not TEMPLATE.exists():
        raise SystemExit(f"Template not found: {TEMPLATE}")
    agent_dir = args.agent_dir
    if not agent_dir.exists():
        raise SystemExit(f"Agent dir not found: {agent_dir}")

    train_args: List[str] = []
    if args.train_args:
        import shlex

        train_args = shlex.split(args.train_args)

    # ensure parent of output model exists
    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    # copy template into agent_dir/.train_run so AGENT_ROOT inside the template
    # resolves to agent_dir when run from there.
    runner_dir = agent_dir / ".train_run"
    runner_dir.mkdir(parents=True, exist_ok=True)
    runner_path = runner_dir / "train.py"
    try:
        shutil.copy2(TEMPLATE, runner_path)
        runner_path = runner_path.resolve()

        cmd = [sys.executable, str(runner_path)] + train_args + ["--output-model", str(args.output_model)]

        print(f"Running template in {agent_dir} -> output {args.output_model}")
        if args.dry_run:
            print("DRY RUN:", cmd, "cwd=", agent_dir)
            return

        subprocess.run(cmd, cwd=agent_dir, check=True, timeout=args.timeout if args.timeout > 0 else None)
        print("Training finished; model written to", args.output_model)
    finally:
        try:
            runner_path.unlink()
            runner_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
