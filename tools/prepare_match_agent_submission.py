"""Prepare a match_agents learner as a Kaggle-submittable agent.

This script builds agents/match_agents/{name}/src from:

- executable code copied from agents/rl_mcts/src
- deck.csv and model.pth copied from agents/match_agents/{name}

It then calls tools/build_submission.py to create the tar.gz archive.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = ROOT / "agents"
MATCH_AGENTS_ROOT = AGENTS_ROOT / "match_agents"
DIST_ROOT = ROOT / "dist"

CODE_FILES = ("main.py",)
CODE_DIRS = ("cg", "rl_mcts")
PARAM_FILES = ("deck.csv", "model.pth")
OPTIONAL_PARAM_FILES = ("opponent_model.pth",)


def ignore_generated(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 card IDs: {path} ({len(deck)})")
    return deck


def discover_match_agents(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"match_agents directory not found: {root}")
    names = sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "deck.csv").exists())
    if not names:
        raise FileNotFoundError(f"No match agent folders with deck.csv found under: {root}")
    return names


def validate_source(template_src: Path, match_agent_dir: Path) -> None:
    if not template_src.is_dir():
        raise FileNotFoundError(f"template src directory not found: {template_src}")

    missing_code: list[str] = []
    for name in CODE_FILES:
        if not (template_src / name).is_file():
            missing_code.append(name)
    for name in CODE_DIRS:
        if not (template_src / name).is_dir():
            missing_code.append(name)
    if missing_code:
        raise FileNotFoundError("template src is missing: " + ", ".join(missing_code))

    missing_params = [name for name in PARAM_FILES if not (match_agent_dir / name).is_file()]
    if missing_params:
        raise FileNotFoundError(f"{match_agent_dir} is missing: " + ", ".join(missing_params))

    read_deck(match_agent_dir / "deck.csv")


def prepare_src(template_src: Path, match_agent_dir: Path) -> Path:
    validate_source(template_src, match_agent_dir)

    target_src = match_agent_dir / "src"
    target_src.mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        shutil.copy2(template_src / name, target_src / name)

    for name in CODE_DIRS:
        shutil.copytree(
            template_src / name,
            target_src / name,
            dirs_exist_ok=True,
            ignore=ignore_generated,
        )

    for name in PARAM_FILES:
        shutil.copy2(match_agent_dir / name, target_src / name)
    for name in OPTIONAL_PARAM_FILES:
        source = match_agent_dir / name
        if source.is_file():
            shutil.copy2(source, target_src / name)

    return target_src


def build_submission(match_agent_name: str, output: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "build_submission.py"),
        "--agent",
        f"match_agents/{match_agent_name}",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent", action="append", help="match_agents subfolder name, e.g. 00. Can be repeated.")
    target.add_argument("--all", action="store_true", help="prepare every match_agents subfolder with deck.csv.")
    parser.add_argument(
        "--match-agents-root",
        type=Path,
        default=MATCH_AGENTS_ROOT,
        help="directory containing 00, 01, ... learner folders.",
    )
    parser.add_argument(
        "--template-agent",
        default="rl_mcts",
        help="agent whose src code is copied as the executable template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIST_ROOT,
        help="directory for submission_match_agent_{name}.tar.gz files.",
    )
    parser.add_argument("--no-build", action="store_true", help="only prepare src/, do not create tar.gz.")
    parser.add_argument("--dry-run", action="store_true", help="print planned actions without copying or building.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_src = AGENTS_ROOT / args.template_agent / "src"
    names = discover_match_agents(args.match_agents_root) if args.all else args.agent

    for name in names:
        match_agent_dir = args.match_agents_root / name
        output = args.output_dir / f"submission_match_agent_{name}.tar.gz"
        print(f"Preparing match agent {name}")
        print(f"  template: {template_src}")
        print(f"  source:   {match_agent_dir}")
        print(f"  target:   {match_agent_dir / 'src'}")
        if not args.no_build:
            print(f"  archive:  {output}")

        if args.dry_run:
            validate_source(template_src, match_agent_dir)
            continue

        prepare_src(template_src, match_agent_dir)
        if not args.no_build:
            build_submission(name, output)


if __name__ == "__main__":
    main()
