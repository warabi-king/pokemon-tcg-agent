"""Prepare a match_agents learner as a Kaggle-submittable agent.

This script builds agents/match_agents/{name}/src from:

- executable code copied from agents/rl_mcts/src
- deck.csv and model.pth copied from agents/match_agents/{name}

If agents/match_agents/{name}/opponent_model.pth exists, it is also bundled and
main.py is generated so the agent uses model.pth for its own turns and
opponent_model.pth for the opponent nodes during MCTS search (self+opp pair).
Without it, the plain self-only template main.py is copied as before.

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
OPP_PARAM_FILE = "opponent_model.pth"  # 任意: あれば探索時の相手モデルとして同梱・配線する

# opponent_model.pth がある場合に生成する main.py（self+opp を配線）。
_OPP_MAIN_TEMPLATE = '''from __future__ import annotations

from pathlib import Path

import rl_mcts
from cg.api import Observation, to_observation_class
from rl_mcts.agent import RlMctsAgent
from rl_mcts.deck import read_deck_csv

# Kaggleはmain.pyをexecでロードし__file__を定義しないため、main.py内で__file__は使えない。
# インポート済みモジュール(rl_mcts)の__file__からsrc/直下を解決する。
# 自分の手番は model.pth、MCTS探索木の相手手番は opponent_model.pth で評価する。
_SRC = Path(rl_mcts.__file__).resolve().parent.parent
_AGENT = RlMctsAgent(
    model_path=_SRC / "model.pth",
    opponent_model_path=_SRC / "opponent_model.pth",
    search_count={search_count},
)


def agent(obs_dict: dict) -> list[int]:
    """Kaggle/cabtから呼び出されるエージェント本体。"""
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return _AGENT.select_action(obs_dict)
'''


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


def has_opponent_model(match_agent_dir: Path) -> bool:
    return (match_agent_dir / OPP_PARAM_FILE).is_file()


def prepare_src(template_src: Path, match_agent_dir: Path, search_count: int = 50) -> Path:
    validate_source(template_src, match_agent_dir)

    target_src = match_agent_dir / "src"
    target_src.mkdir(parents=True, exist_ok=True)

    for name in CODE_DIRS:
        shutil.copytree(
            template_src / name,
            target_src / name,
            dirs_exist_ok=True,
            ignore=ignore_generated,
        )

    for name in PARAM_FILES:
        shutil.copy2(match_agent_dir / name, target_src / name)

    if has_opponent_model(match_agent_dir):
        # 探索時の相手モデルを同梱し、self+opp を配線した main.py を生成する。
        shutil.copy2(match_agent_dir / OPP_PARAM_FILE, target_src / OPP_PARAM_FILE)
        (target_src / "main.py").write_text(
            _OPP_MAIN_TEMPLATE.format(search_count=search_count), encoding="utf-8"
        )
    else:
        # opp が無ければ従来どおりテンプレートの self-only main.py をコピー。
        for name in CODE_FILES:
            shutil.copy2(template_src / name, target_src / name)

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
    parser.add_argument(
        "--search-count",
        type=int,
        default=50,
        help="MCTS search count wired into the generated main.py when opponent_model.pth is bundled.",
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
        opp = has_opponent_model(match_agent_dir)
        print(f"  opponent: {'bundled (self+opp, search_count=%d)' % args.search_count if opp else 'none (self-only)'}")
        if not args.no_build:
            print(f"  archive:  {output}")

        if args.dry_run:
            validate_source(template_src, match_agent_dir)
            continue

        prepare_src(template_src, match_agent_dir, search_count=args.search_count)
        if not args.no_build:
            build_submission(name, output)


if __name__ == "__main__":
    main()
