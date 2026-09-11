"""self モデル + opp モデル + deck を、実行可能な src/ 一式に梱包する。

prepare_match_agent_submission.py の拡張版。生成する main.py は
RlMctsAgent(model_path=self, opponent_model_path=opp, search_count=...) を配線し、
「自分の手番は self、探索木の相手手番は opp」で動くエージェントにする。

src/ の構成:
  main.py            … 配線済みエントリ
  cg/ rl_mcts/       … テンプレート(agents/rl_mcts/src)からコピー
  model.pth          … self
  opponent_model.pth … opp
  deck.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config

CODE_DIRS = ("cg", "rl_mcts")

_MAIN_TEMPLATE = '''from __future__ import annotations

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
    obs: Observation = to_observation_class(obs_dict)
    if obs.select is None:
        return read_deck_csv()
    return _AGENT.select_action(obs_dict)
'''


def _ignore_pyc(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n == "__pycache__" or n.endswith(".pyc")}


def package_agent(
    self_pth: Path,
    opp_pth: Path,
    deck_csv: Path,
    out_src: Path,
    search_count: int = None,  # type: ignore[assignment]
    template_src: Path = None,  # type: ignore[assignment]
) -> Path:
    search_count = config.SEARCH_COUNT if search_count is None else search_count
    template_src = config.TEMPLATE_SRC if template_src is None else template_src

    for p, what in ((self_pth, "self"), (opp_pth, "opp"), (deck_csv, "deck")):
        if not Path(p).exists():
            raise FileNotFoundError(f"{what} が見つかりません: {p}")

    out_src.mkdir(parents=True, exist_ok=True)
    for name in CODE_DIRS:
        shutil.copytree(template_src / name, out_src / name, dirs_exist_ok=True, ignore=_ignore_pyc)
    shutil.copy2(self_pth, out_src / "model.pth")
    shutil.copy2(opp_pth, out_src / "opponent_model.pth")
    shutil.copy2(deck_csv, out_src / "deck.csv")
    (out_src / "main.py").write_text(_MAIN_TEMPLATE.format(search_count=search_count), encoding="utf-8")
    return out_src


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self", dest="self_pth", type=Path, required=True)
    parser.add_argument("--opp", dest="opp_pth", type=Path, required=True)
    parser.add_argument("--deck", dest="deck_csv", type=Path, required=True)
    parser.add_argument("--out", dest="out_src", type=Path, required=True)
    parser.add_argument("--search-count", type=int, default=config.SEARCH_COUNT)
    args = parser.parse_args()
    out = package_agent(args.self_pth, args.opp_pth, args.deck_csv, args.out_src, args.search_count)
    print(f"梱包完了: {out}")


if __name__ == "__main__":
    main()
