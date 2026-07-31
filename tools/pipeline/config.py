"""自己対戦パイプラインの設定。

すべての可変パラメータは **環境変数** でここに集約する（既定値つき）。
プログラム冒頭で読み込まれ、以降のモジュールは `import config` して参照する。

副作用として、pipeline / tools / tools/train / rl_mcts の src を sys.path に追加し、
既存ライブラリ（deck_signature, episode_io, imitation_data, cg, rl_mcts）を
そのまま import できるようにする。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- sys.path 準備（既存ツール/ライブラリを再利用するため） ----
_HERE = Path(__file__).resolve().parent                 # tools/pipeline
_TOOLS = _HERE.parent                                    # tools
REPO_ROOT = _TOOLS.parent                                # pokemon-tcg-agent
_RL_SRC = REPO_ROOT / "agents" / "rl_mcts" / "src"

for _p in (_HERE, _TOOLS, _TOOLS / "train", _RL_SRC):
    _sp = str(_p)
    if _sp not in sys.path:
        sys.path.insert(0, _sp)


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ==== パイプライン設定（環境変数で上書き可能） ====
GENERATIONS = _int("PIPE_GENERATIONS", 5)          # 世代ループ回数 G（固定・途中評価なし）
EPOCHS_PER_GEN = _int("PIPE_EPOCHS_PER_GEN", 3)    # 各世代の train_imitation epochs
PHASE0_EPOCHS = _int("PIPE_PHASE0_EPOCHS", 5)      # Phase0 模倣学習 epochs
LR = _float("PIPE_LR", 3e-4)                       # 学習率
BATCH_SIZE = _int("PIPE_BATCH_SIZE", 128)          # バッチサイズ
SEARCH_COUNT = _int("PIPE_SEARCH_COUNT", 50)       # 梱包エージェントの MCTS 探索回数
SIM_THRESHOLD = _float("PIPE_SIM_THRESHOLD", 0.75) # 近いデッキ判定 & Phase0 クラスタ割当のしきい値
WARM_START = _int("PIPE_WARM_START", 1)            # 1: 世代間で前世代重みから継続学習
SHARD_SIZE = _int("PIPE_SHARD_SIZE", 20000)        # 前処理シャードあたりサンプル数

# ---- パス ----
ROOT = _path("PIPE_ROOT", REPO_ROOT / "pipeline")                                  # 全成果物の根
OFFICIAL_EPISODES = _path("PIPE_OFFICIAL_EPISODES", REPO_ROOT / "episodes" / "official")  # Phase0 用リプレイ
DECKGEN_JSONL = _path(
    "PIPE_DECKGEN_JSONL",
    REPO_ROOT / "deck_generator" / "generated" / "deck_candidates_by_wins.jsonl",
)
TEMPLATE_SRC = _path("PIPE_TEMPLATE_SRC", _RL_SRC)   # 梱包時にコピーする実行コードのテンプレート

# ---- ① リーグ（友人の並列版・黒箱） ----
# manifest.json と出力先を渡すと episode JSON を出すコマンド。
#   実行時に "{manifest}" "{out}" が置換される。空ならスタブ(league_stub.py)を使う。
LEAGUE_CMD = _str("PIPE_LEAGUE_CMD", "")
LEAGUE_STUB_GAMES = _int("PIPE_LEAGUE_STUB_GAMES", 2)  # スタブ時の各カード試合数

# ---- 既存の事前学習済み資産（近いデッキのコピー元）----
# name -> そのデッキと self モデル。opp は新規学習方針のため持たない。
PRETRAINED = {
    "imitation_group0": REPO_ROOT / "agents" / "match_agents" / "imitation_group0",
    "imitation_group1": REPO_ROOT / "agents" / "match_agents" / "imitation_group1",
    "imitation_group2": REPO_ROOT / "agents" / "match_agents" / "imitation_group2",
}

PYTHON = _str("PIPE_PYTHON", sys.executable)  # サブプロセス起動に使う python


def summary() -> str:
    return (
        "=== pipeline config ===\n"
        f"  GENERATIONS={GENERATIONS} EPOCHS_PER_GEN={EPOCHS_PER_GEN} PHASE0_EPOCHS={PHASE0_EPOCHS}\n"
        f"  LR={LR} BATCH_SIZE={BATCH_SIZE} SEARCH_COUNT={SEARCH_COUNT}\n"
        f"  SIM_THRESHOLD={SIM_THRESHOLD} WARM_START={WARM_START} SHARD_SIZE={SHARD_SIZE}\n"
        f"  ROOT={ROOT}\n"
        f"  OFFICIAL_EPISODES={OFFICIAL_EPISODES}\n"
        f"  DECKGEN_JSONL={DECKGEN_JSONL}\n"
        f"  LEAGUE_CMD={'(stub)' if not LEAGUE_CMD else LEAGUE_CMD}\n"
    )
