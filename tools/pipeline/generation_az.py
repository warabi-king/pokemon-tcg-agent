"""AlphaZero版の世代ループ1回: 自己対戦＋サンプル収集 → self/opp を学習。

現行 generation.py（梱包→棋譜→preprocess→模倣学習）の高速・強化版。
通常は orchestrate.py 経由（PIPE_GEN_BACKEND=az）で呼ばれる。
単体実行: `python tools/pipeline/generation_az.py -g <世代番号>`。

- 各エージェントを self.pth(model)＋opp.pth(opponent_model) のペアで扱い、
  batched エンジンで全対戦を同時進行（NN評価をGPUバッチ化, lanes 本同時）。
  対戦収集は az_collect_parallel.collect_parallel でプロセス並列化
  （PIPE_AZ_COLLECT_WORKERS。単一プロセスは libcg 状態遷移が1コア直列でCPU律速）。
- 収集サンプルを「打ち手→self」「相手→opp」へ二重ルーティング（opp_samples）。
- naoki の AlphaZero 損失(train_one_iteration_local: value=Huber回帰,
  policy=MCTS訪問分布へのマスク付きHuber)で self/opp を継続学習(warm-start)。
- gen_{g+1}/agents/<name>/{self.pth,opp.pth} へ保存。データ不足は前世代を引き継ぐ。

相手デッキ推定は batched 側の deck_belief（候補DB復元, ②方式）が使われる。

主な環境変数(config.py): PIPE_GEN_BACKEND=az, PIPE_LEAGUE_GAMES, PIPE_LEAGUE_SEARCH_COUNT,
    PIPE_AZ_COLLECT_WORKERS, PIPE_AZ_COLLECT_THREADS, PIPE_LANES, PIPE_LAMBDA_VALUE,
    PIPE_INFER_BATCH_SIZE, PIPE_EPOCHS_PER_GEN, PIPE_BATCH_SIZE, PIPE_LR。
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

import config
from deck_utils import load_agents, read_deck_csv, write_deck_csv

from rl_mcts.model import create_model  # noqa: E402
from rl_mcts.mcts import LearnInput, MAX_ACTIONS  # noqa: E402
from run_train_round_robin import train_one_iteration_local  # noqa: E402
from az_collect_parallel import collect_parallel  # noqa: E402

_API_MOD = {"LearnInput": LearnInput, "MAX_ACTIONS": MAX_ACTIONS}


def _load_model(path: Path, device: torch.device):
    model = create_model()
    if Path(path).exists():
        model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)


def _train_side(model, samples, out_path: Path, device: torch.device,
                epochs: int, batch_size: int, lr: float, metrics_file: Path) -> bool:
    """1モデルを AlphaZero 損失で epochs 回学習して保存。成功時 True。

    サンプルが batch_size 未満なら学習せず False（呼び出し側で前世代を引き継ぐ）。
    model は既に前世代の重みなので、そのまま学習＝warm-start 継続学習になる。
    """
    if len(samples) < batch_size:
        return False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        stats = train_one_iteration_local(
            model, optimizer, samples, batch_size, device, _API_MOD,
        )
        with open(metrics_file, "a", encoding="utf-8") as f:
            f.write(f"{ep},{stats.batches},{stats.loss:.6f},"
                    f"{stats.loss_value:.6f},{stats.loss_policy:.6f}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    return True


def run_generation_az(g: int, root: Path | None = None) -> Path:
    root = config.ROOT if root is None else root
    agents_meta = load_agents(root / "clusters.json")
    gen_g = root / f"gen_{g:03d}"
    gen_next = root / f"gen_{g + 1:03d}"
    logs_dir = gen_g / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) エージェント spec（パス）を集める。並列 collect ではワーカーが自分でロードするため、
    #    親プロセスはここではモデルを載せない（collect と学習でメモリのピークを重ねない）。
    specs: list[tuple[str, str, str, list[int]]] = []
    for a in agents_meta:
        name = a["name"]
        agdir = gen_g / "agents" / name
        if not ((agdir / "self.pth").exists() and (agdir / "opp.pth").exists()
                and (agdir / "deck.csv").exists()):
            print(f"[gen_az {g}] {name}: モデル/デッキ不足のため除外")
            continue
        specs.append((name, str(agdir / "self.pth"), str(agdir / "opp.pth"),
                      read_deck_csv(agdir / "deck.csv")))
    if len(specs) < 2:
        raise SystemExit(f"[gen_az {g}] 対戦可能なエージェントが2体未満です。")

    names = [s[0] for s in specs]
    deck_by_name = {s[0]: s[3] for s in specs}
    pairings = list(itertools.combinations(names, 2))
    if config.LEAGUE_INCLUDE_SELF:
        pairings += [(n, n) for n in names]

    # 2) batched 対戦＋サンプル収集（self/opp 二重ルーティング、プロセス並列）
    print(f"[gen_az {g}] collect: agents={len(names)} pairings={len(pairings)} "
          f"games={config.LEAGUE_GAMES} workers={config.AZ_COLLECT_WORKERS} "
          f"lanes={config.LANES} search={config.LEAGUE_SEARCH_COUNT} device={device}", flush=True)
    samples, opp_samples, done, total = collect_parallel(
        specs, pairings, canonical_src=str(config.TEMPLATE_SRC),
        workers=config.AZ_COLLECT_WORKERS, games=config.LEAGUE_GAMES,
        device=str(device), batch_size=config.INFER_BATCH_SIZE, lanes=config.LANES,
        search_count=config.LEAGUE_SEARCH_COUNT, lambda_value=config.LAMBDA_VALUE,
        threads=config.AZ_COLLECT_THREADS, seed=g,
    )
    print(f"[gen_az {g}] 対戦完了={done}/{total} "
          f"self合計={sum(len(v) for v in samples.values())} "
          f"opp合計={sum(len(v) for v in opp_samples.values())}", flush=True)

    # 3) self/opp を AlphaZero 損失で継続学習 → gen_{g+1}
    #    ここで初めて親がモデルをロード（collect ワーカーは終了済みなのでメモリが空いている）。
    import shutil
    for name in names:
        out_dir = gen_next / "agents" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_deck_csv(out_dir / "deck.csv", deck_by_name[name])
        prev = gen_g / "agents" / name

        for side, prev_pth, sample_list in (
            ("self", prev / "self.pth", samples[name]),
            ("opp", prev / "opp.pth", opp_samples[name]),
        ):
            model = _load_model(prev_pth, device)  # 前世代重み＝warm-start 起点
            out_pth = out_dir / f"{side}.pth"
            trained = _train_side(
                model, sample_list, out_pth, device,
                epochs=config.EPOCHS_PER_GEN, batch_size=config.BATCH_SIZE,
                lr=config.LR, metrics_file=logs_dir / f"{name}_{side}.csv",
            )
            if trained:
                print(f"[gen_az {g}] {name}/{side}: 学習 samples={len(sample_list)}")
            else:
                shutil.copy2(prev_pth, out_pth)  # データ不足→前世代を引き継ぎ
                print(f"[gen_az {g}] {name}/{side}: samples={len(sample_list)}<batch → 前世代引き継ぎ")

    print(f"[gen_az {g}] 完了 -> {gen_next}")
    return gen_next


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--generation", type=int, default=0)
    args = parser.parse_args()
    run_generation_az(args.generation)
