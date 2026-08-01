"""AlphaZero版の世代ループ1回: batched gpu-tree で対戦＋サンプル収集 → self/opp を学習。

現行 generation.py（梱包→棋譜→preprocess→模倣学習）の高速・強化版。

- 各エージェントを self.pth(model)＋opp.pth(opponent_model) で読み込み、
  batched エンジンで全対戦を同時進行（NN評価をGPUバッチ化, lanes 本同時）。
- 収集サンプルを「打ち手→self」「相手→opp」へ二重ルーティング
  （collect_batched_training_samples の opp_samples）。
- naoki の AlphaZero 損失(train_one_iteration_local: value=Huber回帰,
  policy=MCTS訪問分布へのマスク付きHuber)で self/opp を継続学習(warm-start)。
- gen_{g+1}/agents/<name>/{self.pth,opp.pth} へ保存。データ不足は前世代を引き継ぐ。

相手デッキ推定は batched 側の deck_belief（候補DB復元, ②方式）が使われる。
"""

from __future__ import annotations

import itertools
from pathlib import Path

import torch

import config
from deck_utils import load_agents, read_deck_csv, write_deck_csv

from rl_mcts.model import create_model  # noqa: E402
from rl_mcts.mcts import LearnInput, MAX_ACTIONS  # noqa: E402
from batched_training import BatchedTrainingAgent, collect_batched_training_samples  # noqa: E402
from run_train_round_robin import train_one_iteration_local  # noqa: E402

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

    # 1) エージェント読み込み（self=model, opp=opponent_model）
    bt_agents: list[BatchedTrainingAgent] = []
    for a in agents_meta:
        name = a["name"]
        agdir = gen_g / "agents" / name
        if not ((agdir / "self.pth").exists() and (agdir / "opp.pth").exists()
                and (agdir / "deck.csv").exists()):
            print(f"[gen_az {g}] {name}: モデル/デッキ不足のため除外")
            continue
        deck = read_deck_csv(agdir / "deck.csv")
        bt_agents.append(BatchedTrainingAgent(
            name=name, deck=deck,
            model=_load_model(agdir / "self.pth", device),
            opponent_model=_load_model(agdir / "opp.pth", device),
        ))
    if len(bt_agents) < 2:
        raise SystemExit(f"[gen_az {g}] 対戦可能なエージェントが2体未満です。")

    names = [a.name for a in bt_agents]
    pairings = list(itertools.combinations(names, 2))
    if config.LEAGUE_INCLUDE_SELF:
        pairings += [(n, n) for n in names]

    # 2) batched 対戦＋サンプル収集（self/opp 二重ルーティング）
    print(f"[gen_az {g}] collect: agents={len(names)} pairings={len(pairings)} "
          f"games={config.LEAGUE_GAMES} lanes={config.LANES} "
          f"search={config.LEAGUE_SEARCH_COUNT} device={device}", flush=True)
    out = collect_batched_training_samples(
        bt_agents, pairings, games=config.LEAGUE_GAMES,
        canonical_src=config.TEMPLATE_SRC, device=device,
        batch_size=config.INFER_BATCH_SIZE, lanes=config.LANES,
        search_count=config.LEAGUE_SEARCH_COUNT, lambda_value=config.LAMBDA_VALUE,
        seed=g,
    )
    completed = sum(1 for r in out.results if r.result is not None)
    print(f"[gen_az {g}] 対戦完了={completed}/{len(out.results)} "
          f"self合計={sum(len(v) for v in out.samples.values())} "
          f"opp合計={sum(len(v) for v in out.opp_samples.values())}", flush=True)

    # 3) self/opp を AlphaZero 損失で継続学習 → gen_{g+1}
    import shutil
    for a in bt_agents:
        name = a.name
        out_dir = gen_next / "agents" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        write_deck_csv(out_dir / "deck.csv", a.deck)
        prev = gen_g / "agents" / name

        for side, model, sample_list, prev_pth in (
            ("self", a.model, out.samples[name], prev / "self.pth"),
            ("opp", a.opponent_model, out.opp_samples[name], prev / "opp.pth"),
        ):
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
