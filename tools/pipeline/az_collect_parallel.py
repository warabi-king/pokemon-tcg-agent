"""AZ サンプル収集をプロセス並列化する（CPU律速の shared-batch collect を並列に回す）。

単一プロセスの collect_batched_training_samples は libcg 状態遷移が1コア直列でCPU律速。
本モジュールは対戦カードを workers 個のプロセスへ分割し、各プロセスが自分の担当分を
shared-batch collect（プロセス内でNNをGPUバッチ化）で回して、samples/opp_samples を
統合する。各ワーカーが全32モデルを GPU に載せる（重み共有はしないが 8GB GPU に収まる）。

使い方は generation_az から collect_parallel(...) を呼ぶ。
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path


def _worker(args):
    """1ワーカー: 担当ペアリングを collect し (samples, opp_samples, n_done) を返す。"""
    agent_specs, pairings, params, canonical_src = args
    import sys
    for p in ("tools", "tools/train", "agents/rl_mcts/src"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    from rl_mcts.checkpoint import load_state_dict_and_temperature
    from rl_mcts.model import create_model
    from batched_training import BatchedTrainingAgent, collect_batched_training_samples

    try:
        torch.set_num_threads(max(1, params["threads"]))
    except Exception:
        pass
    device = torch.device(params["device"])

    def _load(pth):
        m = create_model()
        state, temperature = load_state_dict_and_temperature(pth, map_location=device)
        m.load_state_dict(state)
        # 埋め込み温度が無い(旧形式)重みは自己対戦系の既定値10.0にフォールバック。
        return m, (10.0 if temperature is None else temperature)

    agents = []
    for name, self_pth, opp_pth, deck in agent_specs:
        model, policy_temperature = _load(self_pth)
        opponent_model, opponent_policy_temperature = _load(opp_pth)
        agents.append(
            BatchedTrainingAgent(
                name=name, deck=deck,
                model=model, opponent_model=opponent_model,
                policy_temperature=policy_temperature,
                opponent_policy_temperature=opponent_policy_temperature,
            )
        )
    out = collect_batched_training_samples(
        agents, pairings, games=params["games"],
        canonical_src=Path(canonical_src), device=device,
        batch_size=params["batch_size"], lanes=params["lanes"],
        search_count=params["search_count"], lambda_value=params["lambda_value"],
        seed=params["seed"],
    )
    done = sum(1 for r in out.results if r.result is not None)
    return out.samples, out.opp_samples, done, len(out.results)


def collect_parallel(
    agent_specs: list[tuple[str, str, str, list[int]]],
    pairings: list[tuple[str, str]],
    *,
    canonical_src: str,
    workers: int,
    games: int,
    device: str,
    batch_size: int,
    lanes: int,
    search_count: int,
    lambda_value: float,
    threads: int,
    seed: int = 0,
):
    """samples/opp_samples を統合して返す。workers<=1 なら単一プロセス。

    agent_specs: [(name, self_pth, opp_pth, deck_list), ...]
    """
    names = [s[0] for s in agent_specs]
    samples: dict[str, list] = {n: [] for n in names}
    opp_samples: dict[str, list] = {n: [] for n in names}

    params = dict(games=games, device=device, batch_size=batch_size, lanes=lanes,
                  search_count=search_count, lambda_value=lambda_value,
                  threads=threads, seed=seed)

    if workers <= 1 or len(pairings) <= 1:
        s, o, done, total = _worker((agent_specs, pairings, params, canonical_src))
        for n in names:
            samples[n] += s.get(n, [])
            opp_samples[n] += o.get(n, [])
        return samples, opp_samples, done, total

    # ペアリングをラウンドロビン分割（負荷を均す）。seed はワーカーごとにずらす。
    chunks = [pairings[i::workers] for i in range(workers)]
    jobs = [
        (agent_specs, chunk, {**params, "seed": seed + i}, canonical_src)
        for i, chunk in enumerate(chunks) if chunk
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        results = pool.map(_worker, jobs)

    done_total = 0
    match_total = 0
    for s, o, done, total in results:
        done_total += done
        match_total += total
        for n in names:
            samples[n] += s.get(n, [])
            opp_samples[n] += o.get(n, [])
    return samples, opp_samples, done_total, match_total
