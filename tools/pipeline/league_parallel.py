"""① リーグ（並列版）: manifest のエージェントを総当たり対戦させ、各試合を
full kaggle episode JSON（env.toJSON()）として out_dir に書き出す。

league_stub.py と同じ **kaggle steps 形式**（extract_samples_from_episode が読める、
観測を簡約しない完全形式）を、プロセス並列で生成する点だけが違う。

なぜ GPUバッチ版(batched_tournament)の episode を使わないか:
  batched_tournament の学習JSON(imitation-episode-v1)は観測をカードIDへ簡約しており、
  pipeline の get_encoder_input が要求する HP・付随エネルギー・状態異常・トラッシュ・
  手札などを保持しない。そのまま preprocess に渡すと Phase0/推論と特徴量が食い違う。
  そこで env.toJSON() の完全観測をプロセス並列で作り、特徴量の一致を保証する。

PIPE_LEAGUE_CMD 例:
  export PIPE_LEAGUE_CMD='python tools/pipeline/league_parallel.py --agents {manifest} --out {out} --games 50 --workers 6'
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import config  # noqa: F401  (sys.path 設定の副作用)
import run_matches_round_robin as rr

# ワーカー用グローバル（initializer で設定）。
_OUT_DIR: Path | None = None


@dataclass(frozen=True)
class _Req:
    idx: int
    a_name: str
    a_src: str
    a_deck: str
    b_name: str
    b_src: str
    b_deck: str
    swap: bool


def _init_worker(out_dir: str, threads: int) -> None:
    global _OUT_DIR
    _OUT_DIR = Path(out_dir)
    # ワーカーを増やすほどCPUが取り合いになるため、1ワーカーあたりのthreadを絞る。
    try:
        import torch
        torch.set_num_threads(max(1, threads))
    except Exception:
        pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(max(1, threads)))


def _run_one(req: _Req) -> str | None:
    """1試合を実行し、episode JSON を書き出す。エラー時は説明文字列を返す。"""
    from kaggle_environments import make

    mod_a = f"lp_{os.getpid()}_{req.idx}_a"
    mod_b = f"lp_{os.getpid()}_{req.idx}_b"
    try:
        a = rr.load_agent(
            rr.AgentSpec(req.a_name, Path(req.a_src) / "main.py", Path(req.a_deck)), mod_a
        )
        b = rr.load_agent(
            rr.AgentSpec(req.b_name, Path(req.b_src) / "main.py", Path(req.b_deck)), mod_b
        )
        env = make("cabt")
        funcs = [b.func, a.func] if req.swap else [a.func, b.func]
        env.run(funcs)
        episode = env.toJSON()
        if not isinstance(episode, str):
            episode = json.dumps(episode, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return f"{req.a_name} vs {req.b_name} #{req.idx}: {exc}"
    finally:
        # sys.modules の肥大とモジュール間汚染を避けるため、都度片付ける。
        for m in (mod_a, mod_b):
            sys.modules.pop(m, None)

    assert _OUT_DIR is not None
    (_OUT_DIR / f"ep_{req.idx:07d}.json").write_text(episode, encoding="utf-8")
    return None


def build_requests(entries: list[dict], games: int, include_self: bool) -> list[_Req]:
    """manifest から (総当たり × games) の試合リクエストを作る（先手/後手を交互）。"""
    by_name = {e["name"]: e for e in entries}
    names = list(by_name)
    pairs = list(itertools.combinations(names, 2))
    if include_self:
        pairs += [(n, n) for n in names]

    reqs: list[_Req] = []
    idx = 0
    for a_name, b_name in pairs:
        a, b = by_name[a_name], by_name[b_name]
        for gi in range(games):
            reqs.append(_Req(
                idx=idx,
                a_name=a_name, a_src=a["src"], a_deck=a["deck"],
                b_name=b_name, b_src=b["src"], b_deck=b["deck"],
                swap=(gi % 2 == 1),
            ))
            idx += 1
    return reqs


def run_league_parallel(
    manifest_path: Path,
    out_dir: Path,
    games: int,
    workers: int,
    include_self: bool,
    threads_per_worker: int,
) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    reqs = build_requests(entries, games, include_self)
    print(f"[league_parallel] agents={len(entries)} pairings×games={len(reqs)} "
          f"workers={workers} threads/worker={threads_per_worker}", flush=True)

    t0 = time.time()
    written = 0
    errors = 0
    if workers <= 1:
        _init_worker(str(out_dir), threads_per_worker)
        for r in reqs:
            err = _run_one(r)
            if err:
                errors += 1
                print(f"  [league] エラー {err}", flush=True)
            else:
                written += 1
    else:
        # maxtasksperchild でワーカーを定期再起動し、torch/モジュールのメモリ蓄積を抑える。
        with Pool(workers, initializer=_init_worker,
                  initargs=(str(out_dir), threads_per_worker),
                  maxtasksperchild=100) as pool:
            for i, err in enumerate(pool.imap_unordered(_run_one, reqs, chunksize=1), 1):
                if err:
                    errors += 1
                    print(f"  [league] エラー {err}", flush=True)
                else:
                    written += 1
                if i % 200 == 0:
                    print(f"  [league_parallel] {i}/{len(reqs)} "
                          f"elapsed={time.time()-t0:.1f}s written={written} errors={errors}",
                          flush=True)

    print(f"[league_parallel] 完了: {written} episodes, {errors} errors, "
          f"{time.time()-t0:.1f}s -> {out_dir}", flush=True)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", type=Path, required=True, help="manifest JSON（{name,src,deck}の配列）")
    p.add_argument("--out", type=Path, required=True, help="episode JSON の出力先ディレクトリ")
    p.add_argument("--games", type=int, default=int(os.environ.get("PIPE_LEAGUE_GAMES", "10")),
                   help="各対戦カードの試合数（先手/後手を交互）")
    p.add_argument("--workers", type=int, default=int(os.environ.get("PIPE_LEAGUE_WORKERS", "6")),
                   help="並列プロセス数")
    p.add_argument("--threads-per-worker", type=int,
                   default=int(os.environ.get("PIPE_LEAGUE_THREADS", "2")),
                   help="各ワーカーのtorch/BLASスレッド数")
    p.add_argument("--no-self", action="store_true", help="自己対戦カードを除外する")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_league_parallel(
        args.agents, args.out, games=args.games, workers=args.workers,
        include_self=not args.no_self, threads_per_worker=args.threads_per_worker,
    )


if __name__ == "__main__":
    main()
