"""official episodes をクラスタ方式で前処理し、shards を出力する（学習はしない）。

phase0.run_phase0 の「前処理（単一パス・全クラスタ routing）」部分だけを切り出した薄い
ドライバ。割引は config.FIRST_MOVE_DISCOUNT（既定 0.3、対局ごとに1手目≈0.3・最終手=±1）。
出力は <out>/<cluster>_own, <out>/<cluster>_opp（train_imitation.py の --shards に渡す）。

--since / --until でエピソードを日付範囲に絞れる（パス中の YYYY-MM-DD を見る）。
継続学習用に「新しく落とした日付だけ」を前処理したいときに使う。

使い方:
    # 07-25〜08-15 の新データだけをクラスタ方式で前処理（割引0.3）
    PIPE_WORKERS=3 python tools/pipeline/preprocess_shards.py \
        --out pipeline/shards_new --since 2026-07-25 --until 2026-08-15

    # 全 official episodes を前処理
    PIPE_WORKERS=3 python tools/pipeline/preprocess_shards.py --out pipeline/shards_all
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from deck_utils import load_agents
from preprocess import episode_sources
from preprocess_multi import preprocess_all

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _date_of(path: Path) -> str:
    m = _DATE.search(str(path))
    return m.group() if m else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="shards 出力先ディレクトリ")
    ap.add_argument("--episodes", type=Path, default=config.OFFICIAL_EPISODES,
                    help=f"episodes ディレクトリ（既定: {config.OFFICIAL_EPISODES}）")
    ap.add_argument("--since", type=str, default=None, help="この日付(YYYY-MM-DD)以降のみ")
    ap.add_argument("--until", type=str, default=None, help="この日付(YYYY-MM-DD)以前のみ")
    ap.add_argument("--workers", type=int, default=config.WORKERS, help=f"並列数（既定 PIPE_WORKERS={config.WORKERS}）")
    args = ap.parse_args()

    sources = episode_sources(args.episodes)
    if args.since or args.until:
        sources = [
            s for s in sources
            if (not args.since or _date_of(s) >= args.since)
            and (not args.until or _date_of(s) <= args.until)
        ]
    if not sources:
        raise SystemExit(f"対象 episodes が見つかりません: {args.episodes} (since={args.since} until={args.until})")

    reps = load_agents(config.ROOT / "clusters.json")
    args.out.mkdir(parents=True, exist_ok=True)

    dates = sorted({_date_of(s) for s in sources})
    print(f"episodes(zip)={len(sources)} 日付範囲={dates[0]}〜{dates[-1]} クラスタ={len(reps)} "
          f"out={args.out} discount={config.FIRST_MOVE_DISCOUNT} shard_size={config.SHARD_SIZE} "
          f"workers={args.workers}", flush=True)

    preprocess_all(
        sources, args.out, reps=reps, threshold=config.SIM_THRESHOLD,
        shard_size=config.SHARD_SIZE, workers=args.workers,
        first_move_discount=config.FIRST_MOVE_DISCOUNT,
    )
    print("PREPROCESS_DONE", flush=True)


if __name__ == "__main__":
    main()
