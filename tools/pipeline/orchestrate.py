"""自己対戦パイプラインのオーケストレータ（入口）。

設定はすべて環境変数（config.py）。実行:

    # 全体（クラスタ→エージェント生成 → Phase0 → 世代ループ G 回）
    python tools/pipeline/orchestrate.py

    # エージェント生成だけ確認（約63デッキと clusters.json を出すだけ）
    python tools/pipeline/orchestrate.py --dry-run

    # Phase0 をスキップして既存 gen_000 から世代ループだけ回す
    python tools/pipeline/orchestrate.py --skip-gen-agents --skip-phase0

主な環境変数（既定値は config.py 参照）:
    PIPE_GENERATIONS, PIPE_EPOCHS_PER_GEN, PIPE_PHASE0_EPOCHS, PIPE_SEARCH_COUNT,
    PIPE_SIM_THRESHOLD, PIPE_LR, PIPE_BATCH_SIZE, PIPE_WARM_START, PIPE_SHARD_SIZE,
    PIPE_ROOT, PIPE_OFFICIAL_EPISODES, PIPE_DECKGEN_JSONL, PIPE_LEAGUE_CMD
"""

from __future__ import annotations

import argparse
import shutil

import config
import gen_agents
import generation
import phase0


def _prune_generation(g: int, root) -> None:
    """消費済みの世代 gen_g を削除して中間成果物のディスクを回収する。

    - gen_g/shards・gen_g/episodes は常に純粋な中間物なので削除。
    - g>=1 の gen_g は agents（中間世代の重み）ごと削除する。
    - gen_000 は Phase0 のベースライン兼再開マーカーとして agents を残す
      （重い shards/episodes だけ削除）。
    最新世代 gen_{g+1} には触れない。
    """
    gen_dir = root / f"gen_{g:03d}"
    if g >= 1:
        if gen_dir.exists():
            shutil.rmtree(gen_dir)
            print(f"[prune] 中間世代を削除: {gen_dir}")
    else:
        for sub in ("shards", "episodes"):
            d = gen_dir / sub
            if d.exists():
                shutil.rmtree(d)
                print(f"[prune] 中間物を削除: {d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="エージェント生成のみ（Phase0/世代ループを行わない）")
    parser.add_argument("--skip-gen-agents", action="store_true", help="clusters.json/decks 生成をスキップ")
    parser.add_argument("--skip-phase0", action="store_true", help="Phase0 をスキップ（既存 gen_000 を使う）")
    parser.add_argument("--generations", type=int, default=None, help="世代数の上書き（既定は PIPE_GENERATIONS）")
    parser.add_argument(
        "--keep-intermediate",
        action=argparse.BooleanOptionalAction,
        default=config.KEEP_INTERMEDIATE,
        help="中間世代（gen_001..gen_{N-1}）と各世代の shards/episodes を残す。"
             " --no-keep-intermediate で消費後に削除しディスクを節約（既定は PIPE_KEEP_INTERMEDIATE）",
    )
    args = parser.parse_args()

    print(config.summary())
    root = config.ROOT
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_gen_agents:
        gen_agents.generate(config.DECKGEN_JSONL, root)

    if args.dry_run:
        print("dry-run: エージェント生成のみで終了。")
        return

    if not args.skip_phase0:
        phase0.run_phase0(root)

    generations = config.GENERATIONS if args.generations is None else args.generations
    if config.GEN_BACKEND == "az":
        import generation_az
        run_gen = generation_az.run_generation_az
        print(f"[orchestrate] 世代バックエンド=az（batched gpu-tree + AlphaZero, self/opp二重）")
    else:
        run_gen = generation.run_generation
        print(f"[orchestrate] 世代バックエンド=league（梱包→棋譜→模倣学習）")
    for g in range(generations):
        print(f"\n========== generation {g} / {generations} ==========")
        run_gen(g, root)
        if not args.keep_intermediate:
            # gen_{g+1} が出来た時点で gen_g は消費済み。
            _prune_generation(g, root)

    print(f"\n完了: 最終世代 = gen_{generations:03d}（{root}）")


if __name__ == "__main__":
    main()
