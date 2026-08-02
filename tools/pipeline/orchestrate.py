"""自己対戦パイプラインのオーケストレータ（入口）。

クラスタ→16エージェント生成 → Phase0(模倣事前学習) → 世代ループ×G(対戦学習)。
設定はすべて環境変数（既定値は config.py）。

実行例:
    # エージェント生成だけ確認（clusters.json と16デッキを出すだけ）
    python tools/pipeline/orchestrate.py --dry-run

    # ① 模倣学習 Phase0 のみ（gen_000 を作る。--generations 0 で世代ループを回さない）
    PIPE_WORKERS=3 python tools/pipeline/orchestrate.py --skip-gen-agents --generations 0

    # ② 対戦学習 世代ループのみ（既存 gen_000 起点。AlphaZero）
    PIPE_GEN_BACKEND=az python tools/pipeline/orchestrate.py \\
        --skip-gen-agents --skip-phase0 --no-keep-intermediate

    # ③ 既存 gen_005 から継続学習（gen_005/agents/<name>/ に全エージェント分の
    #    self.pth・opp.pth・deck.csv が揃っている必要がある）
    PIPE_GEN_BACKEND=az python tools/pipeline/orchestrate.py \\
        --skip-gen-agents --skip-phase0 --start-gen 5 --no-keep-intermediate

CLI引数:
    --dry-run            エージェント生成のみ（Phase0/世代ループを行わない）
    --skip-gen-agents    clusters.json/decks 生成をスキップ
    --skip-phase0        Phase0 をスキップ（既存 gen_000 を使う）
    --generations N      世代数の上書き（既定は PIPE_GENERATIONS。0でPhase0のみ）
    --start-gen N        世代ループの開始世代番号（既定0）。既存 gen_N から継続する場合に指定
    --[no-]keep-intermediate  中間世代・shards・episodes を残す/消費後に削除

主な環境変数（既定値は config.py 参照）:
    共通  : PIPE_GENERATIONS, PIPE_EPOCHS_PER_GEN, PIPE_LR, PIPE_BATCH_SIZE, PIPE_ROOT,
            PIPE_OFFICIAL_EPISODES, PIPE_KEEP_INTERMEDIATE
    Phase0: PIPE_PHASE0_EPOCHS, PIPE_SIM_THRESHOLD, PIPE_WARM_START, PIPE_SHARD_SIZE,
            PIPE_WORKERS, PIPE_MIN_AVAIL_MB
    世代  : PIPE_GEN_BACKEND(league|az), PIPE_LEAGUE_GAMES, PIPE_LEAGUE_SEARCH_COUNT,
            PIPE_LEAGUE_INCLUDE_SELF
    league: PIPE_LEAGUE_CMD（{manifest} {out} を置換する外部リーグコマンド）
    az    : PIPE_AZ_COLLECT_WORKERS, PIPE_AZ_COLLECT_THREADS, PIPE_LANES,
            PIPE_LAMBDA_VALUE, PIPE_INFER_BATCH_SIZE
"""

from __future__ import annotations

import argparse
import shutil

import config
import gen_agents
import generation
import phase0


def _prune_generation(g: int, root) -> None:
    """2世代前の gen_g を削除して中間成果物のディスクを回収する。

    呼び出し側は「1つ前の世代は保険として残し、2つ前を消す」ラグで呼ぶ
    （途中でクラッシュしても直前世代の重みからやり直せるようにするため）。

    - gen_g/shards・gen_g/episodes は常に純粋な中間物なので削除。
    - g>=1 の gen_g は agents（中間世代の重み）ごと削除する。
    - g==0（gen_000）は Phase0 のベースライン兼再開マーカーであり、
      学習方法も異なる世代なので agents は絶対に削除しない
      （重い shards/episodes だけ削除）。呼び出し側も g==0 では呼ばない想定だが、
      誤って呼ばれても安全なようにここでも二重にガードする。
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
        "--start-gen", type=int, default=0,
        help="世代ループの開始世代番号（既定0）。既存の gen_N から継続学習する場合に指定する"
             "（gen_N/agents/<name>/ に self.pth・opp.pth・deck.csv が全エージェント分揃っている必要がある）。"
             " 通常 --skip-gen-agents --skip-phase0 と併用する。",
    )
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
    end_gen = args.start_gen + generations
    for g in range(args.start_gen, end_gen):
        print(f"\n========== generation {g} / {end_gen} ==========")
        run_gen(g, root)
        if not args.keep_intermediate:
            # gen_{g+1} が出来た時点で gen_g は消費済みだが、クラッシュ時の
            # フォールバックとして1つ前(gen_g)は残し、2つ前(gen_{g-1})だけ削除する。
            # gen_000 は学習方法が異なるベースラインなので削除対象にしない。
            if g - 1 >= 1:
                _prune_generation(g - 1, root)

    print(f"\n完了: 最終世代 = gen_{end_gen:03d}（{root}）")


if __name__ == "__main__":
    main()
