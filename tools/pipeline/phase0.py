"""Phase0: 公式リプレイ履歴からの模倣学習で各エージェントの初期 {self, opp} を作る。

通常は orchestrate.py 経由で呼ばれる（`--generations 0` で Phase0 のみ）。
単体実行: `python tools/pipeline/phase0.py`。前提: 公式リプレイを PIPE_OFFICIAL_EPISODES
（既定 episodes/official/）に配置。関係する環境変数: PIPE_WORKERS(前処理並列),
PIPE_SHARD_SIZE, PIPE_PHASE0_EPOCHS, PIPE_SIM_THRESHOLD, PIPE_MIN_AVAIL_MB。

前処理は preprocess_all で **全エピソードを1回だけ走査**し、全クラスタの own/opp シャードを
同時生成する（旧版は agent×role ごとに全リプレイを再スキャンしていた）。
完了マーカー(shards/.preprocess_done.json, agents/<name>/.self_done/.opp_done)により、
クラッシュ後に同じコマンドで途中から再開できる。

PRETRAINED（agents/match_agents/imitation_group0-2、shards/ 由来の self+opp 重み）は
**全クラスタで積極的に再利用**する:

- self: 近いデッキ（既存 PRETRAINED と類似度 >= SIM_THRESHOLD）があれば self モデルを
        コピーして流用（事前学習の再実行を省略）。
        しきい値未満でも、own シャードで学習する際は最寄り PRETRAINED の self を
        warm-start 初期値として使う（ランダム初期化より近い重みから始める）。
- opp : 常に opp シャードで学習するが、self と同様に最寄り PRETRAINED の opp を
        warm-start 初期値として使う。

出力: <ROOT>/gen_000/agents/<name>/{self.pth, opp.pth, deck.csv}
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import config
from deck_utils import load_agents, nearest_deck, read_deck_csv, write_deck_csv
from preprocess import episode_sources
from preprocess_multi import preprocess_all
from trainer import train_model


def _pretrained_registry() -> list[dict]:
    """PRETRAINED から {name, deck, self_model, opp_model} のリストを作る。

    opp_model(opponent_model.pth) が無い場合は None のまま残し、warm-start を
    self のみに限定する（古い PRETRAINED エントリとの後方互換）。
    """
    reg: list[dict] = []
    for name, root in config.PRETRAINED.items():
        root = Path(root)
        deck_csv = root / "deck.csv"
        self_model = root / "model.pth"
        opp_model = root / "opponent_model.pth"
        if deck_csv.exists() and self_model.exists():
            reg.append({
                "name": name,
                "deck": read_deck_csv(deck_csv),
                "self_model": self_model,
                "opp_model": opp_model if opp_model.exists() else None,
            })
    return reg


def run_phase0(root: Path | None = None) -> Path:
    root = config.ROOT if root is None else root
    agents = load_agents(root / "clusters.json")
    gen0 = root / "gen_000"
    agents_dir = gen0 / "agents"
    shards_root = gen0 / "shards"
    logs_dir = gen0 / "logs"
    for d in (agents_dir, shards_root, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    pretrained = _pretrained_registry()
    sources = episode_sources(config.OFFICIAL_EPISODES)
    # 前処理の完了マーカー。存在すれば前処理をスキップして学習から再開できる
    # （やり直したい場合はこのファイルか shards ディレクトリごと削除する）。
    preprocess_done = shards_root / ".preprocess_done.json"
    if not sources:
        print(
            f"[phase0] 警告: 公式リプレイが {config.OFFICIAL_EPISODES} に見つかりません。"
            " self のコピーは行いますが、履歴からの新規学習（opp / 近いデッキ無しの self）は"
            " スキップされます。PIPE_OFFICIAL_EPISODES を設定してください。"
        )
    elif preprocess_done.exists():
        print(f"[phase0] 前処理スキップ（完了マーカーあり: {preprocess_done}）")
    else:
        # 前回クラッシュ時の中途半端なシャードが混ざらないよう、作り直す。
        for d in list(shards_root.iterdir()):
            if d.is_dir():
                shutil.rmtree(d)
        # 全クラスタの own/opp シャードを 1 パスで同時生成（並列: PIPE_WORKERS）。
        print(f"[phase0] 前処理（単一パス）: clusters={len(agents)} workers={config.WORKERS}")
        preprocess_all(
            sources, shards_root, reps=agents,
            threshold=config.SIM_THRESHOLD, shard_size=config.SHARD_SIZE,
            workers=config.WORKERS, value_decay=config.VALUE_DECAY,
        )
        preprocess_done.write_text(
            json.dumps({
                "episodes": str(config.OFFICIAL_EPISODES),
                "sources": len(sources),
                "clusters": len(agents),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for a in agents:
        name = a["name"]
        deck = a["deck"]
        out = agents_dir / name
        out.mkdir(parents=True, exist_ok=True)
        write_deck_csv(out / "deck.csv", deck)
        self_pth = out / "self.pth"
        opp_pth = out / "opp.pth"

        # 完了マーカー（クラッシュ後の再実行時、完了済みの side をスキップして再開する）。
        self_done = out / ".self_done"
        opp_done = out / ".opp_done"

        # 最寄りの PRETRAINED（shards/ 由来の group0-2）を探す。しきい値の判定は
        # self の完全コピーにのみ使い、warm-start には常に最寄りを使う（積極活用）。
        best, sim = nearest_deck(deck, pretrained) if pretrained else (None, -1.0)

        # --- self: 近いデッキがあれば完全コピー、無ければ own シャード + warm-start 学習 ---
        if self_done.exists() and self_pth.exists():
            print(f"[phase0] {name}: self スキップ（完了済み）")
        elif best is not None and sim >= config.SIM_THRESHOLD:
            shutil.copy2(best["self_model"], self_pth)
            print(f"[phase0] {name}: self <- コピー {best['name']} (sim={sim:.3f})")
            self_done.touch()
        else:
            warm = best["self_model"] if best is not None else None
            if warm is not None:
                print(f"[phase0] {name}: self <- warm-start {best['name']} (sim={sim:.3f})")
            trained = train_model(shards_root / f"{name}_own", self_pth, config.PHASE0_EPOCHS,
                        initial_model=warm,
                        metrics_file=logs_dir / f"{name}_self.csv")
            if not trained and warm is not None:
                # own シャードが空（対象デッキが履歴に無い等）でも、積極活用の方針上
                # 何も持たないより最寄り PRETRAINED をそのまま採用する。
                shutil.copy2(warm, self_pth)
                print(f"[phase0] {name}: self <- シャード無しのためフォールバックコピー {best['name']}")
            self_done.touch()

        # --- opp: 常に opp シャード + 最寄り PRETRAINED の opp を warm-start ---
        if opp_done.exists() and opp_pth.exists():
            print(f"[phase0] {name}: opp  スキップ（完了済み）")
        else:
            warm_opp = best["opp_model"] if best is not None else None
            if warm_opp is not None:
                print(f"[phase0] {name}: opp  <- warm-start {best['name']} (sim={sim:.3f})")
            trained_opp = train_model(shards_root / f"{name}_opp", opp_pth, config.PHASE0_EPOCHS,
                        initial_model=warm_opp,
                        metrics_file=logs_dir / f"{name}_opp.csv")
            if not trained_opp and warm_opp is not None:
                shutil.copy2(warm_opp, opp_pth)
                print(f"[phase0] {name}: opp  <- シャード無しのためフォールバックコピー {best['name']}")
            opp_done.touch()

    print(f"[phase0] 完了 -> {agents_dir}")
    return gen0


if __name__ == "__main__":
    run_phase0()
