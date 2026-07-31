"""Phase0: 公式リプレイ履歴からの模倣学習で各エージェントの初期 {self, opp} を作る。

前処理は preprocess_all で **全エピソードを1回だけ走査**し、全クラスタの own/opp シャードを
同時生成する（旧版は agent×role ごとに全リプレイを再スキャンしていた）。

方針:
- self: 近いデッキ（既存 PRETRAINED と類似度 >= SIM_THRESHOLD）があれば、その self モデルを
        コピーして流用（事前学習の再実行を省略）。無ければ own シャードで新規学習。
- opp : 常に opp シャードで新規学習（既存 opp モデルは無いため）。

出力: <ROOT>/gen_000/agents/<name>/{self.pth, opp.pth, deck.csv}
"""

from __future__ import annotations

import shutil
from pathlib import Path

import config
from deck_utils import load_agents, nearest_deck, read_deck_csv, write_deck_csv
from preprocess import episode_sources
from preprocess_multi import preprocess_all
from trainer import train_model


def _pretrained_registry() -> list[dict]:
    """PRETRAINED から {name, deck, self_model} のリストを作る（存在するものだけ）。"""
    reg: list[dict] = []
    for name, root in config.PRETRAINED.items():
        deck_csv = Path(root) / "deck.csv"
        model = Path(root) / "model.pth"
        if deck_csv.exists() and model.exists():
            reg.append({"name": name, "deck": read_deck_csv(deck_csv), "self_model": model})
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
    if not sources:
        print(
            f"[phase0] 警告: 公式リプレイが {config.OFFICIAL_EPISODES} に見つかりません。"
            " self のコピーは行いますが、履歴からの新規学習（opp / 近いデッキ無しの self）は"
            " スキップされます。PIPE_OFFICIAL_EPISODES を設定してください。"
        )
    else:
        # 全クラスタの own/opp シャードを 1 パスで同時生成（並列: PIPE_WORKERS）。
        print(f"[phase0] 前処理（単一パス）: clusters={len(agents)} workers={config.WORKERS}")
        preprocess_all(
            sources, shards_root, reps=agents,
            threshold=config.SIM_THRESHOLD, shard_size=config.SHARD_SIZE,
            workers=config.WORKERS,
        )

    for a in agents:
        name = a["name"]
        deck = a["deck"]
        out = agents_dir / name
        out.mkdir(parents=True, exist_ok=True)
        write_deck_csv(out / "deck.csv", deck)
        self_pth = out / "self.pth"
        opp_pth = out / "opp.pth"

        # --- self: 近いデッキがあればコピー、無ければ own シャードで学習 ---
        best, sim = nearest_deck(deck, pretrained) if pretrained else (None, -1.0)
        if best is not None and sim >= config.SIM_THRESHOLD:
            shutil.copy2(best["self_model"], self_pth)
            print(f"[phase0] {name}: self <- コピー {best['name']} (sim={sim:.3f})")
        else:
            train_model(shards_root / f"{name}_own", self_pth, config.PHASE0_EPOCHS,
                        metrics_file=logs_dir / f"{name}_self.csv")

        # --- opp: 常に opp シャードで学習 ---
        train_model(shards_root / f"{name}_opp", opp_pth, config.PHASE0_EPOCHS,
                    metrics_file=logs_dir / f"{name}_opp.csv")

    print(f"[phase0] 完了 -> {agents_dir}")
    return gen0


if __name__ == "__main__":
    run_phase0()
