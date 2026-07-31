"""世代ループ1回分: 梱包 → ①リーグ(履歴生成) → 履歴で self/opp を継続学習 → 次世代。

途中評価・採否ゲートは無し（設定どおり）。学習データが得られなかったエージェントは
前世代の重みをそのまま次世代へ引き継ぐ（copy-forward）。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import config
from deck_utils import load_agents, read_deck_csv, write_deck_csv, exact_deck_filter
from package_agent import package_agent
from preprocess import episode_sources, preprocess
from trainer import train_model


def run_league(manifest_path: Path, out_dir: Path) -> None:
    """① リーグ（黒箱）を呼ぶ。PIPE_LEAGUE_CMD 未設定ならスタブを使う。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if config.LEAGUE_CMD:
        cmd = config.LEAGUE_CMD.replace("{manifest}", str(manifest_path)).replace("{out}", str(out_dir))
        print(f"[league] $ {cmd}")
        subprocess.run(cmd, shell=True, check=True, cwd=str(config.REPO_ROOT))
    else:
        import league_stub  # 遅延 import（kaggle_environments 読み込みを必要時のみ）
        print("[league] PIPE_LEAGUE_CMD 未設定のためスタブで対戦（動作確認用）")
        league_stub.run_stub(manifest_path, out_dir, games=config.LEAGUE_STUB_GAMES)


def _learn_side(
    role: str,
    deck: list[int],
    sources: list[Path],
    shards_dir: Path,
    prev_model: Path,
    out_model: Path,
    metrics_file: Path,
) -> None:
    """role(own/opponent) の履歴を前処理→継続学習。データ無しなら前世代をコピー。"""
    n = preprocess(sources, shards_dir, exact_deck_filter(deck, role),
                   config.SHARD_SIZE, role=role, label=f"{out_model.parent.name}/{role}") if sources else 0
    initial = prev_model if config.WARM_START else None
    trained = train_model(shards_dir, out_model, config.EPOCHS_PER_GEN,
                          initial_model=initial, metrics_file=metrics_file) if n else False
    if not trained:
        if prev_model.exists():
            shutil.copy2(prev_model, out_model)  # copy-forward
            print(f"  [learn] {out_model.parent.name}/{role}: データ無し→前世代を引き継ぎ")
        else:
            print(f"  [learn] {out_model.parent.name}/{role}: 前世代重みも無く更新不可")


def run_generation(g: int, root: Path | None = None) -> Path:
    root = config.ROOT if root is None else root
    agents = load_agents(root / "clusters.json")
    gen_g = root / f"gen_{g:03d}"
    gen_next = root / f"gen_{g + 1:03d}"
    logs_dir = gen_g / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1) 梱包 + manifest
    manifest: list[dict] = []
    for a in agents:
        name = a["name"]
        agdir = gen_g / "agents" / name
        self_pth, opp_pth, deck_csv = agdir / "self.pth", agdir / "opp.pth", agdir / "deck.csv"
        if not (self_pth.exists() and opp_pth.exists() and deck_csv.exists()):
            print(f"[gen {g}] {name}: モデル/デッキ不足のため除外")
            continue
        src = agdir / "src"
        package_agent(self_pth, opp_pth, deck_csv, src)
        manifest.append({"name": name, "src": str(src), "deck": str(deck_csv)})
    manifest_path = gen_g / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gen {g}] 梱包 {len(manifest)} 体 -> {manifest_path}")

    if len(manifest) < 2:
        raise SystemExit(f"[gen {g}] 対戦可能なエージェントが2体未満です。Phase0 の出力を確認してください。")

    # 2) ① リーグで履歴生成
    episodes_dir = gen_g / "episodes"
    run_league(manifest_path, episodes_dir)
    sources = episode_sources(episodes_dir)
    if not sources:
        raise SystemExit(f"[gen {g}] リーグ履歴が {episodes_dir} に生成されませんでした。")

    # 3) 履歴で self/opp を継続学習 → 次世代
    for a in agents:
        name = a["name"]
        prev = gen_g / "agents" / name
        if not (prev / "self.pth").exists():
            continue
        deck = read_deck_csv(prev / "deck.csv")
        out = gen_next / "agents" / name
        out.mkdir(parents=True, exist_ok=True)
        write_deck_csv(out / "deck.csv", deck)
        _learn_side("own", deck, sources, gen_g / "shards" / f"{name}_own",
                    prev / "self.pth", out / "self.pth", logs_dir / f"{name}_self.csv")
        _learn_side("opponent", deck, sources, gen_g / "shards" / f"{name}_opp",
                    prev / "opp.pth", out / "opp.pth", logs_dir / f"{name}_opp.csv")

    print(f"[gen {g}] 完了 -> {gen_next}")
    return gen_next


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-g", "--generation", type=int, default=0)
    args = parser.parse_args()
    run_generation(args.generation)
