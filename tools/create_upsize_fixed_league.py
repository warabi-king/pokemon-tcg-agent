"""develop_naoki の upsize 相手群から Phase 2a 用リーグ設定を生成する。

外部 worktree の絶対パスをリポジトリに固定しないため、相手ルートは実行時に渡す。
生成先は ``results/`` などのローカル実験出力にし、そのJSONを
``search_deck_fixed_league.py --opponents`` へ渡す。

実行例:
    python tools/create_upsize_fixed_league.py \\
        --upsize-root /path/to/16model_pretrained_upsize1 \\
        --output results/phase2a-imitation-upsize1-001/opponents.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CLUSTERS = ("03", "04", "05", "11")


def parse_clusters(raw_clusters: str) -> tuple[str, ...]:
    """カンマ区切りクラスタ番号を重複なしの2桁表記へ正規化する。"""

    clusters = tuple(part.strip().zfill(2) for part in raw_clusters.split(",") if part.strip())
    if not clusters:
        raise ValueError("--clusters にはクラスタ番号を1件以上指定してください")
    if len(clusters) != len(set(clusters)):
        raise ValueError("--clusters に同じクラスタ番号を重複指定できません")
    if any(not cluster.isdigit() or len(cluster) != 2 for cluster in clusters):
        raise ValueError("--clusters は 03,04,05,11 のような2桁番号で指定してください")
    return clusters


def build_opponents(upsize_root: Path, clusters: tuple[str, ...], name_prefix: str) -> list[dict[str, object]]:
    """各クラスタの提出可能な src・deck・model を検証して相手定義へ変換する。"""

    opponents: list[dict[str, object]] = []
    for cluster in clusters:
        cluster_root = upsize_root / f"cluster_{cluster}"
        agent_src = cluster_root / "src"
        deck = agent_src / "deck.csv"
        model = agent_src / "model.pth"
        missing = [str(path) for path in (agent_src, deck, model) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"cluster_{cluster} の必要ファイルがありません: {', '.join(missing)}")
        opponents.append(
            {
                "name": f"{name_prefix}_cluster_{cluster}",
                "agent_src": str(agent_src.resolve()),
                "deck": str(deck.resolve()),
                "model": str(model.resolve()),
                "weight": 1.0,
            }
        )
    return opponents


def main() -> None:
    """CLI引数を検証し、固定リーグJSONをローカル実験出力へ保存する。"""

    parser = argparse.ArgumentParser(description="upsize checkpoint群からPhase 2a固定リーグJSONを作る")
    parser.add_argument("--upsize-root", type=Path, required=True, help="16model_pretrained_upsize1 または upsize2 のルート")
    parser.add_argument("--output", type=Path, required=True, help="生成するローカルJSONの保存先")
    parser.add_argument("--clusters", default=",".join(DEFAULT_CLUSTERS), help="使用クラスタ番号。既定: 03,04,05,11")
    parser.add_argument("--name-prefix", default="upsize1", help="結果JSONに記録する相手名の接頭辞")
    parser.add_argument("--dry-run", action="store_true", help="検証結果を表示するだけで保存しない")
    args = parser.parse_args()

    upsize_root = args.upsize_root.resolve()
    if not upsize_root.is_dir():
        raise FileNotFoundError(f"--upsize-root がディレクトリではありません: {upsize_root}")
    opponents = build_opponents(upsize_root, parse_clusters(args.clusters), args.name_prefix)
    content = {
        "format": "pokemon-tcg-agent/fixed-league-v1",
        "purpose": "phase2a_deck_search",
        "opponents": opponents,
    }
    rendered = json.dumps(content, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
