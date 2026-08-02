"""KaggleエピソードJSONを、模倣学習に必要な情報だけのJSONへ変換する。

出力JSONは ``preprocess_episodes.py --episodes`` にそのまま渡せる。
元のsteps、ログ、表示情報、時間、status、途中rewardなどは削除し、各局面は
NN入力に使う盤面・合法手と正解クラスだけを保持する。

使用例:
    python tools/train/minimize_episodes.py \
        --episodes day1.zip day2.zip \
        --output-dir episodes_minimal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_ROOT))

from episode_io import iter_multi_source  # noqa: E402
from imitation_data import minimize_episode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        nargs="+",
        required=True,
        help="エピソード.json、ディレクトリ、or .zip(複数指定可)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="出力先に既存のepisode_*.jsonがある場合に上書きを許可する",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing = sorted(args.output_dir.glob("episode_*.json")) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise SystemExit(f"出力先に既存の最小JSONがあります: {existing[0]} (--overwriteで上書き)")
    if existing:
        for path in existing:
            path.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inspected = 0
    saved = 0
    skipped = 0
    input_bytes = 0
    output_bytes = 0
    decisions = 0
    started = time.time()

    for _source, name, data in iter_multi_source(args.episodes):
        if Path(name).name in {"manifest.csv", "minimal_manifest.json"}:
            continue
        if args.max_episodes is not None and inspected >= args.max_episodes:
            break
        inspected += 1
        input_bytes += len(data)

        try:
            minimal = minimize_episode(data)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            minimal = None
        if minimal is None:
            skipped += 1
            continue

        encoded = json.dumps(minimal, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        output_path = args.output_dir / f"episode_{saved:08d}.json"
        output_path.write_bytes(encoded)
        output_bytes += len(encoded)
        decisions += len(minimal["decisions"])
        saved += 1

        if inspected % 200 == 0:
            print(f"episodes={inspected} saved={saved} decisions={decisions}", flush=True)

    manifest = {
        "format": "pokemon-tcg-agent/imitation-minimal-manifest-v1",
        "episodesInspected": inspected,
        "episodesSaved": saved,
        "episodesSkipped": skipped,
        "decisions": decisions,
        "inputBytes": input_bytes,
        "outputBytes": output_bytes,
        "sizeRatio": output_bytes / input_bytes if input_bytes else 0.0,
        "elapsedSeconds": time.time() - started,
    }
    manifest_path = args.output_dir / "minimal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"done: inspected={inspected} saved={saved} skipped={skipped} decisions={decisions}")
    print(f"size: {input_bytes:,} -> {output_bytes:,} bytes ({manifest['sizeRatio']:.1%})")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
