"""① リーグの動作確認用スタブ（本番は友人の並列版に差し替え）。

manifest のエージェントを cabt 環境で総当たり対戦させ、各試合を kaggle episode JSON
（env.toJSON()）として out_dir に書き出す。extract_samples_from_episode が読める形式。

注意: O(N^2) の逐次対戦なので、大量エージェントには不向き。少数での配線確認用。
本番リーグは PIPE_LEAGUE_CMD で指定し、このスタブは使わない想定。
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import config  # noqa: F401 (sys.path 設定)
import run_matches_round_robin as rr
from kaggle_environments import make


def run_stub(manifest_path: Path, out_dir: Path, games: int = 2) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    loaded = []
    for i, e in enumerate(entries):
        spec = rr.AgentSpec(
            name=e["name"],
            agent_path=Path(e["src"]) / "main.py",
            deck_path=Path(e["deck"]),
        )
        loaded.append(rr.load_agent(spec, f"stub_agent_{i}"))

    written = 0
    for a, b in itertools.combinations(range(len(loaded)), 2):
        for gi in range(games):
            # 先手/後手を1試合ごとに入れ替え
            order = (a, b) if gi % 2 == 0 else (b, a)
            env = make("cabt")
            try:
                env.run([loaded[order[0]].func, loaded[order[1]].func])
            except Exception as exc:  # noqa: BLE001
                print(f"  [stub] {loaded[a].spec.name} vs {loaded[b].spec.name} #{gi}: エラー {exc}")
                continue
            episode = env.toJSON()
            if not isinstance(episode, str):
                episode = json.dumps(episode, ensure_ascii=False)
            (out_dir / f"ep_{written:06d}.json").write_text(episode, encoding="utf-8")
            written += 1
    print(f"[stub] {written} episodes -> {out_dir}")
    return written
