"""学習ループから定期的に呼び出す、強さ評価とその履歴保存。

これまでの並列self-play学習notebookは、各iteration/updateでtraining loss・
accuracyしか記録しておらず、「lossが下がった」ことと「実際に強くなった」ことを
区別できなかった。このモジュールは既存の ``tools/run_matches_round_robin.py``
（worker-batched backend）をそのまま評価専用で呼び出し、固定baseline
（既定では ``agents/random``）を含めたラウンドロビン結果を取得して、
学習中のcsvに追記する。

学習用self-playとは別プロセス呼び出しであり、self-play専用の行動選択温度・
rootノイズ環境変数は設定しない（＝実戦と同じ決定的なvisit最大選択で評価する）。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Mapping

import pandas as pd

STRENGTH_HISTORY_COLUMNS = [
    "completed_episodes",
    "update",
    "agent",
    "games",
    "wins",
    "losses",
    "draws",
    "unresolved",
    "win_rate",
    "decided_win_rate",
    "win_rate_vs_baseline",
]


def run_round_robin_eval(
    *,
    python: Path,
    runner: Path,
    match_root: Path,
    agents: Mapping[str, Path],
    baseline_name: str,
    baseline_agent: Path,
    workers: int,
    lanes: int,
    batch_size: int,
    search_count: int,
    device: str,
    seed: int,
    games_per_pairing: int,
    max_turns: int,
    max_selections: int,
) -> dict:
    """baselineを含めたラウンドロビンを1回実行し、tournament jsonの中身を返す。

    self-play専用の行動選択温度・rootノイズは有効化しない（環境変数を渡さない）
    ため、評価は常に決定的なvisit最大選択で行われる。
    """
    results_root = match_root / "results"
    before = (
        {path.name for path in results_root.glob("tournament_*.json")}
        if results_root.is_dir()
        else set()
    )

    all_agents: dict[str, Path] = dict(agents)
    all_agents[baseline_name] = baseline_agent

    command = [str(python), str(runner)]
    for name, main_path in all_agents.items():
        command.extend(["--agent", f"{name}={main_path}"])
    command.extend(
        [
            "--backend",
            "worker-batched",
            "--device",
            device,
            "--workers",
            str(workers),
            "--lanes",
            str(lanes),
            "--batch-size",
            str(batch_size),
            "--search-count",
            str(search_count),
            "--max-turns",
            str(max_turns),
            "--max-selections",
            str(max_selections),
            "--games",
            str(games_per_pairing),
            "--seed",
            str(seed),
            "--no-self",
            "--quiet",
            "--save-json",
        ]
    )

    started = time.perf_counter()
    subprocess.run(command, cwd=match_root, check=True)
    elapsed = time.perf_counter() - started

    after = (
        {path.name for path in results_root.glob("tournament_*.json")}
        if results_root.is_dir()
        else set()
    )
    created = sorted(after - before)
    if not created:
        raise RuntimeError("評価ラウンドロビンのtournament json出力が見つかりません。")
    # ファイル名は tournament_{unix時刻}.json なので文字列順=生成順になる。
    latest_path = results_root / created[-1]
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    payload["_eval_elapsed_seconds"] = elapsed
    payload["_eval_json_path"] = str(latest_path)
    return payload


def _win_rate_vs_baseline(head_to_head: Mapping[str, dict], agent_name: str, baseline_name: str) -> float | None:
    forward = head_to_head.get(f"{agent_name}_vs_{baseline_name}")
    if forward is not None:
        total = forward["name0_wins"] + forward["name1_wins"] + forward["draws"] + forward["unresolved"]
        if total <= 0:
            return None
        return forward["name0_wins"] / total
    backward = head_to_head.get(f"{baseline_name}_vs_{agent_name}")
    if backward is not None:
        total = backward["name0_wins"] + backward["name1_wins"] + backward["draws"] + backward["unresolved"]
        if total <= 0:
            return None
        return backward["name1_wins"] / total
    return None


def append_strength_history(
    csv_path: Path,
    completed_episodes: int,
    update_index: int,
    tournament_payload: Mapping,
    baseline_name: str,
) -> pd.DataFrame:
    """tournament jsonのoverall/head_to_headから、agentごとの勝率行を追記する。"""
    overall = tournament_payload["overall"]
    head_to_head = tournament_payload["head_to_head"]

    rows = []
    for name, record in overall.items():
        if name == baseline_name:
            continue
        rows.append(
            {
                "completed_episodes": completed_episodes,
                "update": update_index,
                "agent": name,
                "games": record["games"],
                "wins": record["wins"],
                "losses": record["losses"],
                "draws": record["draws"],
                "unresolved": record["unresolved"],
                "win_rate": record["win_rate"],
                "decided_win_rate": record["decided_win_rate"],
                "win_rate_vs_baseline": _win_rate_vs_baseline(
                    head_to_head, name, baseline_name
                ),
            }
        )

    frame = pd.DataFrame(rows, columns=STRENGTH_HISTORY_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    frame.to_csv(csv_path, mode="a", header=write_header, index=False)
    return frame
