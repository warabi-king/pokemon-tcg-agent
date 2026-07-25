"""Notebookなどから監視するため、学習中のバッチLossを逐次保存する。"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path


class LiveLossRecorder:
    """バッチLossの追記CSVと現在状態のJSONをrun directoryへ保存する。"""

    def __init__(
        self,
        run_dir: Path,
        agent_names: list[str],
        *,
        target_loss: float | None,
    ) -> None:
        self.csv_path = run_dir / "live_loss_batches.csv"
        self.status_path = run_dir / "live_status.json"
        self.agent_names = agent_names
        self.target_loss = target_loss
        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                (
                    "timestamp",
                    "iteration",
                    "agent",
                    "batch",
                    "batches",
                    "batch_loss",
                    "running_loss",
                    "value_loss",
                    "policy_loss",
                )
            )
        self.update_status("initializing", "学習の初期化中")

    def update_status(
        self,
        phase: str,
        message: str,
        *,
        iteration: int | None = None,
        agent: str | None = None,
        batch: int = 0,
        batches: int = 0,
        completed: bool = False,
    ) -> None:
        status = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "phase": phase,
            "message": message,
            "iteration": iteration,
            "agent": agent,
            "batch": batch,
            "batches": batches,
            "completed": completed,
            "agents": self.agent_names,
            "target_loss": self.target_loss,
        }
        temporary_path = self.status_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.status_path)

    def record_batch(
        self,
        *,
        iteration: int,
        agent: str,
        batch: int,
        batches: int,
        batch_loss: float,
        running_loss: float,
        value_loss: float,
        policy_loss: float,
    ) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                (
                    datetime.now().isoformat(timespec="seconds"),
                    iteration,
                    agent,
                    batch,
                    batches,
                    batch_loss,
                    running_loss,
                    value_loss,
                    policy_loss,
                )
            )
        self.update_status(
            "training",
            f"{agent} を更新中",
            iteration=iteration,
            agent=agent,
            batch=batch,
            batches=batches,
        )

    def finish_training(self, message: str = "学習が完了しました") -> None:
        self.update_status("completed", message, completed=True)
