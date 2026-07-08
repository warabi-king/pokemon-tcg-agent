"""Lifecycle manager for private Cloud Run agent matches."""

from __future__ import annotations

import multiprocessing
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.cloud_agent_worker import worker_main

ROOT = Path(__file__).resolve().parents[1]


def available_agents() -> list[str]:
    return sorted(
        path.name for path in (ROOT / "agents").iterdir()
        if (path / "src" / "main.py").is_file()
        and (path / "src" / "deck.csv").is_file()
    )


class AgentWorkerClient:
    def __init__(
        self,
        mode: str,
        agent_names: list[str],
        human_deck: list[int] | str | None = None,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.lock = threading.Lock()
        self.process = context.Process(
            target=worker_main,
            args=(child, mode, agent_names, human_deck),
            daemon=True,
            name=f"cabt-{mode}-{secrets.token_hex(3)}",
        )
        self.process.start()
        child.close()
        if not self.connection.poll(90):
            self.close(force=True)
            raise RuntimeError("対戦エンジンの起動がタイムアウトしました。")
        response = self.connection.recv()
        if not response.get("ok") or not response.get("ready"):
            self.close(force=True)
            raise RuntimeError(response.get("error") or "対戦エンジンを起動できませんでした。")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.process.is_alive():
                raise RuntimeError("対戦エンジンが停止しました。")
            self.connection.send(payload)
            if not self.connection.poll(60):
                raise RuntimeError("対戦エンジンから応答がありません。")
            response = self.connection.recv()
            if not response.get("ok"):
                raise ValueError(response.get("error") or "対戦処理に失敗しました。")
            return response.get("result") or {}

    def close(self, force: bool = False) -> None:
        if self.process.is_alive() and not force:
            try:
                with self.lock:
                    self.connection.send({"operation": "shutdown"})
                    if self.connection.poll(3):
                        self.connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        self.connection.close()


@dataclass
class HostedMatch:
    token: str
    mode: str
    worker: AgentWorkerClient
    last_activity: float = field(default_factory=time.monotonic)


@dataclass
class CompletedMatch:
    mode: str
    result: dict[str, Any]
    finished_at: float = field(default_factory=time.monotonic)


class CloudAgentManager:
    def __init__(self, total_active: Callable[[], int], max_matches: int = 5) -> None:
        self.total_active = total_active
        self.max_matches = max_matches
        self.matches: dict[str, HostedMatch] = {}
        self.completed: dict[str, CompletedMatch] = {}
        self.lock = threading.RLock()

    def cleanup(self) -> None:
        now = time.monotonic()
        with self.lock:
            expired = [
                token for token, match in self.matches.items()
                if now - match.last_activity > 30 * 60 or not match.worker.process.is_alive()
            ]
            matches = [self.matches.pop(token) for token in expired]
            completed_expired = [
                token for token, match in self.completed.items()
                if now - match.finished_at > 10 * 60
            ]
            for token in completed_expired:
                self.completed.pop(token)
        for match in matches:
            match.worker.close()

    def create(
        self,
        mode: str,
        agent_names: list[str],
        human_deck: list[int] | str | None = None,
    ) -> tuple[HostedMatch, dict[str, Any]]:
        choices = set(available_agents())
        expected = 1 if mode == "play" else 2 if mode == "watch" else 0
        if len(agent_names) != expected or any(name not in choices for name in agent_names):
            raise ValueError("エージェントの指定が不正です。")
        self.cleanup()
        if self.total_active() >= self.max_matches:
            raise ValueError("現在満室です。同時に実行できる対戦は5つまでです。")
        if isinstance(human_deck, str) and human_deck not in choices:
            raise ValueError("デッキの指定が不正です。")
        if isinstance(human_deck, list) and (
            len(human_deck) != 60
            or any(type(card_id) is not int or card_id <= 0 for card_id in human_deck)
        ):
            raise ValueError("CSVにはカードIDを60件指定してください。")
        worker = AgentWorkerClient(mode, agent_names, human_deck)
        token = secrets.token_urlsafe(32)
        match = HostedMatch(token, mode, worker)
        with self.lock:
            self.matches[token] = match
        result = worker.request({"operation": "state"})
        if result.get("finished"):
            self._complete(match, result)
        return match, result

    def _complete(self, match: HostedMatch, result: dict[str, Any]) -> None:
        with self.lock:
            self.matches.pop(match.token, None)
            self.completed[match.token] = CompletedMatch(match.mode, result)
        match.worker.close()

    def request(self, token: str, mode: str, operation: str, **payload: Any) -> dict[str, Any]:
        self.cleanup()
        with self.lock:
            match = self.matches.get(token)
            completed = self.completed.get(token)
        if completed is not None and completed.mode == mode:
            if operation == "state":
                return completed.result
            raise ValueError("This match has finished.")
        if match is None or match.mode != mode:
            raise ValueError("対戦情報がありません。新しい対戦を開始してください。")
        result = match.worker.request({"operation": operation, **payload})
        match.last_activity = time.monotonic()
        if result.get("finished"):
            self._complete(match, result)
        return result

    def close_all(self) -> None:
        with self.lock:
            matches = list(self.matches.values())
            self.matches.clear()
            self.completed.clear()
        for match in matches:
            match.worker.close()
