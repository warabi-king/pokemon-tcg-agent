"""In-memory five-room coordinator for the Cloud Run deployment."""

from __future__ import annotations

import multiprocessing
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.duel_worker import worker_main


ROOT = Path(__file__).resolve().parents[1]
ROOM_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def available_decks() -> list[str]:
    agents_root = ROOT / "agents"
    return sorted(
        path.name
        for path in agents_root.iterdir()
        if (path / "src" / "deck.csv").is_file()
    )


class WorkerClient:
    def __init__(self, deck_names: list[str]) -> None:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.lock = threading.Lock()
        self.process = context.Process(
            target=worker_main,
            args=(child, deck_names),
            daemon=True,
            name=f"cabt-room-{secrets.token_hex(3)}",
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
            if not self.connection.poll(30):
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
class Room:
    room_id: str
    deck_names: list[str]
    worker: WorkerClient
    player_tokens: tuple[str, str] = field(
        default_factory=lambda: (secrets.token_urlsafe(32), secrets.token_urlsafe(32))
    )
    player2_joined: bool = False
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


class RoomManager:
    def __init__(self, max_rooms: int = 5) -> None:
        self.max_rooms = max_rooms
        self.rooms: dict[str, Room] = {}
        self.lock = threading.RLock()

    def _room_code(self) -> str:
        while True:
            code = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(8))
            if code not in self.rooms:
                return code

    def cleanup(self) -> None:
        now = time.monotonic()
        expired: list[str] = []
        with self.lock:
            for room_id, room in self.rooms.items():
                waiting_expired = not room.player2_joined and now - room.created_at > 15 * 60
                inactive_expired = now - room.last_activity > 30 * 60
                finished_expired = room.finished_at is not None and now - room.finished_at > 10 * 60
                if waiting_expired or inactive_expired or finished_expired or not room.worker.process.is_alive():
                    expired.append(room_id)
            for room_id in expired:
                room = self.rooms.pop(room_id)
                room.worker.close()

    def create(self, deck_names: list[str]) -> Room:
        choices = set(available_decks())
        if len(deck_names) != 2 or any(name not in choices for name in deck_names):
            raise ValueError("デッキの指定が不正です。")
        self.cleanup()
        with self.lock:
            if len(self.rooms) >= self.max_rooms:
                raise ValueError("現在満室です。同時に作成できるルームは5室までです。")
            room_id = self._room_code()
            worker = WorkerClient(deck_names)
            room = Room(room_id, list(deck_names), worker)
            self.rooms[room_id] = room
            return room

    def join(self, room_id: str) -> Room:
        self.cleanup()
        normalized = (room_id or "").strip().upper()
        with self.lock:
            room = self.rooms.get(normalized)
            if room is None:
                raise ValueError("指定されたルームが見つかりません。")
            if room.player2_joined:
                raise ValueError("このルームにはすでに2人参加しています。")
            room.player2_joined = True
            room.last_activity = time.monotonic()
            return room

    def get(self, room_id: str) -> Room:
        self.cleanup()
        with self.lock:
            room = self.rooms.get(room_id)
            if room is None:
                raise ValueError("対戦ルームが終了したか、サーバーが再起動されました。")
            return room

    def authenticate(self, token: str) -> tuple[str, int]:
        if not token:
            raise ValueError("プレイヤー認証情報がありません。ロビーから入り直してください。")
        self.cleanup()
        with self.lock:
            for room in self.rooms.values():
                for role, expected in enumerate(room.player_tokens):
                    if secrets.compare_digest(token, expected):
                        return room.room_id, role
        raise ValueError("プレイヤー認証情報が無効です。ロビーから入り直してください。")

    def state(self, room_id: str, role: int) -> dict[str, Any]:
        room = self.get(room_id)
        result = room.worker.request({"operation": "state", "role": role})
        room.last_activity = time.monotonic()
        if result.get("finished") and room.finished_at is None:
            room.finished_at = time.monotonic()
        return self._payload(room, role, result)

    def action(self, room_id: str, role: int, indices: Any, step: Any) -> dict[str, Any]:
        room = self.get(room_id)
        if not room.player2_joined:
            raise ValueError("Player 2の参加を待っています。")
        result = room.worker.request(
            {"operation": "action", "role": role, "indices": indices, "step": step}
        )
        room.last_activity = time.monotonic()
        if result.get("finished") and room.finished_at is None:
            room.finished_at = time.monotonic()
        return self._payload(room, role, result)

    def _payload(self, room: Room, role: int, result: dict[str, Any]) -> dict[str, Any]:
        return {
            **result,
            "started": True,
            "cloudMode": True,
            "duelMode": True,
            "roomId": room.room_id,
            "role": role,
            "opponentJoined": room.player2_joined if role == 0 else True,
            "humanTurn": room.player2_joined and bool(result.get("active")),
            "deckNames": room.deck_names,
        }

    def close_all(self) -> None:
        with self.lock:
            rooms = list(self.rooms.values())
            self.rooms.clear()
        for room in rooms:
            room.worker.close()
