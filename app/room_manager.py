"""In-memory five-room coordinator for the Cloud Run deployment."""

from __future__ import annotations

import multiprocessing
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    def __init__(self, deck_names: list[str | list[int]]) -> None:
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
    deck_sources: list[str | list[int]]
    worker: WorkerClient | None
    player_tokens: tuple[str, str] = field(
        default_factory=lambda: (secrets.token_urlsafe(32), secrets.token_urlsafe(32))
    )
    player2_joined: bool = False
    created_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    final_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class RoomManager:
    def __init__(
        self, max_rooms: int = 5, total_active: Callable[[], int] | None = None
    ) -> None:
        self.max_rooms = max_rooms
        self.total_active = total_active
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
                worker_stopped = room.worker is not None and not room.worker.process.is_alive()
                if waiting_expired or inactive_expired or finished_expired or worker_stopped:
                    expired.append(room_id)
            for room_id in expired:
                room = self.rooms.pop(room_id)
                if room.worker is not None:
                    room.worker.close()

    def active_room_count(self) -> int:
        with self.lock:
            return sum(room.worker is not None for room in self.rooms.values())

    def create(self, deck_names: list[str | list[int]]) -> Room:
        choices = set(available_decks())
        valid = len(deck_names) == 2 and all(
            (isinstance(deck, str) and deck in choices)
            or (isinstance(deck, list) and len(deck) == 60 and all(type(card_id) is int and card_id > 0 for card_id in deck))
            for deck in deck_names
        )
        if not valid:
            raise ValueError("デッキの指定が不正です。")
        self.cleanup()
        with self.lock:
            active_count = self.total_active() if self.total_active else self.active_room_count()
            if active_count >= self.max_rooms:
                raise ValueError("現在満室です。同時に作成できるルームは5室までです。")
            room_id = self._room_code()
            worker = WorkerClient(deck_names)
            labels = ["アップロードCSV" if isinstance(deck, list) else deck for deck in deck_names]
            room = Room(room_id, labels, list(deck_names), worker)
            self.rooms[room_id] = room
            return room

    def join(self, room_id: str, deck_source: str | list[int] | None = None) -> Room:
        self.cleanup()
        normalized = (room_id or "").strip().upper()
        with self.lock:
            room = self.rooms.get(normalized)
            if room is None:
                raise ValueError("指定されたルームが見つかりません。")
            if room.player2_joined:
                raise ValueError("このルームにはすでに2人参加しています。")
            if deck_source is not None:
                if isinstance(deck_source, str) and deck_source not in set(available_decks()):
                    raise ValueError("デッキの指定が不正です。")
                if isinstance(deck_source, list) and (
                    len(deck_source) != 60
                    or any(type(card_id) is not int or card_id <= 0 for card_id in deck_source)
                ):
                    raise ValueError("CSVにはカードIDを60件指定してください。")
                room.worker.close()
                room.deck_sources[1] = deck_source
                room.deck_names[1] = "アップロードCSV" if isinstance(deck_source, list) else deck_source
                room.worker = WorkerClient(room.deck_sources)
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
        with room.lock:
            if room.worker is None:
                return self._payload(room, role, room.final_results[role])
            result = room.worker.request({"operation": "state", "role": role})
            room.last_activity = time.monotonic()
            if result.get("finished"):
                self._finish_room(room, role, result)
            return self._payload(room, role, result)

    def action(self, room_id: str, role: int, indices: Any, step: Any) -> dict[str, Any]:
        room = self.get(room_id)
        if not room.player2_joined:
            raise ValueError("Player 2の参加を待っています。")
        with room.lock:
            if room.worker is None:
                return self._payload(room, role, room.final_results[role])
            result = room.worker.request(
                {"operation": "action", "role": role, "indices": indices, "step": step}
            )
            room.last_activity = time.monotonic()
            if result.get("finished"):
                self._finish_room(room, role, result)
            return self._payload(room, role, result)

    def _finish_room(self, room: Room, role: int, result: dict[str, Any]) -> None:
        if room.worker is None:
            return
        room.final_results[role] = result
        other_role = 1 - role
        room.final_results[other_role] = room.worker.request(
            {"operation": "state", "role": other_role}
        )
        room.finished_at = time.monotonic()
        worker = room.worker
        room.worker = None
        worker.close()

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
            if room.worker is not None:
                room.worker.close()
