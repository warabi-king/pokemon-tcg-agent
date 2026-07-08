"""Cloud Run entry point for up to five simultaneous human-vs-human rooms."""

from __future__ import annotations

import atexit
import csv
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.room_manager import RoomManager, available_decks  # noqa: E402


APP_ROOT = ROOT / "app"
CARD_ROOT = ROOT / "docs" / "cards"


def card_metadata() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    path = ROOT / "data" / "EN_Card_Data.csv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            try:
                card_id = int(row["Card ID"])
            except (KeyError, TypeError, ValueError):
                continue
            result.setdefault(
                card_id,
                {"id": card_id, "name": row.get("Card Name") or f"Card #{card_id}"},
            )
    return result


app = Flask(__name__, static_folder=None)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_urlsafe(48)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("K_SERVICE")),
)
manager = RoomManager(max_rooms=5)
CARD_META = card_metadata()
atexit.register(manager.close_all)


def session_player() -> tuple[str, int]:
    room_id = session.get("room_id")
    role = session.get("role")
    if not isinstance(room_id, str) or type(role) is not int or role not in (0, 1):
        raise ValueError("対戦ルームに参加していません。")
    return room_id, role


@app.get("/")
def lobby():
    manager.cleanup()
    return send_from_directory(APP_ROOT, "rooms.html")


@app.get("/duel")
def duel():
    try:
        room_id, _ = session_player()
        manager.get(room_id)
    except Exception:
        return redirect(url_for("lobby"))
    return send_from_directory(APP_ROOT, "index.html")


@app.get("/static/<path:filename>")
def static_file(filename: str):
    return send_from_directory(APP_ROOT, filename)


@app.get("/cards/<path:filename>")
def card_image(filename: str):
    return send_from_directory(CARD_ROOT, filename)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/meta")
def metadata():
    return jsonify({"cards": CARD_META, "attacks": {}, "agents": available_decks()})


@app.get("/api/rooms/status")
def room_status():
    manager.cleanup()
    return jsonify({"activeRooms": len(manager.rooms), "maxRooms": manager.max_rooms})


@app.post("/api/rooms")
def create_room():
    try:
        body = request.get_json(force=True) or {}
        room = manager.create(
            [str(body.get("deckA") or ""), str(body.get("deckB") or "")],
        )
        session.clear()
        session["room_id"] = room.room_id
        session["role"] = 0
        return jsonify(manager.state(room.room_id, 0))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/rooms/join")
def join_room():
    try:
        body = request.get_json(force=True) or {}
        room = manager.join(str(body.get("roomId") or ""))
        session.clear()
        session["room_id"] = room.room_id
        session["role"] = 1
        return jsonify(manager.state(room.room_id, 1))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/duel/state")
def duel_state():
    try:
        room_id, role = session_player()
        return jsonify(manager.state(room_id, role))
    except Exception as exc:
        return jsonify({"started": False, "cloudMode": True, "error": str(exc)}), 401


@app.post("/api/duel/action")
def duel_action():
    try:
        room_id, role = session_player()
        body = request.get_json(force=True) or {}
        return jsonify(manager.action(room_id, role, body.get("indices"), body.get("step")))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
