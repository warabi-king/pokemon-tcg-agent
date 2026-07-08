"""Cloud Run entry point for up to five simultaneous human-vs-human rooms."""

from __future__ import annotations

import atexit
import csv
import os
import secrets
import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, request, send_from_directory, session, url_for

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.room_manager import RoomManager, available_decks  # noqa: E402
from app.cloud_agent_manager import CloudAgentManager, available_agents  # noqa: E402


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
agent_manager = CloudAgentManager(
    lambda: len(manager.rooms) + len(agent_manager.matches), max_matches=5
)
manager.total_active = lambda: len(manager.rooms) + len(agent_manager.matches)
capacity_lock = threading.Lock()
CARD_META = card_metadata()
atexit.register(manager.close_all)
atexit.register(agent_manager.close_all)


def session_player() -> tuple[str, int]:
    player_token = request.headers.get("X-Player-Token", "")
    if player_token:
        return manager.authenticate(player_token)
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


@app.get("/play")
def play():
    return send_from_directory(APP_ROOT, "index.html")


@app.get("/watch")
def watch():
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
    return jsonify({"cards": CARD_META, "attacks": {}, "agents": available_agents()})


@app.get("/api/rooms/status")
def room_status():
    manager.cleanup()
    agent_manager.cleanup()
    return jsonify({
        "activeRooms": len(manager.rooms),
        "activeMatches": len(manager.rooms) + len(agent_manager.matches),
        "maxRooms": manager.max_rooms,
    })


def match_token() -> str:
    token = request.headers.get("X-Match-Token", "")
    if not token:
        raise ValueError("対戦情報がありません。新しい対戦を開始してください。")
    return token


@app.get("/api/state")
def agent_state():
    try:
        return jsonify(agent_manager.request(match_token(), "play", "state"))
    except Exception:
        return jsonify({"started": False, "cloudMode": True})


@app.post("/api/new")
def agent_new_game():
    try:
        body = request.get_json(silent=True) or {}
        with capacity_lock:
            match, payload = agent_manager.create("play", [str(body.get("agent") or "")])
        return jsonify({**payload, "cloudMode": True, "matchToken": match.token})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/action")
def agent_action():
    try:
        body = request.get_json(force=True) or {}
        return jsonify(agent_manager.request(
            match_token(), "play", "action", indices=body.get("indices")
        ))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/agent/step")
def agent_step():
    try:
        return jsonify(agent_manager.request(match_token(), "play", "step"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/watch/state")
def watch_state():
    try:
        return jsonify(agent_manager.request(match_token(), "watch", "state"))
    except Exception:
        return jsonify({"started": False, "watchMode": True, "cloudMode": True})


@app.post("/api/watch/new")
def watch_new_game():
    try:
        body = request.get_json(silent=True) or {}
        with capacity_lock:
            match, payload = agent_manager.create(
                "watch", [str(body.get("agentA") or ""), str(body.get("agentB") or "")]
            )
        return jsonify({**payload, "cloudMode": True, "matchToken": match.token})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/watch/step")
def watch_step():
    try:
        return jsonify(agent_manager.request(match_token(), "watch", "step"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/rooms")
def create_room():
    try:
        body = request.get_json(force=True) or {}
        with capacity_lock:
            room = manager.create(
                [str(body.get("deckA") or ""), str(body.get("deckB") or "")],
            )
        session.clear()
        session["room_id"] = room.room_id
        session["role"] = 0
        payload = manager.state(room.room_id, 0)
        payload["playerToken"] = room.player_tokens[0]
        return jsonify(payload)
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
        payload = manager.state(room.room_id, 1)
        payload["playerToken"] = room.player_tokens[1]
        return jsonify(payload)
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
