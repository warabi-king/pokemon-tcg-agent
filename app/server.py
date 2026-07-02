"""Local browser UI for playing against src/main.py."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
SRC_ROOT = ROOT / "src"
CARD_ROOT = ROOT / "docs" / "cards"


def load_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain exactly 60 card IDs (found {len(deck)}).")
    return deck


# Import the environment first. On Windows the repository SDK has no cg.dll,
# while kaggle-environments ships one. Reusing that already-initialized library
# also avoids initializing the cabt engine twice.
from kaggle_environments import make  # noqa: E402
from kaggle_environments.envs.cabt.cg.game import battle_finish  # noqa: E402
from kaggle_environments.envs.cabt.cg import sim as environment_sim  # noqa: E402
from kaggle_environments.envs.cabt.cg.sim import Battle  # noqa: E402

environment_sim.lib.AllCard.restype = ctypes.c_char_p
environment_sim.lib.AllAttack.restype = ctypes.c_char_p

sys.path.insert(0, str(SRC_ROOT))
import cg  # noqa: E402

sys.modules["cg.sim"] = environment_sim

previous_cwd = Path.cwd()
try:
    os.chdir(SRC_ROOT)
    import main as opponent_module  # noqa: E402
finally:
    os.chdir(previous_cwd)

from cg.api import all_attack  # noqa: E402


def plain(value: Any) -> Any:
    """Convert Kaggle Struct objects and enums into JSON-safe primitives."""
    return json.loads(json.dumps(value))


CARD_META = {
    card.cardId: {
        "id": card.cardId,
        "name": card.name,
        "type": int(card.cardType),
        "hp": card.hp,
        "attacks": card.attacks,
    }
    for card in opponent_module.all_card
}
ATTACK_META = {
    attack.attackId: {
        "id": attack.attackId,
        "name": attack.name,
        "damage": attack.damage,
        "text": attack.text,
    }
    for attack in all_attack()
}


class MatchController:
    """Own the single local cabt match used by the browser UI."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.env = None
        self.last_observation: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None

    def new_game(self) -> dict[str, Any]:
        with self.lock:
            human_deck = load_deck(APP_ROOT / "deck.csv")
            opponent_deck = load_deck(SRC_ROOT / "deck.csv")
            if Battle.battle_ptr:
                battle_finish()
                Battle.battle_ptr = None
                Battle.obs = None
            opponent_module.plan = opponent_module.AttackPlan()
            opponent_module.pre_turn = 0
            opponent_module.ability_used = False

            self.env = make("cabt", debug=True)
            self.env.reset()
            self.env.step([human_deck, opponent_deck])
            self.last_observation = None
            self.events = []
            self.error = None
            self._capture_observation("対戦開始")
            self._advance_opponent()
            return self.payload()

    def act(self, indices: Any) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                raise ValueError("対戦が開始されていません。")
            if self.env.done:
                raise ValueError("この対戦は終了しています。")
            if self.env.state[0].status != "ACTIVE":
                raise ValueError("現在はAIの選択待ちです。")
            if not isinstance(indices, list) or any(type(index) is not int for index in indices):
                raise ValueError("indices must be an array of integers.")

            observation = plain(self.env.state[0].observation)
            selection = observation.get("select")
            if selection is None:
                raise ValueError("合法手情報がありません。")

            option_count = len(selection["option"])
            if len(indices) < selection["minCount"] or len(indices) > selection["maxCount"]:
                raise ValueError(
                    f"{selection['minCount']}〜{selection['maxCount']}個の選択が必要です。"
                )
            if len(indices) != len(set(indices)):
                raise ValueError("同じ選択肢を複数回選ぶことはできません。")
            if any(index < 0 or index >= option_count for index in indices):
                raise ValueError("存在しない選択肢が含まれています。")

            self.env.step([indices, None])
            self._capture_observation("あなた")
            self._advance_opponent()
            return self.payload()

    def _advance_opponent(self) -> None:
        steps = 0
        while self.env is not None and not self.env.done and self.env.state[1].status == "ACTIVE":
            observation = plain(self.env.state[1].observation)
            try:
                action = opponent_module.agent(observation)
                self.env.step([None, action])
            except Exception as exc:  # Surface agent errors in the UI.
                self.error = f"AIエージェントでエラーが発生しました: {exc}"
                break
            self._capture_observation("AI")
            steps += 1
            if steps >= 200:
                self.error = "AIの連続選択が200回を超えたため停止しました。"
                break

    def _capture_observation(self, source: str) -> None:
        if self.env is None:
            return
        candidate = None
        for state in self.env.state:
            observation = plain(state.observation)
            if observation.get("current") is not None:
                candidate = observation
                if state.status == "ACTIVE":
                    break
        if candidate is None:
            return
        self.last_observation = candidate
        for log in candidate.get("logs") or []:
            self.events.append({"source": source, **log})
        self.events = self.events[-120:]

    def payload(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                return {"started": False}
            states = [
                {"status": state.status, "reward": state.reward}
                for state in self.env.state
            ]
            return {
                "started": True,
                "finished": self.env.done,
                "humanTurn": not self.env.done and self.env.state[0].status == "ACTIVE",
                "observation": self.last_observation,
                "states": states,
                "events": self.events,
                "error": self.error,
                "step": len(self.env.steps) - 1,
            }


app = Flask(__name__, static_folder=None)
match = MatchController()


@app.get("/")
def index():
    return send_from_directory(APP_ROOT, "index.html")


@app.get("/static/<path:filename>")
def static_file(filename: str):
    return send_from_directory(APP_ROOT, filename)


@app.get("/cards/<path:filename>")
def card_image(filename: str):
    return send_from_directory(CARD_ROOT, filename)


@app.get("/api/meta")
def metadata():
    return jsonify({"cards": CARD_META, "attacks": ATTACK_META})


@app.get("/api/state")
def state():
    return jsonify(match.payload())


@app.post("/api/new")
def new_game():
    try:
        return jsonify(match.new_game())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/action")
def action():
    try:
        body = request.get_json(force=True) or {}
        return jsonify(match.act(body.get("indices")))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("PokeTCG Battle Table: http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
