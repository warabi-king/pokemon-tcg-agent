"""Local browser UI for playing against src/main.py."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "app"
AGENTS_ROOT = ROOT / "agents"
CARD_ROOT = ROOT / "docs" / "cards"


def available_agents() -> list[str]:
    """Return agent directories that contain a runnable src/main.py and deck."""
    return sorted(
        path.name
        for path in AGENTS_ROOT.iterdir()
        if (path / "src" / "main.py").is_file() and (path / "src" / "deck.csv").is_file()
    )


def agent_src(agent_name: str) -> Path:
    if agent_name not in available_agents():
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENTS_ROOT / agent_name / "src"


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

DEFAULT_AGENT = available_agents()[0]
DEFAULT_SRC_ROOT = agent_src(DEFAULT_AGENT)
sys.path.insert(0, str(DEFAULT_SRC_ROOT))
import cg  # noqa: E402

sys.modules["cg.sim"] = environment_sim

def load_agent_module(name: str, root: Path):
    """Load an agent under a unique module name so its globals stay isolated."""
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(name, root / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load agent module from {root / 'main.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = Path.cwd()
    try:
        os.chdir(root)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


opponent_module = load_agent_module("browser_default_agent", DEFAULT_SRC_ROOT)

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


# cabt stores one process-wide native battle pointer. Both browser modes must
# therefore share the same lock and ownership record. A normally completed
# battle has already been freed by cabt even though Battle.battle_ptr still
# contains its old address; calling battle_finish() again causes an access
# violation on Windows.
BATTLE_LOCK = threading.RLock()
active_environment = None


def discard_active_battle() -> None:
    """Release an unfinished native battle and clear stale Python pointers."""
    global active_environment
    if Battle.battle_ptr and active_environment is not None and not active_environment.done:
        battle_finish()
    Battle.battle_ptr = None
    Battle.obs = None
    active_environment = None


def register_active_battle(env: Any) -> None:
    global active_environment
    active_environment = env


def require_active_battle(env: Any) -> None:
    if env is not active_environment:
        raise ValueError("別の対戦が開始されています。新しい対戦を開始してください。")


class MatchController:
    """Own the single local cabt match used by the browser UI."""

    def __init__(self) -> None:
        self.lock = BATTLE_LOCK
        self.env = None
        self.last_observation: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.error: str | None = None

    def new_game(self, agent_name: str = DEFAULT_AGENT) -> dict[str, Any]:
        with self.lock:
            global opponent_module
            selected_src = agent_src(agent_name)
            opponent_module = load_agent_module("browser_human_opponent", selected_src)
            human_deck = load_deck(APP_ROOT / "deck.csv")
            opponent_deck = load_deck(selected_src / "deck.csv")
            discard_active_battle()
            reset_agent(opponent_module)

            self.env = make("cabt", debug=True)
            self.env.reset()
            register_active_battle(self.env)
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
            require_active_battle(self.env)
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


def reset_agent(module: Any) -> None:
    """Reset the mutable state used by the bundled sample agent."""
    if hasattr(module, "AttackPlan"):
        module.plan = module.AttackPlan()
    if hasattr(module, "pre_turn"):
        module.pre_turn = 0
    if hasattr(module, "ability_used"):
        module.ability_used = False


class AgentMatchController:
    """Run two selected agents, advancing exactly one agent decision at a time."""

    def __init__(self) -> None:
        self.lock = BATTLE_LOCK
        self.env = None
        self.last_observation: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.error: str | None = None
        self.modules: list[Any] = []
        self.agent_names: list[str] = []

    def new_game(self, agent_a: str = DEFAULT_AGENT, agent_b: str = DEFAULT_AGENT) -> dict[str, Any]:
        with self.lock:
            names = [agent_a, agent_b]
            roots = [agent_src(name) for name in names]
            decks = [load_deck(root / "deck.csv") for root in roots]
            self.modules = [
                load_agent_module(f"browser_watch_agent_{index}", root)
                for index, root in enumerate(roots)
            ]
            self.agent_names = names
            discard_active_battle()

            for module in self.modules:
                reset_agent(module)
            self.env = make("cabt", debug=True)
            self.env.reset()
            register_active_battle(self.env)
            self.env.step(decks)
            self.last_observation = None
            self.events = []
            self.actions = []
            self.error = None
            self._capture_observation("SETUP")
            return self.payload()

    def step(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                raise ValueError("観戦対戦が開始されていません。")
            require_active_battle(self.env)
            if self.env.done:
                return self.payload()

            active = [index for index, state in enumerate(self.env.state) if state.status == "ACTIVE"]
            if len(active) != 1:
                raise RuntimeError(f"行動するプレイヤーを特定できません: {active}")
            player_index = active[0]
            module = self.modules[player_index]
            observation = plain(self.env.state[player_index].observation)
            try:
                action = module.agent(observation)
                env_actions = [None, None]
                env_actions[player_index] = action
                self.env.step(env_actions)
            except Exception as exc:
                self.error = f"Player {player_index + 1} のエージェントでエラーが発生しました: {exc}"
                return self.payload()

            self.actions.append(
                {
                    "number": len(self.actions) + 1,
                    "player": player_index,
                    "indices": action,
                }
            )
            self._capture_observation(f"P{player_index + 1}")
            return self.payload()

    def _capture_observation(self, source: str) -> None:
        if self.env is None:
            return
        observations = [plain(state.observation) for state in self.env.state]
        base = next((obs for obs in observations if obs.get("current") is not None), None)
        if base is None:
            return

        # Each player observation exposes that player's own hand. Merge those
        # two views so a spectator can see both agents' cards.
        for player_index, observation in enumerate(observations):
            current = observation.get("current")
            if current is None:
                continue
            own_state = current["players"][player_index]
            base["current"]["players"][player_index]["hand"] = own_state.get("hand", [])
            base["current"]["players"][player_index]["handCount"] = own_state.get("handCount", 0)

        self.last_observation = base
        logs = base.get("logs") or []
        for log in logs:
            self.events.append({"source": source, **log})
        self.events = self.events[-160:]

    def payload(self) -> dict[str, Any]:
        with self.lock:
            if self.env is None:
                return {"started": False, "watchMode": True}
            active = next(
                (index for index, state in enumerate(self.env.state) if state.status == "ACTIVE"),
                None,
            )
            return {
                "started": True,
                "watchMode": True,
                "finished": self.env.done,
                "humanTurn": False,
                "nextPlayer": active,
                "observation": self.last_observation,
                "states": [
                    {"status": state.status, "reward": state.reward}
                    for state in self.env.state
                ],
                "events": self.events,
                "actions": self.actions,
                "error": self.error,
                "step": len(self.actions),
                "agentNames": self.agent_names,
            }


app = Flask(__name__, static_folder=None)
match = MatchController()
agent_match = AgentMatchController()


@app.get("/")
def index():
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


@app.get("/api/meta")
def metadata():
    return jsonify({"cards": CARD_META, "attacks": ATTACK_META, "agents": available_agents()})


@app.get("/api/state")
def state():
    return jsonify(match.payload())


@app.post("/api/new")
def new_game():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(match.new_game(body.get("agent", DEFAULT_AGENT)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/action")
def action():
    try:
        body = request.get_json(force=True) or {}
        return jsonify(match.act(body.get("indices")))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/watch/state")
def watch_state():
    return jsonify(agent_match.payload())


@app.post("/api/watch/new")
def watch_new_game():
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(
            agent_match.new_game(
                body.get("agentA", DEFAULT_AGENT),
                body.get("agentB", DEFAULT_AGENT),
            )
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/watch/step")
def watch_step():
    try:
        return jsonify(agent_match.step())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("PokeTCG Battle Table: http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
