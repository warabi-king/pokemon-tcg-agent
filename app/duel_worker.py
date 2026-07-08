"""One isolated cabt battle process used by the Cloud Run room manager."""

from __future__ import annotations

import ctypes
import json
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def load_deck(deck_source: str | list[int]) -> list[int]:
    if isinstance(deck_source, list):
        if len(deck_source) != 60 or any(type(card_id) is not int or card_id <= 0 for card_id in deck_source):
            raise ValueError("CSV deck must contain exactly 60 card IDs.")
        return deck_source
    agent_name = deck_source
    path = ROOT / "agents" / agent_name / "src" / "deck.csv"
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{agent_name}のdeck.csvは60枚である必要があります。")
    return deck


def configure_environment():
    from kaggle_environments.envs.cabt.cg import sim

    sim.lib.AllCard.restype = ctypes.c_char_p
    sim.lib.AllAttack.restype = ctypes.c_char_p
    sim.lib.AgentStart.restype = ctypes.c_void_p
    sim.lib.SearchBegin.restype = ctypes.c_char_p
    sim.lib.SearchBegin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    sim.lib.SearchStep.restype = ctypes.c_char_p
    sim.lib.SearchStep.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    sim.lib.SearchEnd.argtypes = [ctypes.c_void_p]
    sim.lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    sys.modules["cg.sim"] = sim
    return sim


class BattleWorker:
    def __init__(self, deck_names: list[str | list[int]]) -> None:
        from kaggle_environments import make

        sim = configure_environment()
        from kaggle_environments.envs.cabt.cg.sim import Battle

        self._battle_class = Battle
        self._battle_finish = __import__(
            "kaggle_environments.envs.cabt.cg.game", fromlist=["battle_finish"]
        ).battle_finish
        self._sim = sim
        decks = [load_deck(name) for name in deck_names]
        self.env = make("cabt", debug=False)
        self.env.reset()
        self.env.step(decks)
        self.last_observations: list[dict[str, Any] | None] = [None, None]
        self.events: list[list[dict[str, Any]]] = [[], []]
        self._capture("SETUP")

    def _capture(self, source: str) -> None:
        for role, state in enumerate(self.env.state):
            observation = plain(state.observation)
            if observation.get("current") is None:
                continue
            self.last_observations[role] = observation
            for log in observation.get("logs") or []:
                self.events[role].append({"source": source, **log})
            self.events[role] = self.events[role][-120:]

    def state(self, role: int) -> dict[str, Any]:
        if role not in (0, 1):
            raise ValueError("プレイヤー番号が不正です。")
        observation = self.last_observations[role]
        if observation is not None:
            observation = plain(observation)
            if role == 1 and observation.get("current") is not None:
                current = observation["current"]
                players = current.get("players")
                if isinstance(players, list) and len(players) == 2:
                    current["players"] = [players[1], players[0]]
                    current["yourIndex"] = 0
        states = [
            {"status": state.status, "reward": state.reward}
            for state in self.env.state
        ]
        if role == 1:
            states.reverse()
        return {
            "finished": self.env.done,
            "active": not self.env.done and self.env.state[role].status == "ACTIVE",
            "observation": observation,
            "states": states,
            "events": self.events[role],
            "step": len(self.env.steps) - 1,
        }

    def action(self, role: int, indices: Any, expected_step: Any) -> dict[str, Any]:
        if role not in (0, 1):
            raise ValueError("プレイヤー番号が不正です。")
        if self.env.done:
            raise ValueError("この対戦は終了しています。")
        if self.env.state[role].status != "ACTIVE":
            raise ValueError("現在は相手のターンです。")
        current_step = len(self.env.steps) - 1
        if expected_step != current_step:
            raise ValueError("対戦状態が更新されています。画面を更新してください。")
        if not isinstance(indices, list) or any(type(index) is not int for index in indices):
            raise ValueError("indices must be an array of integers.")

        observation = plain(self.env.state[role].observation)
        selection = observation.get("select")
        if selection is None:
            raise ValueError("選択可能な合法手がありません。")
        option_count = len(selection["option"])
        if not selection["minCount"] <= len(indices) <= selection["maxCount"]:
            raise ValueError(
                f"{selection['minCount']}〜{selection['maxCount']}個の選択が必要です。"
            )
        if len(indices) != len(set(indices)):
            raise ValueError("同じ選択肢を複数回選ぶことはできません。")
        if any(index < 0 or index >= option_count for index in indices):
            raise ValueError("存在しない選択肢が含まれています。")

        actions = [None, None]
        actions[role] = indices
        self.env.step(actions)
        self._capture(f"P{role + 1}")
        return self.state(role)

    def close(self) -> None:
        if self._battle_class.battle_ptr and not self.env.done:
            self._battle_finish()
        self._battle_class.battle_ptr = None
        self._battle_class.obs = None


def worker_main(connection: Connection, deck_names: list[str | list[int]]) -> None:
    worker: BattleWorker | None = None
    try:
        worker = BattleWorker(deck_names)
        connection.send({"ok": True, "ready": True})
        while True:
            command = connection.recv()
            operation = command.get("operation")
            if operation == "shutdown":
                connection.send({"ok": True})
                break
            try:
                if operation == "state":
                    result = worker.state(command["role"])
                elif operation == "action":
                    result = worker.action(
                        command["role"], command.get("indices"), command.get("step")
                    )
                else:
                    raise ValueError("不明な対戦コマンドです。")
                connection.send({"ok": True, "result": result})
            except Exception as exc:
                connection.send({"ok": False, "error": str(exc)})
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send({"ok": False, "error": str(exc)})
        except Exception:
            pass
    finally:
        if worker is not None:
            worker.close()
        connection.close()
