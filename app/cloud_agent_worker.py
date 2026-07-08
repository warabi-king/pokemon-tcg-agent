"""Isolated workers for human-vs-agent and agent-vs-agent Cloud Run matches."""

from __future__ import annotations

from multiprocessing.connection import Connection


def worker_main(
    connection: Connection, mode: str, agent_names: list[str], human_deck: list[int] | None = None
) -> None:
    controller = None
    try:
        # Importing server initializes cabt's process-global native bindings.
        # This happens inside the child process, so every hosted match remains isolated.
        from app.server import AgentMatchController, MatchController

        if mode == "play":
            controller = MatchController(auto_advance=False)
            controller.new_game(agent_names[0], human_deck)
        elif mode == "watch":
            controller = AgentMatchController()
            controller.new_game(agent_names[0], agent_names[1])
        else:
            raise ValueError("Unknown cloud match mode.")

        connection.send({"ok": True, "ready": True})
        while True:
            command = connection.recv()
            operation = command.get("operation")
            if operation == "shutdown":
                connection.send({"ok": True})
                break
            try:
                if operation == "state":
                    result = controller.payload()
                elif operation == "action" and mode == "play":
                    result = controller.act(command.get("indices"))
                elif operation == "step" and mode == "play":
                    result = controller.advance_opponent_step()
                elif operation == "step" and mode == "watch":
                    result = controller.step()
                else:
                    raise ValueError("Invalid cloud match operation.")
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
        if controller is not None and getattr(controller, "env", None) is not None:
            from app.server import discard_active_battle

            discard_active_battle()
        connection.close()
