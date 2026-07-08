"""Start the local 1v1 browser server and a Cloudflare Quick Tunnel."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_URL = "http://127.0.0.1:8000"
TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def ensure_port_available() -> None:
    """Fail before startup if another local server already owns the duel port."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 8000))
    except OSError as exc:
        raise RuntimeError(
            "ポート8000はすでに使用中です。以前起動した app/server.py または "
            "tools/run_duel.py を Ctrl+C で終了してから、もう一度実行してください。"
        ) from exc
    finally:
        probe.close()


def wait_for_server(
    process: subprocess.Popen, instance_id: str, timeout: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("対戦サーバーが起動前に終了しました。")
        try:
            with urllib.request.urlopen(f"{LOCAL_URL}/api/duel/health", timeout=1) as response:
                import json

                payload = json.load(response)
                if payload.get("instanceId") == instance_id:
                    return
                raise RuntimeError(
                    "ポート8000で別の対戦サーバーが応答しています。"
                    "以前のサーバーを終了してから、もう一度実行してください。"
                )
        except (urllib.error.URLError, TimeoutError, ValueError):
            time.sleep(0.25)
    raise TimeoutError("対戦サーバーの起動確認がタイムアウトしました。")


def stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> None:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if npx is None:
        raise SystemExit("Node.js / npx が見つかりません。Node.jsをインストールしてください。")

    ensure_port_available()
    host_token = secrets.token_urlsafe(18)
    instance_id = secrets.token_urlsafe(16)
    environment = os.environ.copy()
    environment["DUEL_HOST_TOKEN"] = host_token
    environment["DUEL_INSTANCE_ID"] = instance_id
    environment["DUEL_MANAGED"] = "1"
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "app" / "server.py")],
        cwd=ROOT,
        env=environment,
    )
    tunnel: subprocess.Popen | None = None
    try:
        wait_for_server(server, instance_id)
        print("Cloudflare Quick Tunnelを起動しています…", flush=True)
        tunnel = subprocess.Popen(
            [npx, "--yes", "wrangler", "tunnel", "quick-start", LOCAL_URL],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert tunnel.stdout is not None
        announced = False
        for line in tunnel.stdout:
            print(line, end="", flush=True)
            match = TUNNEL_URL_RE.search(line)
            if match and not announced:
                public_url = match.group(0)
                print("\n=== 1対1対戦 ===", flush=True)
                print(f"ホスト用URL: {public_url}/h/{host_token}", flush=True)
                print("このURLは共有しないでください。画面に表示される招待リンクだけをPlayer 2へ共有します。", flush=True)
                print("終了するには Ctrl+C を押してください。\n", flush=True)
                announced = True
        if tunnel.wait() != 0:
            raise RuntimeError("Cloudflare Quick Tunnelが異常終了しました。")
    except KeyboardInterrupt:
        print("\n対戦サーバーを終了します。", flush=True)
    finally:
        stop(tunnel)
        stop(server)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"エラー: {exc}") from None
