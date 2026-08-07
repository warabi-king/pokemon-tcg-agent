"""提出用エージェントを使ってcabtのローカル対戦を実行する。"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
from pathlib import Path
import os
import sys
from types import ModuleType

from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = ROOT / "agents"
RESULTS_ROOT = ROOT / "results"


def agent_src_dir(agent_name: str) -> Path:
    """agent名からsrcディレクトリを返す。"""
    return AGENTS_ROOT / agent_name / "src"


def load_module(path: Path, module_name: str, src_root: Path) -> ModuleType:
    """src配下のmain.pyをユニークなモジュール名で読み込む。"""
    sys.path.insert(0, str(src_root))
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} を読み込めませんでした。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_deck(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が存在しません。カードIDを60枚分書いたdeck.csvを先に作成してください。"
        )

    deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csvはカードIDを60枚分だけ含める必要があります。現在: {len(deck)}枚")
    return deck


def load_agent(agent_name: str, module_name: str):
    """agent名からagent関数とデッキを読み込む。"""
    src_root = agent_src_dir(agent_name)
    main_path = src_root / "main.py"
    deck_path = src_root / "deck.csv"
    if not main_path.exists():
        raise FileNotFoundError(f"{main_path} が存在しません。")

    deck = load_deck(deck_path)
    module = load_module(main_path, module_name, src_root)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{main_path} に agent(obs_dict) が定義されていません。")
    if hasattr(module, "read_deck_csv"):
        module.read_deck_csv = lambda: list(deck)  # type: ignore[assignment]
    return module.agent, deck, src_root


def build_result_html(steps: list, agent_names: list[str]) -> str:
    steps_json = json.dumps(steps, ensure_ascii=False, default=str)
    escaped_steps_json = html.escape(steps_json)
    final_step = steps[-1] if steps else []
    summary_rows = []
    for player_index, state in enumerate(final_step):
        agent_name = agent_names[player_index] if player_index < len(agent_names) else str(player_index)
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(agent_name)}</td>"
            f"<td>{html.escape(str(state.get('status')))}</td>"
            f"<td>{html.escape(str(state.get('reward')))}</td>"
            "</tr>"
        )
    match_title = " vs ".join(agent_names)

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(match_title)} - cabt local match result</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #1f2933;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    section {{
      background: white;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid #e5eaf0;
      padding: 8px;
      text-align: left;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #111827;
      color: #e5e7eb;
      padding: 16px;
      border-radius: 6px;
      max-height: 70vh;
      overflow: auto;
    }}
    input {{
      width: 120px;
      padding: 6px 8px;
    }}
    button {{
      padding: 6px 10px;
      margin-left: 4px;
    }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(match_title)}</h1>
  <section>
    <h2>Summary</h2>
    <p>steps: <strong>{len(steps)}</strong></p>
    <table>
      <thead><tr><th>agent</th><th>status</th><th>reward</th></tr></thead>
      <tbody>{''.join(summary_rows)}</tbody>
    </table>
  </section>
  <section>
    <h2>Step Viewer</h2>
    <p>
      <input id="stepInput" type="number" min="0" max="{max(len(steps) - 1, 0)}" value="{max(len(steps) - 1, 0)}">
      <button id="prevButton">Prev</button>
      <button id="nextButton">Next</button>
      <button id="lastButton">Last</button>
    </p>
    <pre id="stepOutput"></pre>
  </section>
  <section>
    <h2>Raw Steps JSON</h2>
    <pre>{escaped_steps_json}</pre>
  </section>
</main>
<script>
const steps = {steps_json};
const input = document.getElementById("stepInput");
const output = document.getElementById("stepOutput");

function renderStep(index) {{
  const normalized = Math.max(0, Math.min(steps.length - 1, Number(index || 0)));
  input.value = normalized;
  output.textContent = JSON.stringify(steps[normalized], null, 2);
}}

document.getElementById("prevButton").addEventListener("click", () => renderStep(Number(input.value) - 1));
document.getElementById("nextButton").addEventListener("click", () => renderStep(Number(input.value) + 1));
document.getElementById("lastButton").addEventListener("click", () => renderStep(steps.length - 1));
input.addEventListener("change", () => renderStep(input.value));
renderStep(input.value);
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="random", help="同一agent同士で対戦するagent名")
    parser.add_argument("--agent-a", default=None, help="player0側のagent名")
    parser.add_argument("--agent-b", default=None, help="player1側のagent名")
    parser.add_argument("--debug", action="store_true", help="kaggle_environmentsのdebugログを出す")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_a_name = args.agent_a or args.agent
    agent_b_name = args.agent_b or args.agent

    agent_a, deck_a, src_a = load_agent(agent_a_name, "local_agent_a")
    agent_b, deck_b, src_b = load_agent(agent_b_name, "local_agent_b")

    # deck.csvなどの相対パス参照に対応するため、player0側のsrcで実行する。
    os.chdir(src_a)
    env = make("cabt", configuration={"decks": [deck_a, deck_b]}, debug=args.debug)
    env.run([agent_a, agent_b])

    RESULTS_ROOT.mkdir(exist_ok=True)
    match_name = f"{agent_a_name}_vs_{agent_b_name}"
    match_root = RESULTS_ROOT / match_name
    match_root.mkdir(exist_ok=True)
    result_path = match_root / "result.html"
    kaggle_result_path = match_root / "result_kaggle.html"
    kaggle_result_path.write_text(env.render(mode="html"), encoding="utf-8")
    result_path.write_text(
        build_result_html(env.steps, [agent_a_name, agent_b_name]),
        encoding="utf-8",
    )
    print(f"シミュレーションが完了しました: {result_path}")
    print(f"Kaggle標準HTMLも出力しました: {kaggle_result_path}")
    print(f"player0: {agent_a_name} ({src_a})")
    print(f"player1: {agent_b_name} ({src_b})")


if __name__ == "__main__":
    main()
