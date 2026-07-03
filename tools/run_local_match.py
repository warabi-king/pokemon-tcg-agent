"""提出用エージェントを使ってcabtのローカル対戦を実行する。"""

from __future__ import annotations

import html
import json
import ctypes
from pathlib import Path
import os
import sys

from kaggle_environments import make
from kaggle_environments.envs.cabt.cg import sim as environment_sim

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
RESULTS_ROOT = ROOT / "results"
sys.path.insert(0, str(SRC_ROOT))

# The repository SDK contains libcg.so for Kaggle/Linux but no cg.dll.
# On Windows, reuse the cabt engine bundled with kaggle-environments.
environment_sim.lib.AllCard.restype = ctypes.c_char_p
environment_sim.lib.AllAttack.restype = ctypes.c_char_p
sys.modules["cg.sim"] = environment_sim

previous_cwd = Path.cwd()
try:
    os.chdir(SRC_ROOT)
    from main import agent  # noqa: E402
finally:
    os.chdir(previous_cwd)


def load_deck(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} が存在しません。カードIDを60枚分書いたdeck.csvを先に作成してください。"
        )

    deck = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csvはカードIDを60枚分だけ含める必要があります。現在: {len(deck)}枚")
    return deck


def build_result_html(steps: list) -> str:
    steps_json = json.dumps(steps, ensure_ascii=False, default=str)
    escaped_steps_json = html.escape(steps_json)
    final_step = steps[-1] if steps else []
    summary_rows = []
    for player_index, state in enumerate(final_step):
        summary_rows.append(
            "<tr>"
            f"<td>{player_index}</td>"
            f"<td>{html.escape(str(state.get('status')))}</td>"
            f"<td>{html.escape(str(state.get('reward')))}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cabt local match result</title>
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
  <h1>cabt local match result</h1>
  <section>
    <h2>Summary</h2>
    <p>steps: <strong>{len(steps)}</strong></p>
    <table>
      <thead><tr><th>player</th><th>status</th><th>reward</th></tr></thead>
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


def main() -> None:
    deck = load_deck(SRC_ROOT / "deck.csv")
    os.chdir(SRC_ROOT)
    env = make("cabt", configuration={"decks": [deck, deck]}, debug=True)
    env.run([agent, agent])

    RESULTS_ROOT.mkdir(exist_ok=True)
    result_path = RESULTS_ROOT / "result.html"
    kaggle_result_path = RESULTS_ROOT / "result_kaggle.html"
    kaggle_result_path.write_text(env.render(mode="html"), encoding="utf-8")
    result_path.write_text(build_result_html(env.steps), encoding="utf-8")
    print(f"シミュレーションが完了しました: {result_path}")
    print(f"Kaggle標準HTMLも出力しました: {kaggle_result_path}")


if __name__ == "__main__":
    main()
