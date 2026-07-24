# pokemon-tcg match training server

This repository is trimmed to run the continuous match-training server for
Pokemon TCG match agents.

## Layout

```text
agents/
  match_agents/
    {agent}/
      deck.csv
      model.pth
      train/logs/
  rl_mcts/src/
    cg/
    rl_mcts/
    model.pth
tools/train/
  train_match_server.py
  train_match_agents.py
  train.py
requirements.txt
```

`agents/match_agents/{agent}` directories are discovered dynamically. Any
subdirectory with a 60-card `deck.csv` is treated as one trainable agent. New
agent directories can be added while the server is running.

## Run

Windows smoke test:

```powershell
.venv\Scripts\python.exe tools/train/train_match_server.py `
  --max-generations 1 `
  --search-count 1
```

macOS long-running server:

```bash
python3 tools/train/train_match_server.py \
  --agents-root agents/match_agents \
  --search-count 10 \
  --batch-size 128
```

Each generation randomly selects two distinct agents, plays two games with
first/second order swapped, trains both selected models, and saves results under
each agent directory.

## Outputs

```text
agents/match_agents/{agent}/model.pth
agents/match_agents/{agent}/train/logs/train_metrics.csv
agents/match_agents/{agent}/train/logs/match_results.csv
```

Use `Ctrl+C` to stop after the current generation.
