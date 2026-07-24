# tools/train

## continuous match training server

`train_match_server.py` is the main entry point. It keeps running, re-scans
`agents/match_agents` before every generation, randomly chooses two distinct
agents, plays two games with first/second order swapped, trains both selected
models, and saves per-agent metrics.

```powershell
.venv\Scripts\python.exe tools/train/train_match_server.py `
  --max-generations 1 `
  --search-count 1
```

```bash
python3 tools/train/train_match_server.py \
  --agents-root agents/match_agents \
  --search-count 10 \
  --batch-size 128
```

## support modules

`train_match_agents.py` and `train.py` remain because the server reuses their
game collection, sample labeling, checkpoint saving, and training functions.

The model and cabt runtime are loaded from `agents/rl_mcts/src`.
