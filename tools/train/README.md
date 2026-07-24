# tools/train

`rl_mcts` 系モデルの学習と学習ログ可視化を行うスクリプト群です。

## ファイル

- `train.py`
  - `agents/rl_mcts/src` のTransformer + MCTS実装を使う単体学習スクリプトです。
  - 通常は1モデルの自己対戦学習を行います。
  - `--dual` を付けると2モデルを同じプロセス内で対戦させ、それぞれを更新します。

- `plot_metrics.py`
  - `train.py` が出力した `train_metrics.csv` からPNGグラフを作ります。
  - loss、評価勝率、サンプル数、batch数、経過時間を確認できます。

- `train_match_agents.py`
  - `agents/match_agents/{agent}/deck.csv` を読み、複数agentを総当たりで対戦させながら学習します。
  - 各agentは同じモデル構造を使いますが、`agents/match_agents/{agent}/model.pth` に個別パラメータを持ちます。
  - デフォルトでは `deck.csv` を持つサブディレクトリだけをagentとして扱います。

- `plot_match_metrics.py`
  - `train_match_agents.py` の複数agent用ログをPNGグラフ化します。
  - agent別の勝率、loss、サンプル数、batch数、相手別勝率ヒートマップを出力します。

- `train_imitation.py`
  - Kaggle公式配布のエピソードJSON（リプレイログ）を使った模倣学習スクリプトです。
  - MCTS探索は行わず、実際に選ばれた手を正解クラスとした交差エントロピーでpolicyを、
    そのエピソードの実際の勝敗をラベルにしたHuberLossでvalueを学習します。

## 単体学習

```powershell
.venv\Scripts\python.exe tools/train/train.py `
  --iterations 5 `
  --eval-games 10 `
  --self-play-games 30 `
  --search-count 5 `
  --batch-size 64 `
  --plot
```

主な出力:

```text
agents/rl_mcts/train/checkpoints/
agents/rl_mcts/train/logs/train_metrics.csv
agents/rl_mcts/train/logs/*.png
agents/rl_mcts/src/model.pth
```

## 複数agent学習

事前に以下のように、各フォルダへ `deck.csv` を置きます。

```text
agents/match_agents/
  00/deck.csv
  01/deck.csv
  ...
  07/deck.csv
```

実行例:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py `
  --iterations 5 `
  --games-per-pair 4 `
  --search-count 10 `
  --batch-size 128 `
  --plot
```

デフォルトでは、各agentの初期重みは以下の優先順で読みます。

```text
1. agents/match_agents/{agent}/model.pth
2. agents/rl_mcts/src/model.pth
3. ランダム初期化
```

最初から全agentを共通初期モデルで始めたい場合:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py --fresh
```

完全にランダム初期化したい場合:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py --fresh --no-initial-model
```

特定agentだけ学習したい場合:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py --agent 00 --agent 01
```

自己対戦も含めたい場合:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py --include-self
```

主な出力:

```text
agents/match_agents/{agent}/model.pth
agents/match_agents/train/checkpoints/{run-name}/{agent}/model_{iteration}.pth
agents/match_agents/train/logs/train_metrics.csv
agents/match_agents/train/logs/pair_metrics.csv
agents/match_agents/train/logs/match_*.png
```

`--run-name` を省略すると、checkpointは実行時刻のディレクトリへ保存されます。
過去の `model_2.pth` などを直接上書きしないため、Windowsのファイルロックによる保存失敗を避けやすくなります。

## 模倣学習（公式リプレイ）

事前にKaggleの日次エピソードデータセット（例: `pokemon-tcg-ai-battle-episodes-2026-07-23`）を
ダウンロードしておきます。`--episodes`には展開済みディレクトリと`.zip`のどちらも渡せます
（`.zip`ならディスク展開せずそのまま読みます）。

```powershell
.venv\Scripts\python.exe tools/train/train_imitation.py `
  --episodes path\to\pokemon-tcg-ai-battle-episodes-2026-07-23.zip `
  --epochs 5 `
  --batch-size 128
```

まず少数だけで動作確認したい場合:

```powershell
.venv\Scripts\python.exe tools/train/train_imitation.py `
  --episodes path\to\episodes.zip `
  --max-episodes 5 `
  --epochs 2 `
  --batch-size 32
```

主な引数:

```text
--episodes        エピソードJSONのディレクトリ or .zip（必須）
--max-episodes    読み込むエピソード数の上限（省略時は全件）
--max-samples     収集する学習サンプル数の上限（省略時は無制限）
--val-ratio       検証用に取り分ける割合（デフォルト0.05）
--epochs          学習エポック数（デフォルト5）
--batch-size      バッチサイズ（デフォルト128）
--lr              AdamWのlearning rate（デフォルト3e-4）
--initial-model   続きから学習する場合の初期重み
--output-model    保存先（デフォルト agents/rl_mcts/train/checkpoints/imitation_model.pth）
--metrics-file    CSVログ保存先（デフォルト agents/rl_mcts/train/logs/imitation_metrics.csv）
```

`--output-model`のデフォルトは提出用の`agents/rl_mcts/src/model.pth`を誤って
上書きしないよう、checkpoints配下にしています。結果を提出物として使う場合は
`--output-model agents/rl_mcts/src/model.pth`を明示的に指定してください。

主な出力:

```text
agents/rl_mcts/train/checkpoints/imitation_model.pth
agents/rl_mcts/train/logs/imitation_metrics.csv
```

`imitation_metrics.csv`の主な列:

```text
epoch, batches, loss, loss_value, loss_policy, train_accuracy, val_accuracy, elapsed_seconds
```

## グラフだけ再生成

単体学習ログ:

```powershell
.venv\Scripts\python.exe tools/train/plot_metrics.py
```

複数agent学習ログ:

```powershell
.venv\Scripts\python.exe tools/train/plot_match_metrics.py
```

## 軽い確認

実際に対戦・学習せず、読み込み対象と組み合わせだけ確認する場合:

```powershell
.venv\Scripts\python.exe tools/train/train_match_agents.py --dry-run
```

## match_agentsを提出形式にする

`agents/match_agents/00` などで学習した `deck.csv` と `model.pth` を使い、提出可能な `src/` と `tar.gz` を作る場合は、リポジトリ直下から以下を実行します。

```powershell
.venv\Scripts\python.exe tools/prepare_match_agent_submission.py --agent 00
```

このコマンドは以下を行います。

```text
1. agents/rl_mcts/src から main.py, cg/, rl_mcts/ をコピー
2. agents/match_agents/00/deck.csv と model.pth を src/ へコピー
3. tools/build_submission.py を呼び出して tar.gz を作成
```

出力例:

```text
agents/match_agents/00/src/
dist/submission_match_agent_00.tar.gz
```

全agentをまとめて作る場合:

```powershell
.venv\Scripts\python.exe tools/prepare_match_agent_submission.py --all
```

`src/` の作成だけ確認したい場合:

```powershell
.venv\Scripts\python.exe tools/prepare_match_agent_submission.py --agent 00 --no-build
```
