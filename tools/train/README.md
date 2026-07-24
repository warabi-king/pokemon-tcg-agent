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

- `preprocess_episodes.py`
  - Kaggle公式配布のエピソードJSON（リプレイログ、複数日分）を模倣学習サンプルに変換し、
    シャード（`.pkl`、既定2万件/ファイル）としてディスクへ保存します。
  - 一度に全サンプルをメモリに載せないため、数万〜数十万試合規模でも処理できます。
  - `tools/group_decks.py`が出力したグループJSONを指定すると、特定のデッキアーキタイプの
    対戦データだけを抽出できます。

- `train_imitation.py`
  - `preprocess_episodes.py`が作ったシャードを使った模倣学習スクリプトです。
  - シャード単位でストリーミング学習するため、常時メモリ上に持つのは1シャード分だけです。
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
必要な日数分ダウンロードしておきます。`--episodes`には展開済みディレクトリと`.zip`のどちらも、
複数日分まとめても渡せます（`.zip`ならディスク展開せずそのまま読みます）。

流れは「前処理（シャード作成）→ 学習」の2段階です。前処理を独立させているのは、
JSON解析とエンコードのコストを毎エポック払わずに済ませるためです。

### 1. 前処理（シャード作成）

```powershell
.venv\Scripts\python.exe tools/train/preprocess_episodes.py `
  --episodes path\to\day1.zip path\to\day2.zip path\to\day3.zip `
  --output-dir shards\all `
  --shard-size 20000
```

まず少数だけで動作確認したい場合:

```powershell
.venv\Scripts\python.exe tools/train/preprocess_episodes.py `
  --episodes path\to\day1.zip `
  --max-episodes 50 `
  --output-dir shards\test
```

主な引数:

```text
--episodes        エピソードJSONのディレクトリ or .zip（複数指定可、必須）
--output-dir      シャードの保存先（必須）
--shard-size      1シャードあたりのサンプル数上限（デフォルト20000）
--max-episodes    全ソース合計で読み込むエピソード数の上限（省略時は全件）
--max-samples     収集する学習サンプル数の上限（省略時は無制限）
--deck-groups     tools/group_decks.pyが出力したJSON（特定デッキだけ抽出する場合）
--target-group    --deck-groups内のgroup_id
```

主な出力:

```text
shards/all/shard_00000.pkl, shard_00001.pkl, ...
shards/all/manifest.json
```

### 2. 学習

```powershell
.venv\Scripts\python.exe tools/train/train_imitation.py `
  --shards shards\all `
  --epochs 3 `
  --batch-size 128
```

主な引数:

```text
--shards          preprocess_episodes.pyの--output-dir（必須）
--val-shards      検証用に取り分けるシャード数（デフォルト1）
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

### デッキアーキタイプ単位で学習する

`tools/group_decks.py`（リポジトリ直下、詳細は後述）で作った`deck_groups.json`を使うと、
特定のデッキアーキタイプの対戦データだけで前処理・学習できます。

```powershell
.venv\Scripts\python.exe tools/train/preprocess_episodes.py `
  --episodes path\to\day1.zip path\to\day2.zip `
  --deck-groups deck_groups.json --target-group 0 `
  --output-dir shards\group0

.venv\Scripts\python.exe tools/train/train_imitation.py `
  --shards shards\group0 `
  --epochs 3 `
  --output-model agents\rl_mcts\train\checkpoints\imitation_group0.pth
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
