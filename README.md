# pokemon-tcg match training server

このリポジトリは、Pokemon TCG の match agent 同士を常時対戦・学習させる
サーバを動かすための最小構成です。

## 構成

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

`agents/match_agents/{agent}` の各フォルダをエージェントとして扱います。
60枚のカードIDを持つ `deck.csv` があるサブディレクトリは、自動的に学習対象
として検出されます。サーバ実行中に新しいエージェントフォルダを追加しても、
次の世代から再スキャンされます。

## 実行

Windowsでの簡易確認:

```powershell
.venv\Scripts\python.exe tools/train/train_match_server.py `
  --max-generations 1 `
  --search-count 1
```

macOSで常時実行する場合:

```bash
python3 tools/train/train_match_server.py \
  --agents-root agents/match_agents \
  --search-count 10 \
  --batch-size 128
```

1世代では、ランダムに異なる2つのエージェントを選び、先攻後攻を入れ替えて
2試合行います。その後、両エージェントをその世代の対戦サンプルで学習し、
各エージェントフォルダへ結果を保存します。

## 出力

```text
agents/match_agents/{agent}/model.pth
agents/match_agents/{agent}/train/logs/train_metrics.csv
agents/match_agents/{agent}/train/logs/match_results.csv
```

停止するときは `Ctrl+C` を押します。現在の世代が終わったところで停止します。
