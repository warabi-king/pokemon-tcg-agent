# tools/train

## 常時対戦学習サーバ

`train_match_server.py` がメインの起動スクリプトです。サーバは起動し続け、
各世代の前に `agents/match_agents` を再スキャンします。その中からランダムに
異なる2つのエージェントを選び、先攻後攻を入れ替えて2試合を行い、選ばれた
両方のモデルを学習します。

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

## 補助モジュール

`train_match_agents.py` と `train.py` は、サーバが既存の対戦収集、サンプルの
ラベル付け、checkpoint保存、学習処理を再利用するために残しています。

モデル定義と cabt 実行環境は `agents/rl_mcts/src` から読み込みます。
