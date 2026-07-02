# rl_mcts_sample agent

## 概要

`rl_mcts_sample` は、ニューラルネットワークで局面と候補手を評価し、その評価を
MCTSの探索に使うagentです。

モデル実装にはPyTorchを使います。提出用 `src/main.py` は `src/model.pth` を必須とし、
モデルが存在しない、または読み込めない場合はエラーにします。ランダム選択へのfallbackはしません。

## ディレクトリ

```text
agents/rl_mcts_sample/
  src/
    main.py
    deck.csv
    cg/
    rl_mcts/
      features.py
      mcts.py
      model.py
      agent.py
  train/
    train.py
```

`src/` は提出対象です。`train/` は提出に含めない学習用コードです。

## 学習方針

学習は次の流れで実装します。

```text
固定デッキで自己対戦
  ↓
各局面でMCTSを実行
  ↓
MCTSの探索結果と最終勝敗から教師ラベルを作成
  ↓
ニューラルネットワークを更新
  ↓
採用する重みを src/model.pth として保存
```

PyTorch依存:

```bash
python -c "import torch; print(torch.__version__)"
```

## 学習実行

デフォルト設定で学習します。

```bash
python agents/rl_mcts_sample/train/train.py
```

短時間で動作確認だけする場合:

```bash
python agents/rl_mcts_sample/train/train.py \
  --iterations 1 \
  --eval-games 0 \
  --self-play-games 1 \
  --search-count 1 \
  --batch-size 16
```

主な出力:

```text
agents/rl_mcts_sample/train/checkpoints/model_{iteration}.pth
agents/rl_mcts_sample/src/model.pth
```

`src/model.pth` は提出アーカイブに含まれます。途中checkpointは提出に含めません。

## ローカル対戦

```bash
python tools/run_local_match.py --agent rl_mcts_sample
```

比較する場合:

```bash
python tools/run_local_match.py --agent-a rl_mcts_sample --agent-b random
```

## 提出ファイル作成

```bash
python tools/build_submission.py --agent rl_mcts_sample
```
