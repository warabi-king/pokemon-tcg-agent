# rl_mcts agent

## 概要

`rl_mcts` は、固定デッキのプレイングを自己対戦で学習するagentです。
ニューラルネットワークで局面価値と候補手の価値を予測し、その予測をMCTSの探索に使って手を選びます。

このagentは `rl_mcts_sample` を元にしていますが、学習ログをCSVに保存し、損失や勝率をPNGグラフで確認できるようにしています。

## ディレクトリ

```text
agents/rl_mcts/
  src/
    main.py
    deck.csv
    model.pth
    cg/
    rl_mcts/
      agent.py
      deck.py
      features.py
      mcts.py
      model.py
  train/
    train.py
    plot_metrics.py
    checkpoints/
    logs/
```

`src/` は提出対象です。`train/` は学習用で、提出アーカイブには含めません。

## 相手手札の推定

MCTS開始時の非公開カードは、固定カードIDではなく次の手順で具体化します。

1. 公開された相手カードを `deck_generator_dbrecon` 由来の633候補と照合する。
2. ログのcard ID/serialから、公開サーチ後も手札に残っているカードを追跡する。
3. 候補60枚から公開領域と確定手札を引き、学習済み保持スコアで未知手札を抽出する。
4. 残余カードをサイドと山札へ無作為配置し、カード枚数の整合性を保つ。
5. 3つのhidden-state粒子でMCTSを行い、root訪問数を集約する。

手札モデルはカードを直接生成せず、残存カードの抽出確率だけを補正します。そのため、
候補デッキにないカードや残存枚数を超えるカードは生成されません。

学習データは `daily_dataset` の各プレイヤー自身の手札を教師にし、相関の強い重複を
減らすため、各プレイヤー・各ターンの最初と最後のMAIN局面だけを使います。指定した
episode上限は日付ZIPへ均等配分し、特定の日のデッキ分布に偏らないようにします。

```powershell
.\.venv\Scripts\python.exe agents\rl_mcts\train\train_hand_model.py `
  --max-episodes 2000 `
  --max-samples 30000 `
  --epochs 5
```

出力される `src/rl_mcts/hand_model.pth` は提出アーカイブに含まれます。モデルがない場合は
保持スコアを0として、残存カード枚数だけに比例する制約付きサンプリングへフォールバックします。

## 学習の流れ

1 iteration で以下を実行します。

```text
checkpoint保存
  ↓
評価用の試合
  ↓
セルフプレイの試合
  ↓
セルフプレイから学習サンプルを作成
  ↓
PyTorchでモデル更新
  ↓
CSVにメトリクス追記
```

評価用の試合は、現在モデルの強さを見るための試合です。学習には使いません。

セルフプレイの試合は、学習データを作るための試合です。各局面のvalue/policy教師ラベルを作り、ニューラルネットワーク更新に使います。

## 学習引数

### `--iterations`

学習サイクルの回数です。

デフォルト:

```text
5
```

1 iteration ごとに、評価、セルフプレイ、モデル更新、ログ出力を行います。
増やすと学習を何周も繰り返します。

### `--eval-games`

各 iteration で行う評価用の試合数です。

デフォルト:

```text
50
```

現在モデルが random agent 相手にどれくらい勝てるかを測ります。
学習データには使いません。増やすと勝率ログは安定しますが、直接モデルが強くなるわけではありません。

評価を省略する場合:

```bash
--eval-games 0
```

### `--self-play-games`

各 iteration で行うセルフプレイの試合数です。

デフォルト:

```text
100
```

`rl_mcts` 同士で対戦し、学習サンプルを作ります。
増やすと学習データが増えるため、精度改善に直接効きやすいです。ただし実行時間も増えます。

### `--search-count`

1手を選ぶときのMCTS探索回数です。

デフォルト:

```text
10
```

増やすと候補手をよりよく探索できます。セルフプレイで作る教師データの質も上がりやすいです。
一方で、1手ごとの計算量が増えるため、学習時間はかなり長くなります。

### `--batch-size`

1回の `optimizer.step()` に使う学習サンプル数です。

デフォルト:

```text
128
```

例えば `samples=1024` で `--batch-size 128` の場合、8 batch 学習します。

サンプル数が batch size 未満の場合は、その iteration の学習更新をスキップします。

### `--lr`

AdamWのlearning rateです。

デフォルト:

```text
3e-4
```

大きくすると学習が速く進む可能性がありますが、不安定になりやすいです。
小さくすると安定しやすい一方で、学習が遅くなります。

### `--lambda-value`

終局結果を過去の局面へ逆向きに反映するときの係数です。

デフォルト:

```text
0.9
```

大きいほど最終勝敗の影響を強く残します。
小さいほど各局面のMCTS評価をより強く反映します。

### `--checkpoint-dir`

iterationごとのcheckpoint保存先です。

デフォルト:

```text
agents/rl_mcts/train/checkpoints/
```

以下のようなファイルが保存されます。

```text
model_0.pth
model_1.pth
...
```

これは学習途中のモデルです。提出用の最終モデルとは別です。

### `--output-model`

学習終了後に保存する提出用モデルです。

デフォルト:

```text
agents/rl_mcts/src/model.pth
```

`tools/build_submission.py --agent rl_mcts` を実行すると、この `model.pth` が提出アーカイブに含まれます。

### `--log-dir`

CSVログとPNGグラフの保存先です。

デフォルト:

```text
agents/rl_mcts/train/logs/
```

### `--metrics-file`

CSVログファイルを明示指定します。

デフォルト:

```text
{log-dir}/train_metrics.csv
```

通常は指定不要です。実験ごとにログファイルを分けたい場合に使います。

### `--plot`

学習終了後にPNGグラフを生成します。

指定しない場合はCSVログのみ出力します。

## 学習ログ

学習時に以下を出力します。

```text
agents/rl_mcts/train/logs/train_metrics.csv
agents/rl_mcts/train/logs/loss.png
agents/rl_mcts/train/logs/eval_win_rate.png
agents/rl_mcts/train/logs/samples_batches.png
agents/rl_mcts/train/logs/elapsed_seconds.png
```

`train_metrics.csv` の主な列:

```text
iteration
eval_games
eval_win
eval_lose
eval_draw
eval_win_rate
self_play_games
samples
batches
loss
loss_value
loss_policy
elapsed_seconds
checkpoint_path
model_path
```

グラフの意味:

```text
loss.png
  total loss, value loss, policy loss の推移

eval_win_rate.png
  random agent 相手の評価勝率

samples_batches.png
  セルフプレイから得られたサンプル数と学習batch数

elapsed_seconds.png
  iterationごとの所要時間
```

## 学習実行

短時間確認:

```bash
python agents/rl_mcts/train/train.py \
  --iterations 1 \
  --eval-games 0 \
  --self-play-games 1 \
  --search-count 1 \
  --batch-size 16 \
  --plot
```

軽めの学習:

```bash
python agents/rl_mcts/train/train.py \
  --iterations 5 \
  --eval-games 10 \
  --self-play-games 30 \
  --search-count 5 \
  --batch-size 64 \
  --plot
```

本格学習例:

```bash
python agents/rl_mcts/train/train.py \
  --iterations 20 \
  --eval-games 50 \
  --self-play-games 300 \
  --search-count 20 \
  --batch-size 128 \
  --plot
```

グラフだけ再生成:

```bash
python agents/rl_mcts/train/plot_metrics.py
```

## ローカル対戦

```bash
python tools/run_local_match.py --agent-a rl_mcts --agent-b random
```

複数試合で比較:

```bash
python tools/run_matches.py --agent-a rl_mcts --agent-b random --games 20
```

## 提出ファイル作成

```bash
python tools/build_submission.py --agent rl_mcts
```

提出アーカイブ:

```text
dist/submission_rl_mcts.tar.gz
```

## 調整の目安

強くしたい場合は、まず以下の順で調整します。

1. `--self-play-games`
   学習データ量を増やします。

2. `--iterations`
   学習サイクル数を増やします。

3. `--search-count`
   MCTSの探索回数を増やして教師データの質を上げます。

4. `--batch-size`
   サンプル数とメモリに合わせて調整します。

5. `--lr`
   lossが不安定な場合や下がらない場合に調整します。
