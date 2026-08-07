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
      deck_candidates_by_wins.jsonl
      deck_reconstructor.py
      features.py
      hand_model.pth
      mcts.py
      model.py
      opponent_hand.py
  train/
    train.py
    train_hand_model.py
    test_opponent_hand.py
    plot_metrics.py
    checkpoints/
    logs/
```

`src/` は提出対象です。`train/` は学習用で、提出アーカイブには含めません。

## 相手手札の推定

MCTS探索の開始時、相手の非公開カード(手札・山札・サイド・裏向きばけポケ)は固定IDではなく
`rl_mcts.opponent_hand.OpponentBelief`が次の手順で具体化する(hidden-state粒子)。

1. 公開された相手カードを`deck_candidates_by_wins.jsonl`収録の候補デッキと照合し、
   最も一致する実在デッキ群を推定する(`deck_reconstructor.py`)。
2. 対戦ログのcard ID/serialから、公開サーチ後も手札に残っているカードを追跡する
   (`OpponentHandTracker`)。
3. 候補デッキから公開領域・確定手札を引いた残りに対し、学習済み手札保持モデル
   (`hand_model.pth`)で残存確率を補正して未知の手札を抽出する。
4. 残余カードをサイド・山札へ無作為配置し、カード枚数の整合性を保つ。
5. `belief_samples`個(既定3)のhidden-state粒子それぞれでMCTSを行い、root訪問数を
   合算した多数決で最終手を選ぶ(`mcts.mcts_agent`の`determinizations`引数)。

手札モデルはカードを直接生成せず、残存カードの抽出確率だけを補正する。そのため、
候補デッキにないカードや残存枚数を超えるカードは生成されない。

`RlMctsAgent`の`opponent_model_path`(自分/相手で別モデルを使うMCTS評価)とは独立した
機能で、両方を同時に指定できる。

### 手札予測モデルの学習

```bash
python agents/rl_mcts/train/train_hand_model.py \
  --max-episodes 2000 \
  --max-samples 50000 \
  --epochs 5 \
  --batch-size 128 \
  --hidden-size 64 \
  --device cpu
```

既定の入出力:

```text
入力: tools/deck_generator/daily_dataset/*/*.zip
出力: agents/rl_mcts/src/rl_mcts/hand_model.pth
```

モデルが無い場合は保持スコアを0として、残存カード枚数だけに比例する制約付き
サンプリングへフォールバックする(提出環境で決して落とさない)。

単体テスト:

```bash
python -m unittest agents/rl_mcts/train/test_opponent_hand.py
```

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
