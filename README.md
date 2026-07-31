# pokemon-tcg-agent

Pokémon TCG AI Battle Challenge Simulation向けのAIエージェント開発用リポジトリです。

このリポジトリでは、複数agentを並行して開発できるように、agentごとのディレクトリに
提出用コードと学習用コードをまとめます。

## ブランチ運用

このリポジトリはGitHub Flowで運用します。`main` を常に安定版かつ作業用ブランチの
起点にし、機能追加は `feat/*`、バグ修正は `fix/*` の短命ブランチで進めます。
作業が完了したらPull Requestで確認し、`main` に取り込みます。

開発速度を優先するため、長期運用の `develop` ブランチは通常使いません。

## ディレクトリ構成

```text
.
├── agents/
│   ├── random/
│   │   ├── src/          # 提出対象
│   │   │   ├── main.py
│   │   │   ├── deck.csv
│   │   │   └── cg/
│   │   └── README.md
│   └── rl_mcts_sample/
│       ├── src/          # 提出対象
│       │   ├── main.py
│       │   ├── deck.csv
│       │   ├── cg/
│       │   └── rl_mcts/
│       ├── train/        # 提出に含めない学習用コード
│       └── README.md
│   └── rl_mcts/
│       ├── src/          # rl_mcts_sampleから派生した改善用agent
│       ├── train/        # CSV/PNGログ付き学習コード
│       └── README.md
├── tools/                # agent横断の補助ツール
├── sample_submission/    # Kaggle配布sample。参照用
├── data/                 # Kaggle DataのカードCSV
├── docs/                 # Kaggle DataのPDF資料
├── dist/                 # 提出アーカイブ出力先。Git管理対象外
├── results/              # ローカル対戦結果。Git管理対象外
├── requirements.txt
└── AGENTS.md
```

## agent構成

各agentは次の形にします。

```text
agents/{agent-name}/
  src/
    main.py
    deck.csv
    cg/
    ...                 # 推論時に必要なagent固有コード
  train/
    ...                 # 学習時だけ使うコード
  README.md             # 仕組み、学習方法、使い方
```

`src/` はKaggle提出物の元です。`train/` は提出に含めません。

agent名はPythonやファイルパスで扱いやすいように、ハイフンではなくアンダースコアを使います。

例:

```text
random
rl_mcts_sample
rule_based
```

## セットアップ

Python 3.11で動作確認しています。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` では `kaggle-environments==1.30.2` を使っています。

## ローカル対戦

同じagent同士で1試合実行します。

```bash
python tools/run_local_match.py --agent random
python tools/run_local_match.py --agent rl_mcts_sample
```

異なるagent同士で1試合実行します。

```bash
python tools/run_local_match.py --agent-a rl_mcts_sample --agent-b random
```

複数試合の統計を取る場合:

```bash
python tools/run_matches.py --agent-a rl_mcts_sample --agent-b random --games 20
```

複数試合はデフォルトでCPU数に応じて最大4ワーカーへ分配されます。ワーカー数を
明示する場合は `--workers`、従来どおり直列で実行する場合は `--workers 1` を使います。

```bash
python tools/run_matches.py --agent-a rl_mcts_sample --agent-b random --games 100 --workers 4
```

総当たり戦も同じ指定で、対戦カードをまたいで全試合を並列実行できます。

```bash
python tools/run_matches_round_robin.py --games 20 --workers 4
```

libcg・特徴量生成を複数CPUプロセスで並列化し、NN評価だけを中央のGPUへ集約する場合は
`worker-batched` backendを使います。各CPU workerは独立したlibcgを所有するため、
盤面遷移同士は直列化されません。worker間の要求を中央で再結合し、小batch時は
8モデルをmodel軸へ積んだ1つのGPU演算、大batch時はモデル別batchへ自動切替します。
model-axisの疎埋め込みはモデルごとの語彙indexをずらして1回の標準EmbeddingBagへ
統合します。RTX 3050実測に基づき、平均model batchが32未満ならmodel-axis、
それ以上ならper-modelへ切り替えます。
CPU workerはtorchをimportせず、疎特徴量を`int32`/`float32`のNumPy連続配列へ
まとめてQueueへ渡します。中央GPU batcherは`torch.from_numpy()`でCPUバッファを
共有してからCUDAへ転送するため、中央でのPython list→Tensor変換は行いません。
worker jobは中央batchへ隙間なく詰められるよう`--batch-size / --workers`を上限とし、
中央側で同じモデルのjobを`--batch-size`まで結合します。
中央per-model batchはdecoder幅の小さいjobから順に詰め、幅64の候補手が少数混じる
ことで幅1/2/4の全rowまで64へpaddingされる無駄を抑えます。
libcgの`SearchBegin`/`SearchStep`が返すJSONは`msgspec`で必要なfieldだけを軽量Structへ
直接decodeし、SDK既定の再帰的dataclass変換を行いません。Battle観測のJSON decode、
固定deck特徴、option数と選択数が同じ合法手組合せもworker内で再利用します。
MCTS nodeはObservation全体を保持せず、次の展開に必要なSearch ID・手番・勝敗だけを
保持するため、高lane時のworker RSSとobject解放負荷を抑えます。
複数選択の候補手では、同じoptionのdecoder特徴を組合せごとに再計算せず、局面内で
一度だけ生成した小配列を各候補手へ再利用します。

```bash
python tools/run_matches_round_robin.py \
  --backend worker-batched --device cuda --workers 0 \
  --lanes 0 --batch-size 256 \
  --games 1 \
  --agent a00=agents/rl_mcts_match_00/src/main.py \
  --agent a01=agents/rl_mcts_match_01/src/main.py
```

`worker-batched --workers 0`ではlaneごとに独立workerを作ります。したがって
8エージェント・1ラウンドなら36 workerです。中央batcherは最初の要求を受け取ると、
その時点でIPC queueへ届いている全要求だけを待たずに回収し、即GPU評価します。
回収件数用の閾値は設けず、GPU推論を分割する上限には`--batch-size`だけを使います。
`worker-batched`の既定値は実測で最速だった256です。同じ監視方式による
19 workers・3800 lanes・4000試合のA/Bでは、128の5.416 games/sに対して
256は6.994 games/s（+29.1%）でした。
`--lanes 0`は全試合を同時laneへ載せます。
8エージェントでは自己対戦込みで1ラウンド36組なので、`--games k`は合計`36*k`試合・
`36*k` laneになります。

CUDA上で複数モデルのNN評価を試合横断でまとめる場合は、`cuda-ensemble` backendを
使います。モデル当たりの要求が小さいwaveはmodel軸へstackし、大きいwaveは
per-model batchへ自動で切り替わります。カード効果と盤面遷移は共有libcgで処理します。

```bash
python tools/run_matches_round_robin.py \
  --backend cuda-ensemble --device cuda \
  --lanes 128 --batch-size 128 --games 4 --no-self \
  --agent a00=agents/rl_mcts_match_00/src/main.py \
  --agent a01=agents/rl_mcts_match_01/src/main.py
```

出力先:

```text
results/{agent-a}_vs_{agent-b}/result.html
results/{agent-a}_vs_{agent-b}/result_kaggle.html
```

## 提出ファイル作成

agent名を指定して提出アーカイブを作成します。

```bash
python tools/build_submission.py --agent random
python tools/build_submission.py --agent rl_mcts_sample
```

出力先:

```text
dist/submission_{agent-name}.tar.gz
```

アーカイブ直下には `agents/{agent-name}/src/` の中身が入ります。

```text
main.py
deck.csv
cg/
...
```

`agents/` や `src/` ディレクトリ自体はアーカイブに入れません。

## 学習コード

学習コードは各agent配下に置きます。

```text
agents/{agent-name}/train/
```

提出時に必要な推論コード、特徴量変換、モデル定義などは `src/` に置きます。
自己対戦ループ、optimizer、評価ログ、checkpoint保存など、学習時だけ使う処理は `train/` に置きます。

例:

```text
agents/rl_mcts_sample/src/rl_mcts/model.py
agents/rl_mcts_sample/train/train.py
```

学習実行例:

```bash
python agents/rl_mcts_sample/train/train.py
python agents/rl_mcts/train/train.py --plot
```

3つのラウンドロビンagentを、1つのlibcgと試合横断CPUバッチ推論で中央学習する場合:

```bash
.venv/bin/python tools/run_train_round_robin.py \
  --agent agents/rl_mcts_r_robin1 \
  --agent agents/rl_mcts_r_robin2 \
  --agent agents/rl_mcts_r_robin3 \
  --backend shared-cpu-batch \
  --device cpu --inference-device cpu \
  --iterations 5 --games 100 \
  --lanes 128 --inference-batch-size 128 \
  --batch-size 128 --search-count 10
```

各iterationでは自己対戦を含む6組を同時進行し、同一モデルのNN評価要求を
CPUの1回のforwardへまとめます。チェックポイントとメトリクスは
`results/train_round_robin_central/central_*/` に保存され、更新後の重みは各agentの
`src/model.pth` に反映されます。動作確認だけ行い、重みを反映しない場合は
`--no-persist` を追加してください。

`agents/match_agents/00`〜`07` のデッキと初期重みから8体の学習agentを
初回生成する場合:

```bash
.venv/bin/python tools/create_match_training_agents.py
```

生成される `rl_mcts_match_00`〜`07` を、全agentのLossが0.03以下になるまで
共有libcg＋CPUバッチで学習する例:

```bash
.venv/bin/python tools/run_train_round_robin.py \
  --agent agents/rl_mcts_match_00 \
  --agent agents/rl_mcts_match_01 \
  --agent agents/rl_mcts_match_02 \
  --agent agents/rl_mcts_match_03 \
  --agent agents/rl_mcts_match_04 \
  --agent agents/rl_mcts_match_05 \
  --agent agents/rl_mcts_match_06 \
  --agent agents/rl_mcts_match_07 \
  --backend shared-cpu-batch \
  --device cpu --inference-device cpu \
  --iterations 12 --games 5 --search-count 10 \
  --lanes 128 --inference-batch-size 128 --batch-size 128 \
  --target-loss 0.03 --min-iterations 3 --loss-patience 2
```

全8体が閾値以下になった状態を指定回数連続で確認すると早期終了します。
Loss履歴はrun directory直下の `loss_history.csv` と `loss.png` に保存されます。

バッチごとのLossをリアルタイム表示しながら実行する場合は、
`notebooks/train_match_agents_live.ipynb` をJupyterで開きます。設定セルを実行した後、
最後のセルを実行すると学習プロセスが開始され、同じセル内のMatplotlibグラフ、
各agentのLoss表、実行ログが1秒ごとに更新されます。

```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/train_match_agents_live.ipynb
```

学習側は各バッチ後に `live_loss_batches.csv`、収集・学習中の状態を
`live_status.json` へ保存します。Notebookを閉じてもこれらの生データはrun directoryに残ります。

PyTorchなど特定の学習・推論依存があるagentは、必要な依存関係を `requirements.txt` と
各agentの `README.md` に明記します。

学習済み重みを提出に使う場合は、採用する重みだけを `agents/{agent-name}/src/` に置きます。
途中checkpointやログは `agents/{agent-name}/train/checkpoints/` や `logs/` に置き、Gitには含めません。

## 既存sample

`sample_submission/` はKaggle配布sampleの参照用です。通常は編集しません。

`src/`、`src_random/` は旧構成の名残です。今後の開発・提出・検証は
`agents/{agent-name}/src/` を使います。
