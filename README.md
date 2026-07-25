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
│   ├── rl_mcts_sample/
│   │   ├── src/          # 提出対象
│   │   │   ├── main.py
│   │   │   ├── deck.csv
│   │   │   ├── cg/
│   │   │   └── rl_mcts/
│   │   ├── train/        # 提出に含めない学習用コード
│   │   └── README.md
│   ├── rl_mcts/
│   │   ├── src/          # rl_mcts_sampleから派生した改善用agent
│   │   ├── train/        # CSV/PNGログ付き学習コード（checkpoints/・logs/はGit対象外）
│   │   └── README.md
│   └── match_agents/     # デッキアーキタイプ単位の複数agent（例: imitation_group0）
│       └── {agent-name}/
│           ├── deck.csv  # そのagent固有のデッキ
│           ├── model.pth # そのagent固有の重み
│           └── src/      # prepare_match_agent_submission.pyが生成。Git対象外
├── tools/                # agent横断の補助ツール（一覧は「ツール一覧」節を参照）
│   └── train/            # 学習・前処理系ツール（詳細はtools/train/README.md）
├── sample_submission/    # Kaggle配布sample。参照用
├── data/                 # Kaggle DataのカードCSV
├── docs/                 # Kaggle DataのPDF資料
├── episodes/             # Kaggleの日次対戦リプレイ(.zip)。Git管理対象外
├── shards/                # preprocess_episodes.pyが出力する学習用シャード。Git管理対象外
├── dist/                 # 提出アーカイブ出力先。Git管理対象外
├── results/              # ローカルの対戦結果・学習ログ・デッキ集計結果。Git管理対象外
│   ├── logs/             # 各種スクリプトの実行ログ
│   └── deck_analysis/    # group_decks.pyが出力するdeck_groups.json等
├── requirements.txt      # pip用の依存定義（venv + pipで進める場合）
├── pyproject.toml / poetry.lock / poetry.toml  # Poetry用の依存定義（.venvをプロジェクト直下に作る場合）
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

Poetryで進める場合（GPU利用など、他プロジェクトと環境を分離したい場合はこちら）:

```bash
poetry config virtualenvs.in-project true --local
poetry env use 3.11
poetry install
poetry run python tools/run_local_match.py --agent random
```

`.venv`はプロジェクト直下に作られ、`pyproject.toml`/`poetry.lock`で依存を管理します。
`requirements.txt`と依存内容は揃えていますが、片方だけ更新した場合はもう片方にも反映してください。

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

PyTorchなど特定の学習・推論依存があるagentは、必要な依存関係を `requirements.txt` と
各agentの `README.md` に明記します。

学習済み重みを提出に使う場合は、採用する重みだけを `agents/{agent-name}/src/` に置きます。
途中checkpointやログは `agents/{agent-name}/train/checkpoints/` や `logs/` に置き、Gitには含めません。

## ツール一覧

`tools/` 配下のagent横断ツールです。詳しい使い方は各ファイル冒頭のdocstringを参照してください。

対戦・評価:

```text
tools/run_local_match.py             同じ/異なるagent同士で1試合だけ実行する
tools/run_matches.py                 2agentで複数試合実行し、勝率などを集計する
tools/run_matches_round_robin.py     3agent以上を総当たりさせ、対戦カードごとの結果と
                                      総合成績を集計する（--agent name=path[:deck]を複数指定）
tools/compare_models.py              同じagent実装のまま、model.pthを2つ比較対戦させる
                                      （改善前後の重みを比較するときに使う）
```

提出関連:

```text
tools/build_submission.py                agents/{agent-name}/src/ から提出用tar.gzを作る
tools/prepare_match_agent_submission.py  agents/match_agents/{name}/のdeck.csv・model.pthから
                                          提出可能なsrc/一式を組み立て、tar.gzも作る
```

デッキ・カードデータ関連:

```text
tools/inspect_cards.py       cabt SDK(またはdata/配下のCSV)からカードメタデータを一覧表示する
tools/deck_signature.py      デッキを主要ポケモンの集合でアーキタイプ化する共通ロジック
                              （group_decks.py/preprocess_episodes.pyが内部で使うライブラリ。
                              単体実行はしない）
tools/episode_io.py          日次エピソードデータ(展開済みディレクトリ or .zip)を横断して
                              読むための共通ヘルパー（同上、ライブラリ用途）
tools/group_decks.py         複数日分のエピソードからデッキ使用頻度を集計し、アーキタイプ単位の
                              グループ(deck_groups.json)を作る。詳細はtools/train/README.md
```

学習・並列実行関連（詳細は `tools/train/README.md` を参照）:

```text
tools/train/preprocess_episodes.py   エピソードJSONを模倣学習用シャードに変換する
tools/train/train_imitation.py       シャードを使って模倣学習する
tools/train/train.py                 rl_mctsの自己対戦(MCTS)学習を行う
tools/train/train_match_agents.py    agents/match_agents/配下の複数agentを総当たり自己対戦させながら学習する
tools/train/plot_metrics.py          train.pyの学習ログ(CSV)からPNGグラフを作る
tools/run_train_round_robin.py       複数agentのtrain.py実行をサブプロセスとして総当たりで組み合わせて呼び出す
tools/run_train_using_template.py    共通のtrain.pyテンプレートを別agentディレクトリに適用して学習する
```

GPUバッチ推論で自己対戦データ収集を高速化する場合（`develop_naoki`ブランチから選択的に取り込み）:

```text
tools/run_train_round_robin_batched.py   中央管理・GPUバッチ推論対応の総当たり学習実行スクリプト。
                                          run_train_round_robin.pyとは別物（サブプロセスを呼ばず、
                                          各agentを同一プロセス内で扱う）。
                                          --backend shared-batch/shared-cpu-batchで複数試合のNN評価を
                                          batched_training.py経由でバッチ化し、GPUへまとめて渡す。
tools/batched_tournament.py              batched系のコア実装（NN評価バッチ化、GPU device Tensor管理）
tools/batched_training.py                上記を使った自己対戦学習サンプル収集
tools/live_loss_recorder.py              学習中のバッチlossを逐次CSV/JSONへ記録する
                                          （Notebookなどからのライブ監視用）

使用例:
  python tools/run_train_round_robin_batched.py \
    --agent agents/rl_mcts \
    --iterations 5 --games 100 --search-count 10 \
    --backend shared-batch --device cuda
```

`tools/gpu_simulator.py`・`tools/gpu_tree_tournament.py`（GPU上でカード効果やMCTS木そのものを
再現する実験的エンジン）は未完成かつ上記の学習パスに必須ではないため取り込んでいません。

## 既存sample

`sample_submission/` はKaggle配布sampleの参照用です。通常は編集しません。

`src/`、`src_random/` は旧構成の名残です。今後の開発・提出・検証は
`agents/{agent-name}/src/` を使います。
