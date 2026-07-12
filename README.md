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
│   ├── rule_Lucario/
│   │   └── src/          # ルカリオデッキ用ルールベースagent
│   │       ├── main.py
│   │       ├── deck.csv
│   │       └── cg/
│   ├── rl_mcts_sample/
│   │   ├── src/          # MCTSサンプルの提出対象
│   │   │   ├── main.py
│   │   │   ├── deck.csv
│   │   │   ├── cg/
│   │   │   └── rl_mcts/
│   │   ├── train/        # 提出に含めない学習用コード
│   │   └── README.md
│   └── rl_mcts/
│       ├── src/          # rl_mcts_sampleから派生した改善用agent
│       ├── train/        # CSV/PNGログ付き学習コード
│       └── README.md
├── app/                  # ブラウザ対戦・観戦UI
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
rule_Lucario
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
python tools/run_local_match.py --agent rule_Lucario
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

## ブラウザ対戦・観戦

```powershell
.\.venv\Scripts\python.exe app\server.py
```

`http://127.0.0.1:8000` では人間対エージェント、`/watch` ではエージェント同士の
対戦を表示します。画面上部の選択欄には `agents/{agent-name}/src/main.py` と
`deck.csv` が揃っているエージェントが自動で表示されます。

## ローカルデッキビルダー

高機能デッキビルダーは、GCP反映用のブラウザ対戦UIとは分離したローカル専用ツールです。
`data/JP_Card_Data.csv` をカード検索データとして読み込み、カード画像は
`docs/card-assets/cards/` のWebP画像を使います。

```powershell
.\.venv\Scripts\python.exe tools\local_deck_builder.py
```

起動後、次のURLを開きます。

```text
http://127.0.0.1:8010
```

主な機能:

- カード名、ID、ワザ名、効果説明などの日本語テキスト検索
- 「ポケモンの進化の段階/エネルギー・トレーナーズの種類」「ルール」「カテゴリ」「タイプ」での絞り込み
- `deck.csv` の読み込みと保存
- 選択中デッキの一覧コピー
- ボタン操作によるデッキ制約チェック

制約チェックでは、60枚であること、基本エネルギー以外の同名カードが4枚以内であること、
ACE SPECが1枚以内であること、カードデータに存在しないIDがないことを確認します。
チェック結果がNGでも、検証用デッキを扱えるように `deck.csv` は保存できます。

## 提出ファイル作成

agent名を指定して提出アーカイブを作成します。

```bash
python tools/build_submission.py --agent random
python tools/build_submission.py --agent rule_Lucario
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

## 既存sample

`sample_submission/` はKaggle配布sampleの参照用です。通常は編集しません。

旧ルート直下の `src/`、`src_sec/` は廃止しました。開発・提出・検証では
必ず `agents/{agent-name}/src/` を使います。
