# AGENTS.md

このファイルは、このリポジトリで作業する開発者およびAIエージェント向けの作業ルールです。

## 基本方針

- agentごとに `agents/{agent-name}/` を作る。
- 提出・対戦に必要な実装は `agents/{agent-name}/src/` 配下に置く。
- 学習用コードは `agents/{agent-name}/train/` 配下に置く。
- Kaggle Dataから取得した `sample_submission/` は参照用として扱い、通常は編集しない。
- Kaggle提出物は `agents/{agent-name}/src/main.py`、`deck.csv`、`cg/` から作成する。
- ローカル検証用の生成物はGitに含めない。

## ディレクトリの役割

```text
agents/{agent-name}/src/
  開発・提出対象。
  main.py、deck.csv、cg/、推論時に必要なagent固有コードを含む。

agents/{agent-name}/train/
  agent固有の学習用コード。
  提出アーカイブには含めない。

tools/
  agent横断のローカル開発用ツール。
  Kaggle提出物には含めない。

sample_submission/
  Kaggle配布sampleの保管場所。
  参照用として基本的に生のまま残す。

data/
  Kaggle DataのカードCSV。

docs/
  Kaggle DataのPDFや関連資料。

results/
  ローカル対戦結果の出力先。
  Git追跡対象外。

dist/
  提出アーカイブの出力先。
  Git追跡対象外。
```

## agent名

- agent名は `random`、`rl_mcts_sample` のようにアンダースコア区切りにする。
- ハイフンはPython importやツール引数で扱いにくいため使わない。
- `agents/{agent-name}/README.md` に、そのagentの仕組み、学習方法、実行方法を書く。

## 編集対象

通常編集してよいもの:

- `agents/{agent-name}/src/main.py`
- `agents/{agent-name}/src/deck.csv`
- `agents/{agent-name}/src/` 配下のagent固有コード
- `agents/{agent-name}/train/*.py`
- `agents/{agent-name}/README.md`
- `tools/*.py`
- `README.md`
- `AGENTS.md`

慎重に扱うもの:

- `agents/{agent-name}/src/cg/`
  - cabt SDK本体。
  - SDK更新や公式sampleとの差分反映が目的でない限り編集しない。

通常編集しないもの:

- `sample_submission/`
  - 公式sampleの参照用コピー。
  - 変更が必要な場合は、先に理由を明確にする。

## 実行・検証

セットアップ:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

ローカル対戦:

```bash
python tools/run_local_match.py --agent random
python tools/run_local_match.py --agent-a rl_mcts_sample --agent-b random
```

複数試合の比較:

```bash
python tools/run_matches.py --agent-a rl_mcts_sample --agent-b random --games 20
```

提出ファイル作成:

```bash
python tools/build_submission.py --agent rl_mcts_sample
```

提出アーカイブの中身確認:

```bash
tar -tzf dist/submission_rl_mcts_sample.tar.gz | head -30
```

期待される配置:

```text
main.py
deck.csv
cg/
```

`agents/` や `src/` ディレクトリごとアーカイブに入れてはいけない。

## ブランチ運用

このリポジトリはGitHub Flowで運用する。

- `main` を常に安定版かつ作業用ブランチの起点にする。
- 新しい機能追加は `main` から `feat/*` ブランチを作成して進める。
- バグ修正は `main` から `fix/*` ブランチを作成して進める。
- 作業用ブランチは小さく保ち、Pull Requestでレビュー・確認してから `main` に取り込む。
- 長期運用の `develop` ブランチは通常使わない。

## Git管理

Gitに含める:

- `agents/`
- `tools/`
- `sample_submission/`
- `data/`
- `docs/`
- `requirements.txt`
- `README.md`
- `AGENTS.md`

Gitに含めない:

- `.venv/`
- `results/`
- `dist/`
- `submission.tar.gz`
- `__pycache__/`
- `*.pyc`
- `.DS_Store`
- `agents/*/train/checkpoints/`
- `agents/*/train/logs/`

## セキュリティ

- APIキー、トークン、パスワード、Kaggle認証情報をコミットしない。
- `.env` や秘密鍵ファイルを作成・表示・共有しない。
- Kaggle評価中は外部通信できないため、エージェントは提出アーカイブ内のファイルだけで動く前提にする。

## 実装方針

- まずは対象agentの `src/main.py` の `agent(obs_dict)` を改善する。
- cabtは合法手だけを `obs.select.option` に渡すため、合法手生成ではなく合法手の選択ロジックに集中する。
- 変更後は最低限 `python tools/run_local_match.py --agent {agent-name}` を実行し、試合が最後まで進むことを確認する。
- 提出前は `python tools/build_submission.py --agent {agent-name}` を実行し、アーカイブ直下に `main.py`、`deck.csv`、`cg/` が入ることを確認する。
