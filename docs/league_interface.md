# ① リーグ（並列版）インターフェース契約

自己対戦パイプライン（`tools/pipeline/`）の世代ループは、リーグ戦部分を **黒箱** として呼び出す。
この黒箱（友人が並列化して実装）が満たすべき入出力仕様を定める。

## 呼び出し方法

パイプラインは環境変数 `PIPE_LEAGUE_CMD` に設定されたコマンドを実行する。
コマンド文字列中の以下のプレースホルダが置換される。

- `{manifest}` … 入力マニフェスト JSON のパス
- `{out}` … 出力先ディレクトリ

例:

```bash
export PIPE_LEAGUE_CMD='python /path/to/league_parallel.py --agents {manifest} --out {out}'
```

`PIPE_LEAGUE_CMD` が未設定のときは、動作確認用スタブ `tools/pipeline/league_stub.py`
（逐次総当たり）が使われる。本番では必ず設定すること。

## 入力: マニフェスト JSON（`{manifest}`）

対戦させるエージェントの配列。各要素:

```json
[
  {
    "name": "cl00",
    "src":  "pipeline/gen_000/agents/cl00/src",
    "deck": "pipeline/gen_000/agents/cl00/deck.csv"
  },
  ...
]
```

- `src` は梱包済みディレクトリ。直下に `main.py`（`agent(obs_dict)` を持つ）、`cg/`、`rl_mcts/`、
  `model.pth`(self)、`opponent_model.pth`(opp)、`deck.csv` を含む。
  Kaggle 提出物と同じ構造で、そのまま `agent` を import して実行できる。
- 各エージェントは自分の `deck.csv` を使用する（`main.py` は `obs.select is None` のとき
  `read_deck_csv()` を返す）。

## 出力: episode JSON（`{out}/` 配下）

各対戦を 1 ファイルの kaggle episode JSON として `{out}` に書き出す（ファイル名は任意、拡張子 `.json`）。
パイプラインの前処理 `extract_samples_from_episode` が要求する形式:

- トップレベルに `steps`（配列）と `rewards`（長さ2、各プレイヤーの最終報酬）を持つ。
- `steps[1][player]["action"]` に、そのプレイヤーの 60 枚デッキ（カードIDの配列）が入る。
- 以降の各ステップ `steps[i][player]` は `observation`（`select` を含む）と、
  次ステップ `steps[i+1][player]["action"]`（その選択への回答）を持つ
  （kaggle_environments の一般規約: action[i] は observation[i-1] への回答）。

`kaggle_environments` の `env.run(...)` 後に `env.toJSON()` で得られる形式がそのまま該当する
（スタブはこれを利用している）。

## 対戦組み合わせ・試合数・並列度

総当たりの取り方、各カードの試合数、先手/後手の入替、並列実行方法は **リーグ側の責務**。
パイプラインは `{out}` に貯まった episode をすべて読み、各エージェントについて
「自分のデッキで打った手(role=own)」「自分と対戦した相手が打った手(role=opponent)」を
`deck.csv` の完全一致で抽出して学習する。したがって:

- リーグ内では各エージェントに割り当てられた `deck.csv` を厳密に使うこと（改変しない）。
- 十分な試合数を回すほど、各エージェントの self/opp 学習データが増える。
