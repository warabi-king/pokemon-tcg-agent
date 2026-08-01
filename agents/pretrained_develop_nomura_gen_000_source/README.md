# develop_nomura gen_000 事前学習済みモデル原本

`develop_nomura` ブランチの模倣学習済みモデルを、自己対戦学習の初期値として
使用するための専用ディレクトリです。

取得元:

- branch: `develop_nomura`
- commit: `327ba88d241bc5f59fa5d313c34a2cbc71833cf1`
- path: `pipeline/gen_000/agents/clXX/`
- GitHub: <https://github.com/warabi-king/pokemon-tcg-agent/tree/develop_nomura/pipeline/gen_000/agents>

配置:

```text
cl00/self.pth  # cluster_00 の自分手番モデル初期値
cl00/opp.pth   # cluster_00 の相手モデル初期値
...
cl15/self.pth  # cluster_15 の自分手番モデル初期値
cl15/opp.pth   # cluster_15 の相手モデル初期値
```

この原本は既存の `agents/16model_result` の外に分離してあります。この原本から
独立した学習用agent
`agents/16model_pretrained_gen_000_training/cluster_XX/src/` を作成しています。
`notebooks/parallel_selfplay_training_loop.ipynb` はその別agentだけを指定します。
既存の `agents/16model_result/cluster_XX/src/` へは反映しません。

このディレクトリ内のファイルは初期値の原本なので、学習処理から上書きしません。
