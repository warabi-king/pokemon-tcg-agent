# develop_nomura gen_000 事前学習モデルの継続学習

`develop_nomura` の `pipeline/gen_000/agents/cl00`〜`cl15` にある
模倣学習済みモデルから作成した16 agentの学習用ディレクトリです。

対応関係:

```text
cluster_00/src/model.pth          <- cl00/self.pth
cluster_00/src/opponent_model.pth <- cl00/opp.pth
...
cluster_15/src/model.pth          <- cl15/self.pth
cluster_15/src/opponent_model.pth <- cl15/opp.pth
```

取得元:

- branch: `develop_nomura`
- commit: `327ba88d241bc5f59fa5d313c34a2cbc71833cf1`
- path: `pipeline/gen_000/agents/clXX/`

実行入口は `notebooks/parallel_selfplay_training_loop.ipynb` です。更新後モデルは
`model_episode3600/`、`model_episode7200/` のようにこのディレクトリ内へ保存します。
notebookは各 `src/model.pth` と `src/opponent_model.pth` を初期重みとして使い、
存在しないファイルだけ同じパスへランダム初期重みを生成します。
