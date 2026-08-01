# 16 model result

`cluster_00`〜`cluster_15`の16デッキ専用agentと学習パラメータをまとめたディレクトリです。

各clusterの主なファイル:

```text
cluster_XX/src/deck.csv
cluster_XX/src/model.pth
cluster_XX/src/opponent_model.pth  # 最初の学習完了後に作成
```

3600エピソードごとの更新後パラメータは、累計エピソード数別に保存します。

```text
model_episode3600/cluster_XX/model.pth
model_episode3600/cluster_XX/opponent_model.pth
model_episode7200/cluster_XX/model.pth
model_episode7200/cluster_XX/opponent_model.pth
```

各checkpointを保存してから、同じパラメータを`cluster_XX/src/`へ反映し、
次の対戦に使用します。

並列対戦と学習ループは
`notebooks/parallel_selfplay_training_loop.ipynb`から実行します。
