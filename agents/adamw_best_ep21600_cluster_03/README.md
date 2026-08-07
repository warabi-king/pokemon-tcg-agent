# adamw_best_ep21600_cluster_03

`parallel_selfplay_training_loop_stateful_adamw.ipynb` の実行結果から、
自己対戦を除く総当たり成績が最良だったパラメータの組を提出用にまとめたagentです。

## 選定結果

- 学習run: `20260803_095342_717598`
- agent/deck: `cluster_03`
- checkpoint: `model_episode21600`
- 評価に使った次バッチ: `episode25200`
- 成績: 222勝172敗4分 / 398試合
- 勝率: 55.78%
- 決着試合勝率: 56.35%

評価時と同じく、探索中の自分ターンには `model.pth`、相手ターンには
`opponent_model.pth` を使用します。詳細なパラメータと重みのSHA-256は
`src/selection.json` に記録しています。

## ローカル実行

```bash
python tools/run_local_match.py --agent adamw_best_ep21600_cluster_03
```

## 提出アーカイブ作成

```bash
python tools/build_submission.py --agent adamw_best_ep21600_cluster_03
```
