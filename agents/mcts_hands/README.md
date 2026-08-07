# mcts_hands agent

`feat/hands_generator` で学習した手札保持モデルを、現行のMCTS agentへ統合したagentです。
公開済みカードから相手の60枚デッキをDB検索で推定し、MCTSを始めるたびに、整合する
相手手札・サイド・山札を3組生成します。3組それぞれを10 simulation探索し、rootの訪問数を
合算して着手を決めます。

## 非公開領域の予測

1. 公開カードのmultisetと633件の現行デッキDBを照合する。
2. 公開ログから相手の既知手札と、一度でも確認したカードをserial単位で追跡する。
3. 学習済みモデルが残存カードごとの「現在も手札に保持されている」スコアを出す。
4. 実在枚数を上限に、未知手札を重み付き非復元抽出する。
5. 残ったカードをサイドと山札へ配り、60枚の物理的整合性を維持する。

手札モデルはカードを自由生成しません。DB推定した60枚から公開カードと確定手札を引いた
残存カードだけが候補になるため、存在しないカードや枚数超過は起きません。手札へ選ばれた
カードが変わると残余集合も変わるので、山札とサイドの予測にもモデルが反映されます。

## A/B比較

`train/db_baseline_main.py` は同一のMCTS重み、DB、探索回数、粒子数を使い、手札モデルの
スコアだけを0にします。したがって比較差は手札予測モデルの有無です。

```powershell
.\.venv\Scripts\python.exe tools\run_matches.py `
  --agent1 agents\mcts_hands\src\main.py `
  --agent2 agents\mcts_hands\train\db_baseline_main.py `
  --name1 hands_model --name2 db_only `
  --deck1 agents\mcts_hands\src\deck.csv `
  --deck2 agents\mcts_hands\src\deck.csv `
  --games 100 --workers 4 --save-json
```

## 検証

2026-08-07に同一デッキ、先後交互、4 workerで200試合を実行しました。両者とも
3 hidden-state粒子、各10 MCTS simulationで、手札モデルのON/OFFだけが異なります。

| 条件 | 勝利 | 勝率 |
| --- | ---: | ---: |
| 手札モデルあり | 105 | 52.5% |
| DB-only | 94 | 47.0% |
| 引き分け | 1 | 0.5% |

- 未解決・エラー0
- 決着局だけの手札モデルあり勝率: 105/199（52.8%）
- 決着局勝率の95% Wilson信頼区間: 45.8%–59.6%
- 両側二項検定: `p=0.478`
- 手札モデルあり先手: 62勝、後手: 43勝

点推定では手札モデルありが5.5ポイント上回りましたが、信頼区間は50%を跨ぎ、両側検定も
有意ではありません。この200試合では統計的な強さの改善を確認できませんでした。
先手勝率は全体で59.5%と高いため、先後を100回ずつに均等化した集計を採用しています。

再現コマンド:

```powershell
.\.venv\Scripts\python.exe tools\run_matches.py `
  --agent1 agents\mcts_hands\src\main.py `
  --agent2 agents\mcts_hands\train\db_baseline_main.py `
  --name1 hands_model --name2 db_only `
  --deck1 agents\mcts_hands\src\deck.csv `
  --deck2 agents\mcts_hands\src\deck.csv `
  --games 200 --workers 4 --quiet --save-json
```

その他の検証:

```powershell
.\.venv\Scripts\python.exe -m unittest agents\mcts_hands\train\test_opponent_hand.py
.\.venv\Scripts\python.exe tools\run_local_match.py --agent mcts_hands
.\.venv\Scripts\python.exe tools\build_submission.py --agent mcts_hands
```

提出対象は `src/` です。`src/main.py` は手札モデルを有効にします。学習済み
`hand_model.pth` は `feat/hands_generator` の最新commit `6c52135` と同一です。
MCTSのself/opponent評価重みは
`adamw_best_ep21600_cluster_03` の `model.pth` / `opponent_model.pth` を維持しています。
