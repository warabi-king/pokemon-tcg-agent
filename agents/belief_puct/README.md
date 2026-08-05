# belief_puct

## 採用構成

`belief_puct` は、公開情報と観測履歴に矛盾しない非公開状態を復元し、その状態でTransformer方策・価値モデルを用いたPUCT探索を行うagentです。最終設定は次のとおりです。

- アーキテクチャ名: `C′: history-consistent single-determinization PUCT`
- 非公開状態: 観測履歴と60枚デッキ候補DBから1状態を決定的にサンプル
- 探索: 1決定化 × 24探索
- 方策・価値: 約50.1 MBのTransformer checkpoint
- デッキ・checkpoint: 模倣学習完了checkpointを固定し、固定リーグで選ぶ60枚デッキ
- 暗黙fallback: `model.pth`欠落時は例外。ランダム行動へは縮退しない

過去のcluster 05 checkpointは比較基準として `train/candidates/cluster05_pretrained_model.pth` に保持します。現在の `src/model.pth` は模倣学習完了後の重みであり、以後のデッキ探索ではこのcheckpointを固定して比較します。

## ディレクトリ

```text
agents/belief_puct/
  src/                       Kaggle提出対象
    main.py                  agent(obs_dict) entrypoint
    deck.csv                 60枚のcluster 05デッキ
    model.pth                採用checkpoint
    cg/                      cabt SDK native libraryを含む
    rl_mcts/                 belief、特徴量、PUCT、Transformer
  train/
    prepare_imitation.py     日次ZIPを展開せず固定seed抽出
    train_imitation.py       時系列validation・full checkpoint再開
    train.py                 自己対戦RL学習
    checkpoints/             再開可能checkpoint
    logs/                    設定とCSV metrics
    shards/                  前処理済み学習sample
```

## 環境構築

プロジェクトルートでPython 3.11環境を作り、次を実行します。

```bash
python -m pip install -r requirements.txt
```

この実験で使用した主要versionはPython 3.11.15、PyTorch 2.12.1、`kaggle-environments` 1.30.2です。Apple M5上の実行環境ではCUDAとMPSを利用できなかったため、学習・評価ともCPUを使用しました。

## 模倣データの再作成

学習・validationは日付で分離します。`<official-archives-...>` はローカルにある公式日次ZIPを列挙してください。全ZIPを展開せず、各日20 episodeだけを固定seedでstreaming抽出します。

```bash
python agents/belief_puct/train/prepare_imitation.py \
  --archives <official-archives-2026-07-01-through-17> \
  --episodes-per-archive 20 \
  --seed 20260803 \
  --output-dir agents/belief_puct/train/shards/train_20260701_17

python agents/belief_puct/train/prepare_imitation.py \
  --archives <official-archives-2026-07-18-through-20> \
  --episodes-per-archive 20 \
  --seed 20260804 \
  --output-dir agents/belief_puct/train/shards/validation_20260718_20
```

2026-07-21〜23は最終test期間としてモデル選択に使用しませんでした。

## 模倣学習と再開

提出用重みを誤って上書きしないよう、出力先は`train/checkpoints/`にします。

```bash
python agents/belief_puct/train/train_imitation.py \
  --shards agents/belief_puct/train/shards/train_20260701_17 \
  --validation-shards agents/belief_puct/train/shards/validation_20260718_20 \
  --initial-model agents/belief_puct/train/checkpoints/baseline_model.pth \
  --epochs 1 --batch-size 128 --lr 1e-4 --seed 20260803 --device cpu \
  --output-model agents/belief_puct/train/checkpoints/imitation_candidate.pth
```

`--resume`はモデル、optimizer、Python RNG、PyTorch RNG、epochを復元します。

```bash
python agents/belief_puct/train/train_imitation.py \
  --shards agents/belief_puct/train/shards/train_20260701_17 \
  --validation-shards agents/belief_puct/train/shards/validation_20260718_20 \
  --resume agents/belief_puct/train/checkpoints/imitation_epoch_000.pth \
  --epochs 1 --batch-size 128 --lr 1e-4 --seed 20260803 --device cpu \
  --output-model agents/belief_puct/train/checkpoints/imitation_candidate_epoch1.pth
```

今回の模倣学習はvalidation top-1精度を37.96%から53.98%まで改善しました。`imitation_candidate.pth` はepoch 99完了後の軽量なモデル重みであり、現在は `src/model.pth` に反映されています。過去の固定対戦4勝16敗は、旧デッキ・旧評価条件の結果であり、この重みを固定した新しいデッキ探索とは区別します。

## 固定リーグ評価

評価器はagentを別processへ隔離し、同一seedを先後入替の2試合に使用します。`<baseline-src>`、`<baseline-deck>`、`<baseline-model>`は評価対象を固定したパスへ置き換えます。

```bash
python tools/evaluate_fixed_league.py \
  --agent-a agents/belief_puct/src \
  --agent-b <baseline-src> \
  --deck-b <baseline-deck> \
  --model-b <baseline-model> \
  --games 100 --seed-start 80000 \
  --action-timeout-seconds 30 --game-timeout-seconds 120 \
  --output results/reproduced_fixed_league_100.json
```

複数相手のJSONは次のように集約します。

```bash
python tools/summarize_league.py \
  --inputs <evaluation-json-1> <evaluation-json-2> \
  --label reproduced-league \
  --output results/reproduced_league_aggregate.json
```

## 提出物の作成と検証

```bash
python tools/build_submission.py --agent belief_puct

python tools/validate_submission.py \
  --archive dist/submission_belief_puct.tar.gz \
  --games 6 --seed-start 99000 \
  --output results/submission_validation.json
```

検証器はアーカイブ直下の`main.py`、`deck.csv`、`cg/`、60枚デッキ、不要な学習物・cacheの不在、path traversal、`/kaggle_simulations/agent/`を模した展開先からのimport、複数seed自己対戦を確認します。

## 段階的なデッキ・モデル学習

`tools/run_belief_pipeline.py`は、模倣事前学習、固定リーグでのデッキ探索、混合相手RL、候補DB拡張、holdout検証を同じrunとして記録する入口です。最初は既存の60枚DBからクラスタ多様性を保って候補を選びます。公式由来DBを上書きせず、runごとの選抜DBとログを`train/runs/<run-id>/`へ保存します。

```bash
python tools/run_belief_pipeline.py \
  --database agents/belief_puct/src/rl_mcts/deck_candidates_by_wins.jsonl \
  --run-id trial-001 --candidate-count 16 --min-candidate-games 20 --dry-run
```

## Phase 2a: 固定リーグでのデッキ選抜

候補間で変更するのは `deck.csv` だけです。候補側のモデル・探索設定を固定したまま、事前に凍結した多様な相手リーグとの成績で選びます。`ranking.json` は平均勝率、最苦手相手、Wilson下限、相手別勝敗を残します。`--resume` は正常に完走した候補×相手の組だけを再利用します。短時間制約を評価条件へ含めるときは `--errors-as-losses` を指定し、timeout・未判定試合を候補側の敗戦として分母へ含めます。

```bash
python tools/search_deck_fixed_league.py \
  --agent-src agents/belief_puct/src --model agents/belief_puct/src/model.pth \
  --candidate-dir agents/belief_puct/train/runs/trial-001/candidates \
  --opponents fixed_league_opponents.json \
  --games-per-opponent 40 --output-dir results/deck_search_trial-001 --resume
```

`fixed_league_opponents.json` は次の形式です。各パスはこのJSONファイルからの相対パスで指定できます。`model` は省略可能です。

```json
{
  "opponents": [
    {"name": "baseline", "agent_src": "agents/rl_mcts/src", "deck": "agents/rl_mcts/src/deck.csv", "weight": 1.0}
  ]
}
```

`--dry-run`はファイルを作らず、入力DB、選抜候補、後続5工程の計画だけを標準出力します。実行時は`--dry-run`を外します。候補選抜は既定で20試合未満の構成を除外します。候補の正式採用は、自己対戦ではなく、checkpoint・相手・seedを固定した多様な相手リーグの結果で行います。

## Phase 2b: 固定デッキ・混合相手RL

Phase 2aで選んだdeckを固定し、学習中モデル同士の自己対戦と、開始時checkpointをrun内にコピーして凍結したbelief_puctを混ぜます。凍結相手の手は学習データにせず、学習側の手だけで更新するため、過去の弱い方策を誤って教師にしません。追加の過去checkpointは`--frozen-model`で複数指定できます。

```bash
python agents/belief_puct/train/train_mixed_opponents.py \
  --deck agents/belief_puct/train/runs/phase2a-001/candidates/candidate_004/deck.csv \
  --initial-model agents/belief_puct/src/model.pth \
  --run-dir agents/belief_puct/train/runs/phase2b-001 \
  --iterations 10 --games-per-iteration 40 \
  --frozen-opponent-fraction 0.5 --search-count 10
```

各iteration完了時に、`checkpoints/`、`metrics.csv`、`events.jsonl`、モデル・optimizer・乱数状態を含む`training_state_latest.pth`を保存します。途中停止後は同じ総iteration数を指定して再開できます。

```bash
python agents/belief_puct/train/train_mixed_opponents.py \
  --deck agents/belief_puct/train/runs/phase2a-001/candidates/candidate_004/deck.csv \
  --initial-model agents/belief_puct/src/model.pth \
  --run-dir agents/belief_puct/train/runs/phase2b-001 \
  --iterations 10 --games-per-iteration 40 \
  --frozen-opponent-fraction 0.5 --search-count 10 \
  --resume agents/belief_puct/train/runs/phase2b-001/training_state_latest.pth
```

## 記録

- 事前登録: `docs/strategy/preregistration.md`
- 実験条件と全判定: `docs/strategy/experiment-report.md`
- Strategy Track向け報告: `docs/strategy/strategy-report.md`
- 詳細解説: `docs/strategy/strategy-explained.md`
- 生の対戦証拠: `results/*.json`
