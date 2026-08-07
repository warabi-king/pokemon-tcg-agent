# belief_puct

## 採用構成

`belief_puct` は、公開情報と観測履歴に矛盾しない非公開状態を復元し、その状態でTransformer方策・価値モデルを用いたPUCT探索を行うagentです。最終設定は次のとおりです。

- アーキテクチャ名: `C′: history-consistent single-determinization PUCT`
- 非公開状態: 観測履歴と60枚デッキ候補DBから1状態を決定的にサンプル
- 探索: 1決定化 × 24探索
- 方策・価値: 約50.1 MBのTransformer checkpoint
- デッキ・checkpoint: 模倣学習完了checkpointを固定し、学習済み別モデルとのリーグ評価で選ぶ60枚デッキ
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

狙いは、デッキ検索DBを拡充しつつ、自分のデッキ構成を最適化し、多様な相手への対応力を得ることです。以下では**実装済み**（コードがある）、**実行済み**（このagentの成果物へ反映済み）、**未実装**を区別します。

### Phase 1: 模倣学習による初期モデル

- 状態: **実装済み・実行済み**。
- 既存の強い対戦ログから模倣学習を行い、初期 `belief_puct` モデルを作ります。現在の `src/model.pth` は、この模倣学習を完了した重みです。

### 候補選定と manifest 作成

`tools/run_belief_pipeline.py` は全工程を実行する一括学習器ではありません。現状で実行するのは、既存の60枚DBからクラスタ多様性を保って候補を選び、入力DBのSHA-256、候補、各Phaseの予定を run manifest に記録するところまでです。リーグ評価、追加学習、DB追記、holdout評価はこのコマンドからは実行されません。

公式由来DBを上書きせず、実行時は選抜DB・候補deck・イベントログを`train/runs/<run-id>/`へ保存します。

```bash
python tools/run_belief_pipeline.py \
  --database agents/belief_puct/src/rl_mcts/deck_candidates_by_wins.jsonl \
  --run-id trial-001 --candidate-count 16 --min-candidate-games 20 --dry-run
```

## Phase 2a: 学習済み別モデルとのリーグによるデッキ選抜

- 状態: **評価器・リーグ設定生成器は実装済み。候補デッキを採用した実行結果は未記録**。
- 候補間で変更するのは `deck.csv` だけです。候補側の `belief_puct` モデルと探索ロジックを固定し、学習済みの別モデルを相手に評価します。現在は `develop_naoki` の `16model_pretrained_upsize1` と `16model_pretrained_upsize2` を相手候補として利用しますが、対象モデル集合そのものを固定する設計ではありません。
- `random_baseline` は簡易比較用であり、正式なリーグ相手には含めません。
- `ranking.json` は平均勝率、最苦手相手、Wilson下限、相手別勝敗を残します。timeout/error は候補側の負けとして採点します。

upsize 相手のリーグJSONは次のように生成します。`upsize2` では、結果上の相手名を区別するため `--name-prefix upsize2` を必ず指定します。

```bash
python tools/create_upsize_fixed_league.py \
  --upsize-root <16model_pretrained_upsize1-root> \
  --output configs/upsize1_league.json

python tools/create_upsize_fixed_league.py \
  --upsize-root <16model_pretrained_upsize2-root> \
  --name-prefix upsize2 \
  --output configs/upsize2_league.json
```

リーグごとに候補を評価する例です。評価を止めた場合は、同じコマンドへ `--resume` を付けて再開します。完了済みの候補×相手 pairing は再利用されます。

```bash
python tools/search_deck_fixed_league.py \
  --agent-src agents/belief_puct/src --model agents/belief_puct/src/model.pth \
  --candidate-dir agents/belief_puct/train/runs/trial-001/candidates \
  --opponents configs/upsize1_league.json \
  --games-per-opponent 40 --output-dir results/deck_search_trial-001 \
  --errors-as-losses --resume
```

`run_belief_pipeline.py --dry-run` は候補選定・manifest作成の予定を標準出力し、ファイルを作りません。`search_deck_fixed_league.py --dry-run` は候補×相手の予定 pairing を表示します。候補選抜は既定で20試合未満の構成を除外します。

## Phase 2b: 固定デッキ・混合相手RL

- 状態: **実装済み・未実行**。`train_mixed_opponents.py` は、Phase 2aで選んだ自分のデッキを固定し、自己対戦・凍結 `belief_puct` checkpoint・外部agentを混ぜて追加学習できます。凍結相手と外部相手の手は学習sampleへ含めず、学習側だけを更新します。
- 外部相手はPhase 2aと同じ `opponents.json` で指定します。各相手を別processで起動し、相手ごとの `main.py`・deck・任意のモデルを用いるため、`upsize1`／`upsize2` の異なるモデル構造を `belief_puct` の `MyModel` として読み込む必要はありません。

```bash
python agents/belief_puct/train/train_mixed_opponents.py \
  --deck agents/belief_puct/train/runs/phase2a-001/candidates/candidate_012/deck.csv \
  --initial-model agents/belief_puct/src/model.pth \
  --external-opponents results/deck_search_phase2a-imitation-upsize2-001/opponents.json \
  --external-opponent-fraction 0.5 --frozen-opponent-fraction 0.25 \
  --run-dir agents/belief_puct/train/runs/phase2b-001 \
  --iterations 10 --games-per-iteration 40 --search-count 10
```

残りの25%は自己対戦になります。入力deck・外部相手deck・モデルのSHA-256、対戦種別ごとの勝敗、checkpoint、optimizer、乱数状態をrun内に記録し、`--resume`で再開できます。

## Phase 2c: 新デッキ生成・評価・DB追加

- 状態: **未実装**。
- 学習済みモデルで新しいデッキ候補を生成・変異し、Phase 2aと同様に評価します。評価済みのデッキと結果を検索DBへ追加し、DBを増やしてbelief推定の精度向上へつなげます。

## Phase 3: holdout評価と提出

- 状態: **holdout相手による最終評価は未実装**。提出物の作成・ローカル検証は実装済みです。
- 最終デッキとモデルを固定し、Phase 2で使っていないholdout相手で評価してから提出物へまとめます。

## 記録

- 事前登録: `docs/strategy/preregistration.md`
- 実験条件と全判定: `docs/strategy/experiment-report.md`
- Strategy Track向け報告: `docs/strategy/strategy-report.md`
- 詳細解説: `docs/strategy/strategy-explained.md`
- 生の対戦証拠: `results/*.json`
