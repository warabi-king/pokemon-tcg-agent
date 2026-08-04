# belief_puct

## 採用構成

`belief_puct` は、公開情報と観測履歴に矛盾しない非公開状態を復元し、その状態でTransformer方策・価値モデルを用いたPUCT探索を行うagentです。最終設定は次のとおりです。

- アーキテクチャ名: `C′: history-consistent single-determinization PUCT`
- 非公開状態: 観測履歴と60枚デッキ候補DBから1状態を決定的にサンプル
- 探索: 1決定化 × 24探索
- 方策・価値: 約50.1 MBのTransformer checkpoint
- デッキ・checkpoint: 固定リーグで選択したcluster 05 specialist
- 暗黙fallback: `model.pth`欠落時は例外。ランダム行動へは縮退しない

事前登録した候補Cのうち、複数決定化、追加模倣学習、相手モデルは比較実験で悪化したため最終構成から外しました。`candidate_05` の「05」は設計候補A〜Dではなく、デッキ・checkpointのクラスタ番号です。

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

今回の模倣学習はvalidation top-1精度を37.96%から53.98%まで改善しましたが、固定対戦では4勝16敗でした。そのため、`src/model.pth`は模倣重みではなくcluster 05の事前学習checkpointです。

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

## 記録

- 事前登録: `docs/strategy/preregistration.md`
- 実験条件と全判定: `docs/strategy/experiment-report.md`
- Strategy Track向け報告: `docs/strategy/strategy-report.md`
- 詳細解説: `docs/strategy/strategy-explained.md`
- 生の対戦証拠: `results/*.json`
