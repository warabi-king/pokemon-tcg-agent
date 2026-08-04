# 実験・検証レポート

## 1. 判定一覧

| 段階 | 固定条件 | 結果 | 95% Wilson区間 | 判定 |
|---|---|---:|---:|---|
| Stage 1a | 選定時と同じ主要5相手、fresh seed、各20試合 | 54勝46敗、54.0% | 44.26%〜63.44% | 60%未満かつ下限50%以下で未達 |
| Stage 1b | 選定に使わなかったholdout 5相手、各20試合 | 66勝34敗、66.0% | 56.28%〜74.54% | 60%以上かつ下限50%超で達成 |
| Stage 2a | 現行MCTS+RL+Transformer、同一cluster 05デッキ | 91勝9敗、91.0% | 83.77%〜95.19% | 達成 |
| Stage 2b | 両agentが各自のデッキを使用 | 40勝60敗、40.0% | 30.94%〜49.80% | 未達 |
| Stage 3 | Kaggle実対戦 | 未提出 | ― | 未確認 |

結論は一枚岩ではない。C′方策は同一デッキで現行agentを明確に上回り、未見相手リーグでも勝ち越した。一方、主要5相手では不確実性を含めて勝ち越しを証明できず、現行agentとの各自デッキ比較ではcluster 05のデッキ相性を覆せなかった。ローカル結果をKaggle上の勝利とは扱わない。

## 2. 最新ルールと提出制約の確認

2026-08-03時点で公式ページを再確認した。

- [大会公式サイト](https://ptcg-abc.pokemon.co.jp/)では、Strategy Categoryは2026-09-14 8:59 JSTまで、1チーム1回提出で、AI agentの安定性、デッキコンセプト、Simulation成績を総合評価すると説明されている。各プレイヤーの持ち時間は最大10分。
- [Kaggle Simulation説明](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/description)では、`.tar.gz`直下に`main.py`と`deck.csv`が必要で、実行場所は`/kaggle_simulations/agent/`、サイズ上限197.7 MiB、vCPU 2、RAM 12.2 GiB、HDD 11.8 GiBとされる。日次5提出、最新2提出がactive。
- [cabt公式ドキュメント](https://matsuoinstitute.github.io/cabt/)では、観測にログ、公開盤面、手札、サイド、山札枚数、合法選択肢が含まれ、agentは合法optionのindex列を返す。`battle_start`は60枚デッキを要求する。

Kaggleの実アップロードは外部状態を変更するため実施していない。

## 3. 実行環境と資源

| 項目 | 実測・確認結果 |
|---|---|
| CPU | Apple M5 |
| メモリ | 24 GiB |
| Python | 3.11.15 |
| PyTorch | 2.12.1 |
| CUDA | 利用不可 |
| MPS | 使用したPython実行環境では利用不可 |
| 学習・評価device | CPU |
| 最終Transformer | 12,533,634 parameters、50,143,233 bytes |
| CPU短時間benchmark | batch 32、20反復、median 1.408 ms/batch、p95 1.581 ms/batch |
| `codex-auto`使用量 | 約949 MiB |
| agent `src/` | 約54 MiB |
| agent `train/` | 約632 MiB |
| ディスク空き | 約276 GiB |

実validation sampleを使った短時間benchmarkでは、PyTorchはMPS build済みだが`is_available=False`だったためMPSを測定できなかった。CPUは4 thread、batch 32、warmup 3、20反復でmedian 1.408 ms、p95 1.581 msだった。これはニューラルネットワークforwardだけの値であり、cabt探索を含む試合時間とは分けて扱う。追加GPU導入は行わずCPUで探索予算を比較した。生成物は200 GiB上限を大幅に下回る。生結果は`work/audits/hardware-benchmark.json`に保存した。

OS標準GitはCommand Line Tools不備、同梱fallback Gitはsandboxの`Operation not permitted`で実行できなかった。branchはworktreeの`.git`参照で`codex-auto`と確認した。以降の書込み先はすべてこのworktree内へ限定した。

## 4. データ監査と分割

公式日次episode ZIPは2026-07-01〜23を、全展開せずZIP memberとmanifestから監査した。監査結果は`work/audits/strategy-assets.json`に保存している。

模倣学習は日付で次のように分離した。

| split | 期間 | episode | decision sample | 用途 |
|---|---|---:|---:|---|
| train | 07-01〜17 | 340 | 50,339 | 勾配更新 |
| validation | 07-18〜20 | 60 | 9,758 | epoch評価 |
| held-out test | 07-21〜23 | 未抽出 | 未抽出 | モデル選択に使用しない |

各日20 episodeを固定seedで選び、同一episode派生sampleがsplitをまたがない。`prepare_imitation.py`はZIP全体を展開せず選択JSONだけを読み、manifestへ入力archive、seed、失敗数、sample数を保存する。train/validationとも読込失敗は0件だった。

最終提出モデルは今回の模倣学習重みではなく、固定リーグで選定した事前学習checkpointである。したがって07-21〜23のログ精度は最終モデルの選定に利用していない。事前学習checkpoint自体の元学習期間を完全に再構成できていない点は残る制約である。

## 5. 既存資産とStage 2相手の凍結

既存agent、sample submission、cabt SDK、16個のデッキクラスタ、事前学習checkpoint、公式episode、対戦ツールを読み取り専用で監査した。Stage 2相手は`develop_nomura/agents/rl_mcts/src`のMCTS+RL+Transformerに固定した。

| 固定物 | SHA-256 |
|---|---|
| 最終C′ `model.pth` | `28877e029f52666b40dab09046b730c541b9da72d07020f8126b9efc53b95c03` |
| 最終cluster 05 `deck.csv` | `01b152e466925c6c5cf4c0aeaaa019da3a7f7ebd967948e3fd5e1a891df156f9` |
| Stage 2相手 `model.pth` | `d4e95f604f0c0293b3af511e17081282a6d10ea1676df949c6e2e7fbe40af83a` |
| Stage 2相手 `deck.csv` | `1156379af39e71bc83eecb50d1e04c5cc480501d293621fcc89c2d355d99be78` |

## 6. 方式比較とアブレーション

全対戦はagentを別processへ隔離し、同一seedを先後入替の2試合へ使用した。小規模20試合は方式選定専用で、最終成功判定には用いない。

| 仮説・変更 | 条件 | 結果 | 判断 |
|---|---|---:|---|
| 複数の非公開状態を平均すれば強い | 3決定化×8探索 vs 1決定化×24探索、20試合 | 4勝16敗 | 棄却。CPU予算分散で木が浅くなった |
| belief整合化自体が旧決定化より強い | belief 1×24 vs legacy 24、20試合 | 9勝11敗 | 勝率改善は不支持。整合性・説明可能性の層としてのみ残す |
| 探索を24から32へ増やす | 32探索 vs 24探索、20試合 | 9勝11敗 | 棄却。追加時間に見合う改善なし |
| 公式行動模倣で方策が強くなる | epoch 1 vs既存MCTS、20試合 | 4勝16敗 | 棄却。validation精度と対戦強度が乖離 |
| 相手手番専用checkpointが探索を改善 | 主要5相手、各4試合 | 8勝12敗 | 棄却 |

belief単体の勝率効果を主張しないことが重要である。支持されたのは「限られた探索予算を1つの履歴整合状態へ集中する」構成であり、beliefは矛盾防止の役割を担う。

### 模倣学習

| epoch | train loss | train top-1 | validation top-1 | 約所要時間 |
|---:|---:|---:|---:|---:|
| 初期 | ― | ― | 37.96% | 0.55秒 |
| 0 | 1.5854 | 51.45% | 53.61% | 12.0秒 |
| 1（resume） | 1.4648 | 56.31% | 53.98% | 12.0秒 |

モデル、optimizer、Python RNG、PyTorch RNG、epoch、設定をfull checkpointへ保存し、epoch 1は`imitation_epoch_000.pth`から再開した。validation精度は上がったが、対戦4勝16敗のため最終重みへ採用しなかった。

## 7. デッキ・checkpoint候補選定

候補は同じC′方策を使い、対応するクラスタdeckとcheckpointだけを交換した。主要5クラスタ00〜04を各4試合、計20試合でscreeningした。これは選定用であり、区間が広いため一般化主張には用いない。

| cluster候補 | 勝-敗-分 | score |
|---:|---:|---:|
| 01 | 10-10-0 | 50.0% |
| 02 | 9-11-0 | 45.0% |
| 03 | 12-8-0 | 60.0% |
| 04 | 8-11-1 | 42.5% |
| **05** | **13-7-0** | **65.0%** |
| 06 | 2-18-0 | 10.0% |
| 07 | 3-17-0 | 15.0% |
| 08 | 12-8-0 | 60.0% |
| 09 | 4-16-0 | 20.0% |
| 10 | 7-13-0 | 35.0% |

cluster 05を採用した。デッキ構築と方策の交絡を避けるため、Stage 2では同一デッキと各自デッキを別試験にした。

## 8. Stage 1

### 8.1 主要5相手、fresh seed

候補選定と同じ相手identityだが、未使用seed 80000〜80409を使用した。各相手20試合、先後10試合ずつ。

| 相手 | 主なデッキ系統 | 勝-敗 | score |
|---:|---|---:|---:|
| 00 | フーディン／ノココッチ | 15-5 | 75% |
| 01 | マリィのオーロンゲex／ユキメノコ | 9-11 | 45% |
| 02 | イワパレス／メガガルーラex | 12-8 | 60% |
| 03 | メガスターミーex | 7-13 | 35% |
| 04 | シロナのガブリアスex | 11-9 | 55% |
| 合計 | 非加重 | **54-46** | **54.0% [44.26%, 63.44%]** |

主要5相手は元データの約81.49%を占める。母集団shareで補助的に重み付けすると59.65%だが、相手identityを選定時に見ているため正式判定は非加重54%を使う。cluster 03への弱さが全体を押し下げた。

### 8.2 未見相手holdout

候補・パラメータ選定に使用しなかったクラスタ06〜10へ、未使用seed 92060〜92109で各20試合を行った。

| holdout相手 | 勝-敗 | score |
|---:|---:|---:|
| 06 | 17-3 | 85% |
| 07 | 15-5 | 75% |
| 08 | 7-13 | 35% |
| 09 | 13-7 | 65% |
| 10 | 14-6 | 70% |
| 合計 | **66-34** | **66.0% [56.28%, 74.54%]** |

未見相手ゲートは目標60%以上、95%区間下限50%超を満たした。一方でcluster 08には35%であり、全デッキへの頑健性は未達である。holdout 5クラスタの元データshareは主要5より小さいため、Kaggle母集団で66%を期待できるとは言えない。

## 9. Stage 2

### 9.1 同一デッキ

両agentにcluster 05デッキを与え、方策差を隔離した。seed 90000〜90049、100試合、先後各50。

- 91勝9敗、score 91.0%
- 95% Wilson区間 83.77%〜95.19%
- 先攻時45勝、後攻時46勝
- error 0
- 434.84秒、平均4.35秒/試合

点推定60%以上かつ区間下限50%超を満たした。デッキを固定したとき、C′の探索設定と選定checkpointの組合せが現行MCTS+RL+Transformerより強い証拠である。ただし、belief単体の効果ではなく複数差分を含む。

### 9.2 各自デッキ

C′はcluster 05、相手は自身のデッキを使用した。seed 90100〜90149、100試合、先後各50。

- 40勝60敗、score 40.0%
- 95% Wilson区間 30.94%〜49.80%
- 先攻時27勝、後攻時13勝
- error 0
- 211.59秒、平均2.12秒/試合

成功条件を満たさず、区間上限も50%未満だった。大きな先攻差と、同一デッキ91%から各自デッキ40%への反転は、方策改善よりデッキ相性が支配的になった証拠である。次の優先実験は、cluster 08とStage 2相手デッキを固定した対面別deck searchである。

## 10. Stage 3と提出物

### ローカル提出検証

`dist/submission_belief_puct.tar.gz`を構築し、`tools/validate_submission.py`で検証した。

| 項目 | 結果 |
|---|---|
| archive形式 | `.tar.gz` |
| size | 48,594,554 bytes（約46.34 MiB、上限197.7 MiB未満） |
| SHA-256 | `28dfb60c3cfc97718ca2b9f649a7e872e1e5ca9959d87dd150826bc3667089e6` |
| top-level | `main.py`、`deck.csv`、`cg/`、`rl_mcts/`、`model.pth` |
| deck | 60枚、cabt自己対戦で有効 |
| 不要物 | `train/`、`logs/`、`.git`、`__pycache__`、`.pyc`なし |
| 実行path | `/kaggle_simulations/agent/`を模した一時pathから別process import成功 |
| 自己対戦 | seed 99100〜99102、先後入替6試合、6完了、error 0 |

`py_compile`は提出・学習・評価・検証コードで成功し、`unittest` 2件も成功した。アーカイブ検証は静的構造だけでなく、展開済みコピーを別processで実際に動かしている。

### 未確認

Kaggleへはアップロードしていない。したがってValidation Episode、rating、対戦相手、Kaggle CPUでの実時間、leaderboard成績はすべて未確認であり、第三段階は未達ではなく「未確認」とする。実アップロードにはユーザー承認が必要である。

## 11. 重要な失敗と次の実験

1. **複数決定化の失敗**: 理論上の情報頑健性より、24探索を1本へ集中する深さが重要だった。次は単純投票ではなく、root共有またはbatched leaf評価で比較する。
2. **模倣精度と勝率の乖離**: 平均的なログ行動は採用デッキ固有の勝ち筋と探索分布を改善しなかった。次は勝者・rating・デッキ別重み、またはMCTS policyへのdistillationを使う。
3. **cluster 08弱点**: 未見リーグでも7勝13敗。対面の負け局面を抽出し、ブリジュラスex系への必要な打点・エネルギー・サイド交換を特定する。
4. **各自デッキStage 2失敗**: 40勝60敗で、特に後攻13/50。方策を固定したdeck searchと、先後別leagueを行う。
5. **belief効果未証明**: 履歴整合性は実装できたが9勝11敗。候補デッキposteriorのcalibration、複数仮説のroot共有、観測カード数別の勝率を測る。

## 12. 証拠ファイル

- 事前登録: `docs/strategy/preregistration.md`
- Stage 1主要5集約: `results/stage1_final_candidate05_aggregate_100.json`
- Stage 1 holdout集約: `results/stage1_holdout_candidate05_aggregate_100.json`
- Stage 2同一デッキ: `results/stage2_final_candidate05_vs_current_rl_mcts_same_deck_100.json`
- Stage 2各自デッキ: `results/stage2_final_candidate05_vs_current_rl_mcts_own_decks_100.json`
- 提出検証: `results/submission_validation.json`
- 学習設定・metrics: `agents/belief_puct/train/logs/`
- CPU/MPS benchmark: `work/audits/hardware-benchmark.json`
- 再開checkpoint: `agents/belief_puct/train/checkpoints/imitation_epoch_001.pth`
- 再現手順: `agents/belief_puct/README.md`
