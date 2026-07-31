# 自己対戦パイプライン（tools/pipeline）

各エージェントを「自分の手番モデル(self)」＋「探索時の相手モデル(opp)」の **1対1ペア** で持ち、
世代ごとに更新する自己対戦学習パイプライン。

```
Phase0(履歴模倣) ─▶ for g in 0..G-1: [ ①リーグ(履歴生成) ─▶ ②履歴で self/opp を継続学習 ]
```

- 参加エージェント = `deck_generator/generated/deck_candidates_by_wins.jsonl` の 63 クラスタ代表（`cl00`〜）。
- ①リーグは黒箱（`PIPE_LEAGUE_CMD`）。仕様は [`docs/league_interface.md`](../../docs/league_interface.md)。
- 途中評価なし・成果物は世代ごとに保持。

## 実行

```bash
# エージェント生成のみ確認（clusters.json と約63デッキ）
python tools/pipeline/orchestrate.py --dry-run

# 全体（Phase0 → 世代ループ）。要: 公式リプレイ or リーグ
python tools/pipeline/orchestrate.py
```

## 設定（環境変数・既定値は config.py）

| 変数 | 既定 | 意味 |
|---|---|---|
| `PIPE_GENERATIONS` | 5 | 世代数 G |
| `PIPE_EPOCHS_PER_GEN` | 3 | 各世代の学習 epochs |
| `PIPE_PHASE0_EPOCHS` | 5 | Phase0 の epochs |
| `PIPE_SEARCH_COUNT` | 50 | 梱包エージェントの MCTS 回数 |
| `PIPE_SIM_THRESHOLD` | 0.75 | 近いデッキ判定/クラスタ割当しきい値 |
| `PIPE_LR` / `PIPE_BATCH_SIZE` | 3e-4 / 128 | 学習率 / バッチ |
| `PIPE_WARM_START` | 1 | 世代間で前世代重みから継続学習 |
| `PIPE_ROOT` | `pipeline/` | 成果物の根 |
| `PIPE_OFFICIAL_EPISODES` | `episodes/official/` | Phase0 用リプレイ（.zip or JSON ディレクトリ） |
| `PIPE_DECKGEN_JSONL` | `deck_generator/generated/deck_candidates_by_wins.jsonl` | 参加デッキ元 |
| `PIPE_LEAGUE_CMD` | (空=スタブ) | ①リーグ実行コマンド（`{manifest}` `{out}` を置換） |

## モジュール

| ファイル | 役割 |
|---|---|
| `config.py` | 環境変数・パス・sys.path 設定 |
| `deck_utils.py` | デッキ入出力・ヒストグラム交差類似度・deck_filter |
| `preprocess.py` | 履歴→シャード（自前フィルタ対応、既存ライブラリ再利用） |
| `trainer.py` | `train_imitation.py` サブプロセス実行ラッパ |
| `gen_agents.py` | クラスタ→約63エージェント（clusters.json + decks） |
| `package_agent.py` | self+opp+deck を実行可能 src/ に梱包 |
| `phase0.py` | 初期 self/opp をブートストラップ（近いデッキはコピー、opp は新規学習） |
| `generation.py` | 世代1回（梱包→リーグ→継続学習） |
| `orchestrate.py` | 入口（生成→Phase0→世代ループ） |
| `league_stub.py` | ①の動作確認用スタブ（本番は `PIPE_LEAGUE_CMD`） |

## 成果物レイアウト

```
pipeline/
  clusters.json  decks/<name>.csv
  gen_000/agents/<name>/{self.pth,opp.pth,deck.csv,src/}
  gen_000/{manifest.json,episodes/,shards/,logs/}
  gen_001/ ...
```
