# 自己対戦パイプライン（tools/pipeline）

各エージェントを「自分の手番モデル(self)」＋「探索時の相手モデル(opp)」の **1対1ペア** で持ち、
世代ごとに更新する自己対戦学習パイプライン。

```
Phase0(履歴模倣) ─▶ for g in 0..G-1: [ ①リーグ(履歴生成) ─▶ ②履歴で self/opp を継続学習 ]
```

- 参加エージェント = `tools/deck_generator/generated/deck_candidates_by_wins.jsonl` の
  **16 主要クラスタ代表**（`cl00`〜`cl15`）。クラスタリング・代表デッキ選出は
  [`tools/clustering_deck/`](../clustering_deck/README.md)（平均連結法による階層クラスタリング）が担当し、
  `hierarchical_cluster_representative=true` のレコードを `gen_agents.py` が拾う。
- ①リーグは黒箱（`PIPE_LEAGUE_CMD`）。仕様は [`docs/league_interface.md`](../../docs/league_interface.md)。
- 途中評価なし・成果物は世代ごとに保持。
- Phase0 は `agents/match_agents/imitation_group0-2`（`shards/` 由来の self+opp 重み）を
  **全16クラスタで積極的に再利用**する: 完全一致デッキ（`cl00/01/02`）は self を直接コピー、
  それ以外は最寄りの group を warm-start 初期値として own/opp シャードで学習し、
  シャードが空の場合も最寄り重みへフォールバックコピーする（詳細は `phase0.py` 参照）。

## 実行

```bash
# エージェント生成のみ確認（clusters.json と16デッキ）
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
| `PIPE_WORKERS` | 8 | Phase0 前処理（preprocess_all）の並列プロセス数 |
| `PIPE_MIN_AVAIL_MB` | 1200 | 前処理中に空きメモリがこの値を割ったら全バッファをフラッシュ（OOM防止） |
| `PIPE_ROOT` | `pipeline/` | 成果物の根 |
| `PIPE_OFFICIAL_EPISODES` | `episodes/official/` | Phase0 用リプレイ（.zip or JSON ディレクトリ） |
| `PIPE_DECKGEN_JSONL` | `tools/deck_generator/generated/deck_candidates_by_wins.jsonl` | 参加デッキ元（`hierarchical_cluster_*` 付与済み） |
| `PIPE_LEAGUE_CMD` | (空=スタブ) | ①リーグ実行コマンド（`{manifest}` `{out}` を置換） |
| `PIPE_LEAGUE_SEARCH_COUNT` | 10 | リーグ梱包エージェントの MCTS 探索回数（大量対戦を現実的な時間で回すため低め） |
| `PIPE_KEEP_INTERMEDIATE` | 1 | 0で中間世代と各世代の shards/episodes を消費後に削除（`--no-keep-intermediate` と同義） |

## モジュール

| ファイル | 役割 |
|---|---|
| `config.py` | 環境変数・パス・sys.path 設定 |
| `deck_utils.py` | デッキ入出力・ヒストグラム交差類似度・deck_filter |
| `preprocess.py` | 履歴→シャード（自前フィルタ対応、既存ライブラリ再利用）。世代ループの exact-deck 前処理と `episode_sources` を提供 |
| `preprocess_multi.py` | **単一パス多クラスタ前処理**。全エピソードを1回走査し全クラスタの own/opp シャードを同時生成（Phase0 が使用） |
| `trainer.py` | `train_imitation.py` サブプロセス実行ラッパ |
| `gen_agents.py` | 階層クラスタ代表(16)→エージェント（clusters.json + decks） |
| `package_agent.py` | self+opp+deck を実行可能 src/ に梱包 |
| `phase0.py` | 初期 self/opp をブートストラップ（前処理は preprocess_all で1パス。完全一致デッキは self コピー、それ以外も含め全クラスタで最寄り PRETRAINED を warm-start、シャード空はフォールバックコピー） |
| `generation.py` | 世代1回（梱包→リーグ→継続学習） |
| `orchestrate.py` | 入口（生成→Phase0→世代ループ） |
| `league_parallel.py` | ①リーグの並列版。full kaggle episode を多プロセスで生成（`PIPE_LEAGUE_CMD` で指定） |
| `league_stub.py` | ①の動作確認用スタブ（逐次。本番は `league_parallel.py`） |

## 成果物レイアウト

```
pipeline/
  clusters.json  decks/<name>.csv
  gen_000/agents/<name>/{self.pth,opp.pth,deck.csv,src/}
  gen_000/{manifest.json,episodes/,shards/,logs/}
  gen_001/ ...
```
