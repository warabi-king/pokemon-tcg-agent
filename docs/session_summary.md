# 作業サマリ（pokemon-tcg-agent / 2026-08-01）

自己対戦・模倣学習まわりの調査・評価・設計・実装のセッション記録。

---

## 1. 対象資産の把握

- **履歴（模倣学習）由来のデッキ/モデル 3体**: `agents/match_agents/imitation_group0/1/2`（各 `deck.csv` 60枚 + `model.pth`）。`tools/group_decks.py` 由来。
- **自己対戦（round-robin）で学習したモデル 8体**: `agents/match_agents/00〜07`（`train_match_agents.py` 由来）。ただし学習ログ（`train/logs/train_metrics.csv`）は games=1〜4 の**試験run レベル**で本格学習には未到達。
- **モデル構造** (`rl_mcts/model.py`): Transformer系の value + policy 二出力（EmbeddingBag→Encoder→value / Decoder(cross-attn)→policy）。`d_model=128`。重み約48MB。
- **deck_generator**: `deck_completion.py`（`build-index`→`aggregate-wins`→`cluster-weights --threshold 0.75`）が候補デッキDBを生成。`generated/deck_candidates_by_wins.jsonl` = 633デッキ / **63クラスタ**。MLP・word2vec・Transformer-CBOW のデッキ生成モデルも同梱。

---

## 2. 評価（対戦）

### 2-1. 4体総当たり（imitation_group0/1/2 + self-play 00）
- 設定: 各 `RlMctsAgent()`（**search_count=50 のMCTS**）。imitation の探索相手モデルは最終的に「自分と同じ」に統一。各カード5試合・自己対戦除外。

| 順位 | エージェント | 勝率 |
|---|---|---|
| 1 | imitation_group0 | 66.7% |
| 2 | imitation_group2 | 60.0% |
| 3 | imitation_group1 | 53.3% |
| 4 | selfplay_00 | 20.0% |

- **imitation 3体が self-play 00 を圧倒**（00は極少量学習のため順当）。imitation 間は三すくみ（group0→group1→group2→group0）。

### 2-2. group2 vs ①（提出/推論用エージェント）
| 相手 | group2勝率 |
|---|---|
| rl_mcts | 100% (5-0) |
| rl_mcts_sample | 60% (3-2) |
| rl_mcts_r_robin | 60% (3-2) |
| **合計** | **73.3% (11-4)** |

- group2 は① 全相手に勝ち越し。`rl_mcts` と `rl_mcts_r_robin` はモデル・デッキ同一だが探索ロジック差で結果が変わることも確認。

---

## 3. コード理解（Q&A で確定した事実）

- **MCTS = 先読み本体**。`mcts_agent(search_count=N)` が「相手の隠れ札を `random.sample` で仮定（複数世界）→ 木を N 回展開 → 葉を value ネットで評価（AlphaZero型、ランダムプレイアウトではない）」。`SEARCH_COUNT=10` はモジュール既定だが `RlMctsAgent` が 50 に上書きするため未使用。→ **これまでの評価は全て 50回MCTS**。
- **自分の手のNN評価に相手デッキの中身は入らない** (`get_encoder_input(obs, your_deck)`)。相手は「見えている場＋枚数（deckCount/handCount）＋トラッシュ」のみ。完全デッキリストが渡るのは自分側だけ。
- **相手デッキ/手札の“予測”はMCTS探索時のみ** (`opponent.py` の `infer_opponent_deck` 等)。唯一の例外は「相手activeが裏向きのときの推測ポケモン（旧: 全部カビゴン→現: 推定デッキのたねから）」。
- **模倣学習** (`train_imitation.py` + `imitation_data.py`): 探索を使わない教師あり。policy=実際の手の交差エントロピー、value=実際の勝敗。両プレイヤー分を抽出、`--role own/opponent` と `--deck-groups/--target-group` で絞り込み可。
- **「探索時の相手モデルを学習する専用プログラムは存在しない」**。全学習スクリプトの `mcts_agent` 呼び出しは `opponent_model` 未指定（＝木の中の相手も自分のモデルで評価）。ただし `preprocess_episodes.py --role opponent`（相手役データ生成）＋ `train_imitation.py`（汎用）で作る部品は揃っている。

---

## 4. 目標パイプライン（確定方針）

各エージェントを **self（自分の手番モデル）＋ opp（探索時の相手モデル）の1対1ペア**で持ち、世代ごとに更新。

```
Phase0(履歴模倣) ─▶ for g in 0..G-1: [ ①リーグ(履歴生成) ─▶ ②履歴で self/opp を継続学習 ]
```

### 決定事項
1. opp は1エージェントにつき1つ。初期値は `role=opponent` で新規学習。
2. **ループ中の評価なし**・採否ゲートなし。世代数 G を最初に固定。
3. 成果物は**消さない**（retention なし）。
4. ①リーグは「agentsマニフェスト→ episode JSON」を出す黒箱（友人が並列化）。`preprocess_episodes.py` が読める kaggle 形式前提。
5. 参加エージェント = `deck_candidates_by_wins.jsonl` の 63クラスタ代表（by_wins 先頭）。名前 `cl00`〜`cl62`。既存 group0/1/2 は残置し別名で作成。
6. 近いデッキ（類似度≥0.75）は事前学習を**コピー**: `cl00←group0 / cl01←group1 / cl03←group2`（他は次点≤0.72で曖昧さなし）。self はコピー、opp は新規学習。

### データの作り分け
- self データ = 自分のデッキで打った手（role=own / リーグは exact-deck 一致）
- opp データ = 自分と対戦した相手が打った手（role=opponent）
- value = 実際の勝敗

---

## 5. 実装（`tools/pipeline/`）

| ファイル | 役割 |
|---|---|
| `config.py` | 全設定を環境変数化（下表）。sys.path 設定 |
| `deck_utils.py` | ヒストグラム交差類似度・deck_filter・デッキ入出力 |
| `preprocess.py` | 履歴→シャード（自前フィルタ、既存ライブラリ再利用） |
| `trainer.py` | `train_imitation.py` 実行ラッパ（warm-start・空シャードskip） |
| `gen_agents.py` | クラスタ→63エージェント（clusters.json + decks） |
| `package_agent.py` | self+opp+deck を実行可能 src に梱包（`RlMctsAgent(opponent_model_path=...)` 配線） |
| `phase0.py` | 初期 self/opp（近いデッキはコピー、opp は新規学習） |
| `generation.py` | 世代1回（梱包→リーグ→継続学習、データ無しは copy-forward） |
| `orchestrate.py` | 入口（生成→Phase0→世代ループ） |
| `league_stub.py` | ①の動作確認スタブ（本番は `PIPE_LEAGUE_CMD`） |
| `docs/league_interface.md` | ①の入出力契約 |
| `tools/pipeline/README.md` | 使い方 |

### 環境変数（既定値）
`PIPE_GENERATIONS=5`, `PIPE_EPOCHS_PER_GEN=3`, `PIPE_PHASE0_EPOCHS=5`, `PIPE_SEARCH_COUNT=50`,
`PIPE_SIM_THRESHOLD=0.75`, `PIPE_LR=3e-4`, `PIPE_BATCH_SIZE=128`, `PIPE_WARM_START=1`, `PIPE_SHARD_SIZE=20000`,
`PIPE_ROOT=pipeline/`, `PIPE_OFFICIAL_EPISODES=episodes/official/`,
`PIPE_DECKGEN_JSONL=deck_generator/generated/deck_candidates_by_wins.jsonl`, `PIPE_LEAGUE_CMD`(空=スタブ)

### 検証済み
- `orchestrate.py --dry-run` → 63エージェント生成（cl00=cluster0=group0デッキ）。
- **1世代フルサイクル**を軽量設定（search_count=2/1試合/1epoch, 2体）で完走:
  梱包 → スタブ対戦(episode生成) → 前処理(cl00_own=81, cl00_opp=85 … self/opp対称で分離確認) → warm-start学習 → `gen_001/{cl00,cl01}/{self,opp}.pth` 出力。
- バグ修正1件: `env.toJSON()` が dict を返すため `json.dumps` でシリアライズ。

### 本番実行に必要（未配置）
1. 公式リプレイを `PIPE_OFFICIAL_EPISODES` に配置（`deck_generator/download_daily_dataset.py`）。
2. 友人の並列リーグを `PIPE_LEAGUE_CMD='... --agents {manifest} --out {out}'` で指定。

```bash
python tools/pipeline/orchestrate.py   # Phase0 → 世代ループ ×G
```

---

## 6. その他の作業
- group2 の提出物作成手順を確認（`prepare_match_agent_submission.py --agent imitation_group2` → `dist/`）。※途中で dist の tar.gz が外部整理で消失、再生成可。
- README.md に「自己対戦パイプライン（設計メモ）」を追記。
- クリーンアップ: 評価で生成した src 複製・`results/`・スモークテストの一時世代物を削除（`.pth` は 27→23個）。`pipeline/clusters.json` と `decks/*.csv`（63体）は残置。

---

## 7. 次の候補
- ①到着前にスタブで小規模 Phase0＋数世代を試し、学習ログ（loss/accuracy）で挙動確認。
- 63体 × Phase0 の計算量・ディスクの実測見積り。
