# Deck Generator Tools

`tools/deck_generator` には、対戦結果 JSON のダウンロード、候補デッキ DB の生成、MLP によるデッキ補完モデルの学習に使うツールがあります。

生成物は主に以下へ出力します。

- `daily_dataset/`: 日付ごとの daily dataset ZIP
- `generated/`: 候補デッキ DB、集約済み DB、学習済みモデル

## 1. 対戦結果 JSON ZIP をダウンロードする

`manifest.csv` の `daily_dataset_url` から daily dataset を取得します。JSON は展開せず、ZIP のまま保存します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\download_daily_dataset.py --date 2026-07-19 --zip-only
```

複数日を取得する場合は `--date` を繰り返します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\download_daily_dataset.py --date 2026-07-01 --date 2026-07-02 --zip-only
```

主なオプション:

- `--dry-run`: ダウンロード対象だけ確認する。
- `--force`: 既存 ZIP があっても再ダウンロードする。
- `--output-dir PATH`: 保存先を変更する。
- `--ignore-space-check`: 空き容量チェックを無視する。

出力例:

```text
tools/deck_generator/daily_dataset/
  2026-07-19/
    2026-07-19.zip
```

## 2. 候補デッキ DB を生成する

`daily_dataset/*/*.zip` を読み、各試合 JSON に含まれる完全な 60 枚デッキを `deck_candidates.jsonl` に保存します。ZIP は展開せずに直接読みます。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py build-index
```

出力:

```text
tools/deck_generator/generated/deck_candidates.jsonl
tools/deck_generator/generated/deck_candidates_summary.json
```

動作確認用に読み込む episode 数を制限できます。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py build-index --max-episodes 100
```

## 3. 同一デッキを勝利数で集約する

`deck_candidates.jsonl` の同一デッキをまとめ、試合数、勝利数、敗北数、引き分け数、勝率を集約します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py aggregate-wins
```

出力:

```text
tools/deck_generator/generated/deck_candidates_by_wins.jsonl
```

## 4. 類似デッキをクラスタリングして重みを付ける

カード ID のヒストグラム類似度でデッキをクラスタリングし、`deck_candidates_by_wins.jsonl` にクラスタ情報と `cluster_weight` を追加します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py cluster-weights --threshold 0.75
```

`cluster_weight` は次の式です。

```text
全デッキ数 / (cluster_members * クラスタ数)
```

同じようなデッキが多いクラスタほど、各デッキの選択確率は低くなります。

## 5. MLP デッキ補完モデルを学習する

デフォルトでは `generated/deck_candidates_by_wins.jsonl` を使って学習します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train
```

出力:

```text
tools/deck_generator/generated/deck_mlp.pt
```

学習タスクは次の形です。

```text
観測済みカード count + 観測枚数 -> 60枚デッキ全体のカード count
```

各 60 枚デッキから一部カードをランダムに抜き出し、「対戦中に見えているカード」とみなします。元の 60 枚デッキ全体を教師ラベルとして学習します。

学習時のデッキ選択は、`--max-decks` の有無に関わらず `cluster_weight` で重み付けされます。つまり、多く登場する類似デッキのクラスタは、学習バッチに出にくくなります。

カード枚数の正規化はカードごとに異なります。通常カードは `4`、ACE SPEC は `1`、基本エネルギーは学習データ中のそのカードの最大採用枚数を上限として、`count / card_max_count` の形にします。この `card_max_count` は checkpoint の `count_scales` に保存され、推論時の逆変換にも使われます。

主な学習オプション:

- `--epochs N`: 学習 epoch 数。
- `--batch-size N`: batch size。
- `--hidden-size N`: MLP の隠れ層サイズ。
- `--layers N`: MLP の隠れ層数。
- `--samples-per-deck N`: 1 デッキから作る部分観測サンプル数。
- `--min-observed N`: 観測済みカード枚数の最小値。
- `--max-observed N`: 観測済みカード枚数の最大値。
- `--max-decks N`: 学習に使う候補デッキ数の上限。
- `--deck-sampling cluster-weight|cluster-wins`: デッキ選択確率の計算方法。
  デフォルトは`cluster-weight`
  `cluster-weight`:`P ∝ cluster_weight`
  `cluster-wins`:`P ∝ cluster_weight * (1 + log1p(cluster_wins))`

- `--win-rate-weight N`: 勝率が高いデッキの loss 重みを増やす。`0` で無効。
- `--device gpu`: GPU を使う。指定しない場合は CPU。

`--deck-sampling cluster-weight` はデフォルトです。選択確率は `cluster_weight` に比例します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --max-decks 50000
```

勝利数の多いクラスタを少し優先したい場合は `--deck-sampling cluster-wins` を指定します。選択確率は `cluster_weight * (1 + log1p(cluster_wins))` に比例します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --max-decks 50000 --deck-sampling cluster-wins
```

勝率の高いデッキの loss も重くする場合は `--win-rate-weight` を指定します。loss 重みは `1 + win_rate_weight * log1p(win_rate)` です。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --max-decks 50000 --deck-sampling cluster-wins --win-rate-weight 2
```

GPU を使う場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --device gpu
```

軽量な動作確認:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --epochs 1 --batch-size 64 --hidden-size 64 --layers 1 --samples-per-deck 1 --max-decks 100 --output tools\deck_generator\generated\debug_deck_mlp.pt
```

## 6. MLP でデッキを推論する

学習済みモデルを使い、見えているカードから 60 枚デッキ候補を生成します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py predict --observed 431,1186,1 --json
```

checkpoint を指定する場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py predict --checkpoint tools\deck_generator\generated\deck_mlp.pt --observed 431,1186,1 --json
```

推論時に GPU を使う場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py predict --device gpu --observed 431,1186,1 --json
```

推論時は、観測済みカード枚数を必ず満たすように、予測されたカード count をもとに 60 枚へ丸めます。カードを追加する順位は `predicted_count / count_scale` の正規化済みスコアで決めるため、基本エネルギーの `count_scale` が大きいだけで過剰に選ばれることを抑えます。その際、同名 4 枚制限、基本エネルギー例外、ACE SPEC 1 枚制限を考慮します。
