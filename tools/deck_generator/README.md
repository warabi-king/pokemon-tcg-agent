# Deck Generator Tools

`tools/deck_generator` には、日別の対戦結果JSONをダウンロードし、
候補デッキDBを生成し、MLPによるデッキ補完モデルを学習するためのツールを置いています。

以下の生成物はGit管理対象外です。

- `daily_dataset/`: Kaggleからダウンロードした日別dataset ZIP
- `generated/`: 候補デッキDB、summary、学習済みモデル

コマンドはリポジトリルートで実行してください。

## 1. 対戦結果JSON ZIPをダウンロードする

`manifest.csv` には、日別Kaggle datasetのURLが入っています。
JSONを展開せず、ZIPのまま保存するには `--zip-only` を指定します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\download_daily_dataset.py --date 2026-07-19 --zip-only
```

複数日をまとめて取得する場合は、`--date` を繰り返します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\download_daily_dataset.py --date 2026-07-01 --date 2026-07-02 --zip-only
```

主なオプション:

- `--dry-run`: ダウンロード対象だけ表示し、実際には取得しない。
- `--force`: 既にZIPが存在していても再ダウンロードする。
- `--output-dir PATH`: 保存先datasetディレクトリを変更する。
- `--ignore-space-check`: `manifest.csv` のサイズ見積もりによる空き容量チェックを無視する。

デフォルトの保存先:

```text
tools/deck_generator/daily_dataset/
  2026-07-19/
    2026-07-19.zip
```

後続ツールはZIP内のJSONを直接読むため、展開は不要です。

## 2. 候補デッキDBを生成する

`daily_dataset/*/*.zip` にある全ZIPを読み、`deck_candidates.jsonl` を生成します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py build-index
```

デフォルトの出力先:

```text
tools/deck_generator/generated/deck_candidates.jsonl
tools/deck_generator/generated/deck_candidates_summary.json
```

各episodeでは、各プレイヤーの初期60枚デッキが最初の60枚 `action` に入っています。
候補デッキDBはこの完全な60枚デッキ情報をそのまま使用します。

少量データで動作確認する場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py build-index --max-episodes 100
```

生成したDBは、簡易kNN baselineとしても使えます。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py complete --observed 431,1186,1 --json
```

`--observed` には、既に見えている相手カードIDをカンマ区切りで指定します。

## 3. MLPデッキ補完モデルを学習する

`deck_candidates.jsonl` を教師データとして、MLPを学習します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train
```

デフォルトの出力先:

```text
tools/deck_generator/generated/deck_mlp.pt
```

学習タスクは次の形です。

```text
観測済みカードcount + 観測枚数 -> 60枚デッキ全体のカードcount
```

各60枚デッキから一部カードをランダムに抜き出して「対戦中に見えているカード」とみなし、
元の60枚デッキ全体を教師ラベルにします。

主な学習オプション:

- `--epochs N`: 学習epoch数。
- `--batch-size N`: batch size。
- `--hidden-size N`: MLPの隠れ層サイズ。
- `--layers N`: MLPの隠れ層数。
- `--samples-per-deck N`: 1つのデッキから作る部分観測サンプル数。
- `--min-observed N`: 観測済みカード枚数の最小値。
- `--max-observed N`: 観測済みカード枚数の最大値。
- `--max-decks N`: `deck_candidates.jsonl` から学習に使うデッキ数の上限。
- `--device cpu`: CPUで実行する。

軽量な動作確認:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --epochs 1 --batch-size 64 --hidden-size 64 --layers 1 --samples-per-deck 1 --output tools\deck_generator\generated\debug_deck_mlp.pt --device cpu
```

デッキ数を絞って学習する場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py train --max-decks 50000
```

## 4. MLPでデッキを推論する

学習済みモデルを使って、見えているカードから60枚デッキ候補を生成します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py predict --observed 431,1186,1 --json
```

checkpointを明示する場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\train_deck_mlp.py predict --checkpoint tools\deck_generator\generated\deck_mlp.pt --observed 431,1186,1 --json
```

推論時は、既に見えているカード枚数を必ず満たしたうえで、
予測されたカードcountをもとに60枚へ丸めます。
その際、同名4枚制限、基本エネルギー例外、ACE SPEC 1枚制限を考慮します。

## 5. 同一デッキを勝利数で集約する

`deck_candidates.jsonl` の同一デッキをまとめ、元episode JSONの `rewards` から
勝利数、敗北数、引き分け数、試合数、勝率を集計します。

```powershell
.\.venv\Scripts\python.exe tools\deck_generator\deck_completion.py aggregate-wins
```

デフォルトの出力先:

```text
tools/deck_generator/generated/deck_candidates_by_wins.jsonl
```

出力は勝利数が多い順に並びます。
