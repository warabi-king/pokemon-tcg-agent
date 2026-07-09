# デッキ変更戦略

## 概要

`agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py` は、デッキ内のカードを変更して新しいデッキを生成する。

変更戦略は `--strategy` 引数で指定する。現在実装している戦略は以下の2種類である。

- `random`
- `genetic`

生成したデッキは、指定した出力先へ `deck_N.csv` という名前で保存する。`N` には未使用の最小番号を使用する。

## 基本的な使い方

以降のコマンドは、リポジトリルート `feat-mini-chenge-deck/` で実行する。

ヘルプを表示する:

```bash
python3 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py --help
```

第1引数の `m` には、1回の変更で入れ替えるカード枚数を指定する。指定可能な範囲は1枚から60枚までである。

```bash
python3 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py <m> \
  --strategy <randomまたはgenetic>
```

変更元と出力先を省略した場合は、以下のパスを使用する。

```text
変更元:
  agents/rl_mcts_mini_chenge/src/deck.csv

出力先:
  agents/rl_mcts_mini_chenge/train/deck_N.csv
```

任意の変更元と出力先を指定する場合:

```bash
python3 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py 3 \
  --strategy random \
  --input path/to/source_deck.csv \
  --output-dir path/to/output
```

## 共通のデッキ構築ルール

どちらの戦略でも、生成したデッキが以下の条件を満たすまで候補を再生成する。

- デッキは60枚で構成する。
- `data/EN_Card_Data.csv` に存在するカードIDだけを使用する。
- 基本エネルギーを除き、同名カードは4枚までとする。
- ACE SPECは1枚までとする。
- たねポケモンを1枚以上含める。
- 変更対象に選ばれた位置は、変更前と異なるカードIDにする。

有効なデッキを10,000回以内に生成できなかった場合はエラーにする。

## Random戦略

### 内容

変更元デッキから重複しない `m` 個の位置を選び、それぞれを全カード候補からランダムに選んだ別カードへ変更する。

カードの分類、進化ライン、エネルギータイプ、カード間の相性は考慮しない。

### 実行例

```bash
python3 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py 3 \
  --strategy random
```

この例では、変更元デッキの3箇所を変更したデッキを1つ生成する。

乱数を固定する場合:

```bash
python3 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py 3 \
  --strategy random \
  --seed 42
```

### 特徴

- 学習済みモデルやPyTorchを使用しない。
- 対戦による評価を行わない。
- 常に変更済みのデッキを1つ保存する。
- 変更箇所の数は指定した `m` と一致する。

## Genetic戦略

### 内容

交叉を使用せず、エリート選択とランダム変異を繰り返す簡易的な `(μ+λ)` 進化戦略である。

処理手順:

1. 変更元デッキを初期個体として用意する。
2. 変更元デッキから `m` 枚を変更し、指定した個体数まで初期集団を生成する。
3. 各個体を変更元デッキと対戦させる。
4. 勝利を1点、引き分けを0.5点としてfitnessを計算する。
5. fitness上位の個体をエリートとして選択する。
6. エリートから親をランダムに選び、親の `m` 枚を変更して次世代の子を生成する。
7. 指定した世代数まで評価と変異を繰り返す。
8. 最終世代でfitnessが最大の個体を保存する。

対戦では、候補デッキと変更元デッキの両方を同じ学習済みMCTSモデルで操作する。候補デッキは試合ごとにプレイヤー番号を交代する。

fitnessは以下で計算する。

```text
fitness = (勝利数 + 0.5 × 引き分け数) / 試合数
```

### 実行例

```bash
python3.11 agents/rl_mcts_mini_chenge/train/mini_chenge_deck.py 2 \
  --strategy genetic \
  --generations 3 \
  --population-size 8 \
  --elite-count 2 \
  --games 10 \
  --search-count 5
```

この例では、以下の条件で探索する。

- 1回の変異で2枚変更する。
- 3世代探索する。
- 各世代で8個体を評価する。
- 上位2個体をエリートとして残す。
- 各個体を10試合で評価する。
- MCTSで各手を選ぶときに5回探索する。

### 特徴

- `src/model.pth` の学習済みモデルを使用する。
- PyTorchとゲームシミュレータを使用する。
- 対戦結果を次世代の選択へ反映する。
- 各世代で変更を繰り返すため、最終デッキは変更元から `m` 枚以上変わることがある。
- 変更元デッキも初期個体に含むため、変更元が最高評価なら変更されていないデッキが保存されることがある。

## CLI引数

| 引数 | 既定値 | 内容 |
|---|---:|---|
| `m` | 必須 | 1回の変異で変更するカード枚数 |
| `--strategy` | `genetic` | `random` または `genetic` |
| `--input` | `src/deck.csv` | 変更元デッキ |
| `--output-dir` | `train/` | 生成デッキの保存先 |
| `--seed` | 指定なし | 候補生成に使用する乱数seed |
| `--generations` | `3` | Geneticの世代数 |
| `--population-size` | `8` | Geneticの各世代の個体数 |
| `--elite-count` | `2` | Geneticで次世代へ残す個体数 |
| `--games` | `10` | Geneticで各個体を評価する試合数 |
| `--search-count` | `5` | Geneticの各手におけるMCTS探索回数 |
| `--model` | `src/model.pth` | Geneticの対戦評価に使用するモデル |

`--generations`、`--population-size`、`--elite-count`、`--games`、`--search-count`、`--model` は、`--strategy genetic` の場合だけ使用する。

## 出力

出力ファイルは1行に1つのカードIDを記録した60行のCSVである。

保存先に以下のファイルが存在する場合:

```text
deck_1.csv
deck_2.csv
deck_4.csv
```

未使用の最小番号である `deck_3.csv` を生成する。

並行実行によって同じ番号が先に使用された場合は、番号を再取得して別のファイルへ保存する。

コマンドが成功すると、Random戦略では使用した戦略と保存先、Genetic戦略では最良個体の評価結果と保存先を標準出力へ表示する。

Random戦略の出力例:

```text
Strategy: random
Saved: .../train/deck_1.csv
```

Genetic戦略の出力例:

```text
Best deck: score=0.700 (W/L/D=7/3/0)
Saved: .../train/deck_2.csv
```

## 採用デッキをsrcへ反映する

最終的に採用する `deck_N.csv` が決まったら、`apply_deck()` へそのパスを渡す。関数は採用判断を行わず、指定されたデッキを検証して `agents/rl_mcts_mini_chenge/src/deck.csv` へ反映する。

例として `train/deck_3.csv` を採用する場合:

```bash
cd agents/rl_mcts_mini_chenge/train
python3 -c 'from pathlib import Path; from mini_chenge_deck import apply_deck; print(apply_deck(Path("deck_3.csv")))'
```

正常に反映されると、更新した `src/deck.csv` のパスを表示する。

`apply_deck()` は反映前に以下を確認する。

- 60枚であること
- 全カードIDが存在すること
- 同名カード、ACE SPEC、たねポケモンの構築ルールを満たすこと

検証に失敗した場合は `src/deck.csv` を更新せず、エラーにする。検証に成功した場合は、既存の `src/deck.csv` を指定した採用デッキで上書きする。
