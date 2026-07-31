# デッキクラスタリング・主要デッキ生成

このフォルダには、対戦データから集計された実在の60枚デッキを似た構築ごとに分類し、
クラスタリング結果の評価、デンドログラムの作成、各クラスタの主要デッキ選出まで行う
プログラムがあります。

このREADMEでは、初めてこの処理を見る人が次の内容を理解して実行できることを目的とします。

- 何を入力し、何を出力するのか
- デッキ同士の近さをどのように計算するのか
- 階層クラスタをどの位置で分割するのか
- クラスタの代表デッキと競技向けデッキをどう選ぶのか
- コマンドをどのように実行・調整するのか

## 1. このフォルダで行うこと

処理の全体像は次のとおりです。

```text
実在60枚デッキのJSONL
    ↓
カード採用枚数によるデッキ間距離を計算
    ↓
平均連結法によるボトムアップ階層クラスタリング
    ↓
指定クラスタ数で階層を切断
    ↓
クラスタリング精度を定量・定性評価
    ↓
要約デンドログラムをSVG・PNGで作成
    ↓
各クラスタの代表デッキ・競技向けデッキを選出
```

各クラスタから選ぶデッキは、カード採用率を丸めて合成した架空のデッキではありません。
入力データ内で実際に観測された60枚デッキをそのまま出力します。そのため、出力CSVは
対戦ツールのデッキとして直接利用できます。

## 2. ファイル構成

|ファイル|役割|
|---|---|
|`run_hierarchical_analysis.py`|クラスタリング、評価、図、主要デッキ生成を一括実行する|
|`hierarchical_clustering.py`|平均連結法による階層クラスタリングを行う|
|`evaluate_hierarchical_clustering.py`|クラスタ内外類似度などを評価し、日本語レポートを作る|
|`render_hierarchical_dendrogram.py`|切断後のクラスタを葉とした要約デンドログラムを作る|
|`select_cluster_major_decks.py`|代表デッキと競技向けデッキを選び、60枚CSVを作る|
|`tests/test_hierarchical_clustering.py`|距離、クラスタリング、評価、主要デッキ選出を検証する|

## 3. 入力データ

既定では、`deck_generator` が対戦データから集計した次のJSONLを読みます。

```text
tools/deck_generator/generated/deck_candidates_by_wins.jsonl
```

1行が1種類のユニークな60枚デッキです。少なくとも次の情報を使用します。

|フィールド|用途|
|---|---|
|`deck_counts`|カードID別の採用枚数。合計60枚である必要がある|
|`games`|その60枚構築が観測された対戦数|
|`wins` / `losses` / `draws`|勝率とクラスタ統計の計算|

カード名、カード種別、ポケモンタイプの表示には次のカードマスターを使用します。

```text
data/JP_Card_Data.csv
```

## 4. クラスタリング方法

### 4.1 デッキの表現

各デッキを「カードIDごとの採用枚数」のヒストグラムとして扱います。カードの並び順は
使用せず、同じカードを何枚採用しているかだけを比較します。

### 4.2 デッキ類似度と距離

デッキ `i` とデッキ `j` の類似度は、両方に共通して含まれるカード枚数を60で割った
ヒストグラム交差類似度です。

```text
similarity(i, j) = Σ_card min(count_i(card), count_j(card)) / 60
distance(i, j)   = 1 - similarity(i, j)
```

例:

- 完全に同じ60枚: 類似度 `1.00`、距離 `0.00`
- 45枚が共通: 類似度 `0.75`、距離 `0.25`
- 共通カードがない: 類似度 `0.00`、距離 `1.00`

カード名だけではなくカードIDを比較するため、同名でも別カードIDなら別カードとして
扱います。また、ポケモン・トレーナーズ・エネルギーの違いによる個別の重み付けは
行っていません。

### 4.3 平均連結法による階層クラスタリング

手法は、凝集型階層クラスタリングの平均連結法（UPGMA）です。これは
ボトムアップ方式です。

1. 最初は1デッキを1クラスタとする
2. 距離が最も近い2クラスタを統合する
3. 新しいクラスタと他クラスタの距離を、所属デッキ間距離の平均として更新する
4. 指定したクラスタ数になるまで2～3を繰り返す

クラスタ `A` と `B` の距離は次の平均距離です。

```text
distance(A, B) = AとBに属する全デッキ対のdistanceの平均
```

単連結法のように一部の非常に近いデッキだけで統合せず、クラスタ全体の構成の近さを
見ることが目的です。

### 4.4 クラスタ数の決定

階層構造そのものは連続していますが、プログラムでは `--clusters` で指定した数に
なった時点で切断します。既定値は16です。32クラスタにする場合は明示的に
`--clusters 32` を指定します。

JSONLには次の2つの距離も保存します。

- `hierarchical_last_merge_distance`: 指定クラスタ数になるために最後に行った統合距離
- `hierarchical_next_merge_distance`: さらに1クラスタ減らす場合の次の統合距離

両者の差やデンドログラムを確認すると、指定クラスタ数が階層上の自然な切れ目に
近いかを判断できます。

クラスタIDは階層木内部のIDではなく、切断後に対戦数が多いクラスタから `0, 1, 2, ...`
と付け直します。

## 5. クラスタリング精度の評価

クラスタリング後、次の指標をJSONと日本語Markdownへ出力します。

### 定量評価

- クラスタ内・クラスタ間のデッキ類似度分布
- 平均シルエット係数と負のシルエット率
- 最近傍デッキが同じクラスタに属する割合
- 類似度境界を使った「別クラスタの高類似ペア」と「同一クラスタの低類似ペア」の数

平均シルエット係数は1に近いほどクラスタ内が近く、別クラスタから離れていることを
示します。0付近は境界が曖昧、負値は別クラスタの方が近い可能性を示します。

### 定性評価の補助指標

- クラスタ内・クラスタ間のポケモン構成Jaccard類似度
- 主要ポケモンタイプの純度
- カード種別の構成比
- ポケモンタイプの構成比
- エネルギータイプの構成比
- 各クラスタの主要ポケモン

これらは「ポケモンカードのデッキタイプとして意味のあるまとまりか」を人が確認する
ための補助情報です。勝率が高いこと自体は、クラスタリング精度の高さを意味しません。

デンドログラムは全ユニークデッキを葉にするのではなく、指定数で切断した各クラスタを
1つの葉として、その上位の統合関係を表示します。

## 6. 主要デッキの決定方法

各クラスタから、目的の異なる2種類のデッキを選びます。

### 6.1 代表デッキ

代表デッキは「そのクラスタでよく使われた構築群に平均的に最も近い実在デッキ」です。

候補デッキ `i` の中心性を、クラスタ内の各デッキ `j` の対戦数で重み付けして計算します。

```text
centrality(i) = Σ_j games(j) × similarity(i, j) / Σ_j games(j)
```

中心性の最大値との差がデフォルト `0.01` 以内の候補を残し、次の順で1件を選びます。

1. 使用試合数が多い
2. 補正勝率が高い
3. 中心性が高い
4. カード構成による固定順序

最大中心性だけに固定せず、ごく小さな中心性差なら十分な使用実績がある構築を代表に
する設計です。

### 6.2 競技向けデッキ

競技向けデッキは「クラスタの典型性を大きく外さない範囲で、成績がよい実在デッキ」です。

1. 中心性の最大値との差がデフォルト `0.03` 以内の候補を残す
2. デフォルト20試合以上使われた候補だけを残す
3. 補正勝率、使用試合数、中心性の順で1件を選ぶ

20試合以上の候補が存在しない場合は、中心性条件を満たす候補全体へフォールバックします。

少数試合の偶然の高勝率を過大評価しないよう、単純勝率ではなくクラスタ平均へ縮約した
補正勝率を使います。

```text
adjusted_win_rate(i)
    = (wins(i) + m × cluster_win_rate) / (games(i) + m)
```

`m` は事前試合数で、既定値は20です。試合数が少ない構築ほどクラスタ平均勝率へ近づき、
試合数が多い構築ほど実際の勝率に近づきます。

### 6.3 主要クラスタの判定

全対戦数に占めるクラスタの対戦シェアがデフォルト `0.5%` 以上なら、
`hierarchical_is_major_cluster=true` とします。

これは運用上「主要」として扱うための目印です。小規模クラスタを削除する条件ではなく、
すべてのクラスタについて代表・競技向けCSVを出力します。

## 7. 操作方法

### 7.1 準備

リポジトリ直下（`pokemon-tcg-agent/`）でプロジェクトの依存関係をインストールします。
デンドログラム生成には `matplotlib` が必要で、`requirements.txt` に含まれています。

```bash
cd /path/to/pokemon-tcg-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShellでは仮想環境を次のように有効化します。

```powershell
.\.venv\Scripts\Activate.ps1
```

実行前に、既定の入力JSONLが存在することを確認してください。

```text
tools/deck_generator/generated/deck_candidates_by_wins.jsonl
```

### 7.2 推奨: 全処理を一括実行する

32クラスタで、クラスタリング・評価・デンドログラム・主要デッキ生成を実行します。

```bash
python tools/clustering_deck/run_hierarchical_analysis.py --clusters 32
```

注意: `--output` を省略すると、既定の入力JSONLへ `hierarchical_*` フィールドを
追加して安全に置き換えます。元のJSONLを変更したくない場合は、出力先を指定します。

```bash
python tools/clustering_deck/run_hierarchical_analysis.py \
  --clusters 32 \
  --output tools/clustering_deck/generated/deck_candidates_hierarchical_32.jsonl
```

デンドログラムが不要な場合:

```bash
python tools/clustering_deck/run_hierarchical_analysis.py \
  --clusters 32 \
  --no-dendrogram
```

### 7.3 処理を個別に実行する

クラスタリングだけを実行:

```bash
python tools/clustering_deck/hierarchical_clustering.py \
  --clusters 32 \
  --output tools/clustering_deck/generated/deck_candidates_hierarchical_32.jsonl
```

クラスタリング済みJSONLを評価:

```bash
python tools/clustering_deck/evaluate_hierarchical_clustering.py \
  --input tools/clustering_deck/generated/deck_candidates_hierarchical_32.jsonl
```

クラスタリング済みJSONLから主要デッキだけを生成:

```bash
python tools/clustering_deck/select_cluster_major_decks.py \
  --input tools/clustering_deck/generated/deck_candidates_hierarchical_32.jsonl \
  --output-jsonl tools/clustering_deck/generated/deck_candidates_hierarchical_32.jsonl
```

## 8. 主な調整パラメータ

一括実行コマンドで使用できる主要な引数です。

|引数|既定値|意味|
|---|---:|---|
|`--clusters`|16|階層を切断した後のクラスタ数|
|`--input`|`deck_generator/generated/...jsonl`|入力デッキJSONL|
|`--output`|入力と同じ|クラスタ・選出フィールドを追加するJSONL|
|`--boundary-similarity`|0.75|評価時に高類似・低類似を分ける確認用境界|
|`--representative-tolerance`|0.01|代表候補に許容する最大中心性との差|
|`--competitive-tolerance`|0.03|競技向け候補に許容する最大中心性との差|
|`--win-rate-prior-games`|20|補正勝率に加えるクラスタ平均の事前試合数|
|`--minimum-competitive-games`|20|競技向け候補に要求する最低試合数|
|`--minimum-major-game-share`|0.005|主要クラスタと判定する最低対戦シェア|
|`--no-dendrogram`|無効|指定するとSVG・PNGを生成しない|

すべての引数は次のコマンドで確認できます。

```bash
python tools/clustering_deck/run_hierarchical_analysis.py --help
```

## 9. 出力と確認方法

32クラスタで実行した場合、既定では次の成果物を生成します。

```text
tools/clustering_deck/generated/
├── hierarchical_cluster_evaluation_32.json
├── hierarchical_cluster_evaluation_32.md
├── hierarchical_dendrogram_32.svg
├── hierarchical_dendrogram_32.png
└── cluster_major_decks_32/
    ├── cluster_00_representative.csv
    ├── cluster_00_competitive.csv
    ├── ...
    ├── manifest.csv
    ├── selection_summary.json
    └── README.md
```

|成果物|確認できる内容|
|---|---|
|`hierarchical_cluster_evaluation_32.md`|人が読むための精度評価とクラスタ別カード構成|
|`hierarchical_cluster_evaluation_32.json`|同じ評価結果の機械可読データ|
|`hierarchical_dendrogram_32.svg/png`|32クラスタより上位の統合関係|
|`cluster_major_decks_32/manifest.csv`|全クラスタの主要判定と選出スコア一覧|
|`selection_summary.json`|選出パラメータと結果の機械可読データ|
|`cluster_XX_representative.csv`|クラスタの典型性を重視した実在60枚デッキ|
|`cluster_XX_competitive.csv`|典型性の範囲内で補正勝率を重視した実在60枚デッキ|

各デッキCSVはヘッダーなし、カードID 1列、60行です。

### JSONLへ追加される主なフィールド

|フィールド|意味|
|---|---|
|`hierarchical_cluster_id`|対戦数順に付け直したクラスタID|
|`hierarchical_cluster_members`|クラスタ内のユニークデッキ数|
|`hierarchical_cluster_games`|クラスタ内の合計対戦数|
|`hierarchical_cluster_win_rate`|クラスタ全体の勝率|
|`hierarchical_cluster_centrality`|その構築の対戦数加重平均類似度|
|`hierarchical_adjusted_win_rate`|クラスタ平均へ縮約した補正勝率|
|`hierarchical_cluster_representative`|代表デッキとして選ばれたか|
|`hierarchical_cluster_competitive`|競技向けデッキとして選ばれたか|
|`hierarchical_is_major_cluster`|クラスタの対戦シェアが主要判定閾値以上か|
|`hierarchical_last_merge_distance`|指定クラスタ数になる直前の統合距離|
|`hierarchical_next_merge_distance`|さらに1クラスタ減らす場合の統合距離|

## 10. 注意点

- クラスタリング対象はユニークな60枚構築です。`games` は主要デッキの中心性や
  クラスタ統計には使いますが、平均連結法の統合距離自体には重み付けしません。
- カードの役割、進化関係、特性、相性などを直接特徴量にはしていません。定性評価で
  ポケモンタイプや主要カードを後から確認します。
- 競技向けデッキの「競技向け」は観測データ上の補正勝率による選出です。未知環境での
  強さや統計的有意差を保証するものではありません。
- クラスタ数を変えるとクラスタIDも対戦数順に振り直されるため、異なる実行間で同じIDが
  同じアーキタイプを表すとは限りません。
- 入力に60枚でないデッキや `hierarchical_cluster_id` のない評価・選出入力がある場合は、
  エラーとして停止します。

## 11. テスト

```bash
python -m unittest discover -s tools/clustering_deck/tests -v
```

テストでは、完全一致・部分一致・不一致デッキの距離、平均連結法の統合、JSONL項目、
クラスタ評価、代表・競技向けデッキの選出、出力CSVが60行になることを確認します。
