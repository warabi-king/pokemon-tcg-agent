# CBOWによるデッキ類似度とクラスタリング

`cbow_clustering.py` は、`train_deck_word2vec.py` が学習したCBOWカード埋め込みを使い、
60枚デッキの類似度計算とクラスタリングを行います。既存のカードID一致率だけを使う方式と異なり、
一緒に採用されるカードから学習した「役割の近さ」を反映します。

## 類似度

デッキ `d` におけるカード `c` の採用枚数を `n(d,c)`、そのカードを採用するデッキ数から
求める逆文書頻度を `idf(c)`、CBOWカードベクトルを `e(c)` とします。

意味ベクトルは次のIDF付き加重平均です。CBOW空間に共通する方向を除去し、カードベクトルと
デッキベクトルはL2正規化します。

```text
semantic(d) = normalize(Σ_c n(d,c) * idf(c) * normalize(center(e(c))))
```

同じカードの採用枚数も失わないよう、カードIDごとのTF-IDF構成ベクトルを併用します。

```text
composition(d,c) = normalize((1 + log(n(d,c))) * idf(c))  if n(d,c) > 0
```

最終類似度は、既定では意味を70%、構成を30%としたcosine類似度です。

```text
similarity(a,b)
  = 0.7 * cosine(semantic(a), semantic(b))
  + 0.3 * cosine(composition(a), composition(b))
```

`--semantic-weight` で比率を変更できます。1に近づけると役割の近いカード置換を同一視しやすく、
0に近づけるとカードIDと枚数の一致を重視します。

## クラスタリング

重み付きの2成分を連結した単位ベクトルにspherical k-meansを適用します。
`--clusters auto` では、候補となるクラスタ数ごとにcosine silhouette係数を計算し、最大のものを
採用します。乱数seed、k-means++初期化、複数回初期化により再現性と安定性を持たせています。

実行例:

```powershell
.\.venv\Scripts\python.exe tools\clustering_deck\cbow_clustering.py
```

クラスタ数を固定する場合:

```powershell
.\.venv\Scripts\python.exe tools\clustering_deck\cbow_clustering.py --clusters 16
```

既定の入力・モデル・出力は次のとおりです。

```text
入力:     tools/deck_generator_dbrecon/deck_candidates_by_wins.jsonl
CBOW:     tools/deck_generator/generated/deck_word2vec.pt
出力:     tools/clustering_deck/generated/deck_candidates_cbow_clustered.jsonl
レポート: tools/clustering_deck/generated/cbow_cluster_summary.json
概要MD:   tools/clustering_deck/generated/cbow_cluster_summary.md
```

出力JSONLには、元のフィールドを保持したまま `cbow_cluster_id`、クラスタ規模・戦績、
中心類似度、代表デッキフラグなどを追加します。レポートにはクラスタ数候補のsilhouette、
クラスタ別の件数・戦績・代表デッキ・頻出カードを保存します。Markdown概要には各クラスタの
勝率、主な構成カードの採用率、代表デッキの全カード構成を掲載します。

## テスト

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools\clustering_deck\tests -v
```
