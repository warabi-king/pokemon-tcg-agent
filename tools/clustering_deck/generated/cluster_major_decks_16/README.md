# 階層クラスタ主要デッキ

各CSVは観測済みの実在60枚デッキで、カードIDを1行1枚で記載しています。

## 選出方法

- 代表デッキ: 対戦数加重のクラスタ内平均類似度（中心性）が最大付近の候補から、使用試合数を優先して選出。
- 競技向けデッキ: 最大中心性を大きく外れない候補から、クラスタ平均へ縮約した補正勝率を優先して選出。
- 代表中心性許容差: 0.010
- 競技向け中心性許容差: 0.030
- 勝率事前試合数: 20.0
- 競技向け最低試合数: 20
- 主要クラスタ判定の最低対戦シェア: 0.50%

## クラスタ別一覧

|ID|主要|デッキ数|対戦数|シェア|代表中心性|代表CSV|競技向け補正勝率|競技向けCSV|同一構築|
|---:|:---:|---:|---:|---:|---:|---|---:|---|:---:|
|0|○|200|76893|33.92%|0.9164|[cluster_00_representative.csv](cluster_00_representative.csv)|54.54%|[cluster_00_competitive.csv](cluster_00_competitive.csv)|-|
|1|○|69|50819|22.42%|0.9044|[cluster_01_representative.csv](cluster_01_representative.csv)|53.54%|[cluster_01_competitive.csv](cluster_01_competitive.csv)|-|
|2|○|55|28610|12.62%|0.8580|[cluster_02_representative.csv](cluster_02_representative.csv)|50.64%|[cluster_02_competitive.csv](cluster_02_competitive.csv)|-|
|3|○|64|15646|6.90%|0.6342|[cluster_03_representative.csv](cluster_03_representative.csv)|52.50%|[cluster_03_competitive.csv](cluster_03_competitive.csv)|○|
|4|○|24|12786|5.64%|0.9852|[cluster_04_representative.csv](cluster_04_representative.csv)|54.40%|[cluster_04_competitive.csv](cluster_04_competitive.csv)|○|
|5|○|31|12296|5.42%|0.9203|[cluster_05_representative.csv](cluster_05_representative.csv)|58.82%|[cluster_05_competitive.csv](cluster_05_competitive.csv)|-|
|6|○|37|8491|3.75%|0.7068|[cluster_06_representative.csv](cluster_06_representative.csv)|53.58%|[cluster_06_competitive.csv](cluster_06_competitive.csv)|-|
|7|○|40|6841|3.02%|0.9117|[cluster_07_representative.csv](cluster_07_representative.csv)|55.42%|[cluster_07_competitive.csv](cluster_07_competitive.csv)|-|
|8|○|84|6635|2.93%|0.8872|[cluster_08_representative.csv](cluster_08_representative.csv)|50.02%|[cluster_08_competitive.csv](cluster_08_competitive.csv)|-|
|9|○|5|4336|1.91%|0.8602|[cluster_09_representative.csv](cluster_09_representative.csv)|55.94%|[cluster_09_competitive.csv](cluster_09_competitive.csv)|-|
|10|○|9|2107|0.93%|0.8999|[cluster_10_representative.csv](cluster_10_representative.csv)|56.92%|[cluster_10_competitive.csv](cluster_10_competitive.csv)|-|
|11|-|6|765|0.34%|0.9222|[cluster_11_representative.csv](cluster_11_representative.csv)|51.08%|[cluster_11_competitive.csv](cluster_11_competitive.csv)|○|
|12|-|2|313|0.14%|0.9988|[cluster_12_representative.csv](cluster_12_representative.csv)|65.91%|[cluster_12_competitive.csv](cluster_12_competitive.csv)|○|
|13|-|4|125|0.06%|0.9807|[cluster_13_representative.csv](cluster_13_representative.csv)|47.81%|[cluster_13_competitive.csv](cluster_13_competitive.csv)|○|
|14|-|1|32|0.01%|1.0000|[cluster_14_representative.csv](cluster_14_representative.csv)|31.25%|[cluster_14_competitive.csv](cluster_14_competitive.csv)|○|
|15|-|2|13|0.01%|0.9705|[cluster_15_representative.csv](cluster_15_representative.csv)|36.54%|[cluster_15_competitive.csv](cluster_15_competitive.csv)|○|

`is_major_cluster` は対戦シェアによる運用上の目印です。小規模クラスタも削除せず、比較・再現用にCSVを出力しています。
