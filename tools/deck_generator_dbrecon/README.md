# DB reconstruction deck generator

観測できたカードと、同梱した `deck_candidates_by_wins.jsonl` の各60枚デッキを
枚数込みで照合し、一致枚数が最も多い候補を1件、そのまま補完結果として出力します。

一致枚数は、カードIDごとに `min(観測枚数, 候補内枚数)` を求め、その合計を使います。
同じ一致枚数の候補が複数ある場合は勝数だけで比較し、勝数も同じ場合はDB内の
出現順で決定します。同じ入力からは常に同じデッキが得られます。
学習や外部通信は必要ありません。

## 実行例

```powershell
.\.venv\Scripts\python.exe tools\deck_generator_dbrecon\deck_generator_dbrecon.py --observed 741,742,743
```

標準出力には、選択されたデッキのカードIDを1行1枚で60行出力します。
同じカードを複数枚見た場合は、IDをその枚数だけ指定してください。

照合情報も確認する場合:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator_dbrecon\deck_generator_dbrecon.py --observed 741,741,742,743 --json
```

ファイルから入力する場合（カンマ、空白、改行区切りに対応）:

```powershell
.\.venv\Scripts\python.exe tools\deck_generator_dbrecon\deck_generator_dbrecon.py --observed-file observed.txt
```

既定では、このディレクトリにある `deck_candidates_by_wins.jsonl` だけを参照します。
別DBを検証したい場合に限り `--database PATH` で切り替えられます。
