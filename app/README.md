# PokeTCG Battle Table

`agents/` 配下のエージェントとブラウザ上で対戦するローカルUIです。

```powershell
.\.venv\Scripts\python.exe app\server.py
```

起動後、ブラウザで `http://127.0.0.1:8000` を開きます。

- プレイヤー側のデッキ: `app/deck.csv`
- AI側の実装とデッキ: 画面で選択した `agents/{agent-name}/src/`
- カード画像: `docs/cards/`

## エージェント対戦の観戦

`agents/` 配下から選択した2つのエージェントの対戦を1行動ずつ観戦できます。

```text
http://127.0.0.1:8000/watch
```

画面上部でPlayer 1とPlayer 2を選択できます。「次の1行動」で1回だけ進み、
「連続再生」で自動進行します。
