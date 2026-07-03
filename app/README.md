# PokeTCG Battle Table

`src/main.py` のエージェントとブラウザ上で対戦するローカルUIです。

```powershell
.\.venv\Scripts\python.exe app\server.py
```

起動後、ブラウザで `http://127.0.0.1:8000` を開きます。

- プレイヤー側のデッキ: `app/deck.csv`
- AI側のデッキ: `src/deck.csv`
- カード画像: `docs/cards/`

## エージェント対戦の観戦

`src/main.py` と `src_sec/main.py` の対戦を1行動ずつ観戦できます。

```text
http://127.0.0.1:8000/watch
```

「次の1行動」で1回だけ進み、「連続再生」で自動進行します。Player 1は
`src/`、Player 2は`src_sec/`の実装とデッキを使用します。
