# PokeTCG Battle Table

`agents/` 配下のエージェントとの対戦、エージェント同士の観戦、ユーザー同士の1対1対戦をブラウザで行えます。

## Cloud Runで最大5組の対戦を公開する

GCP上で最大10人・5ルームを同時に利用する構成は [CLOUD_RUN.md](CLOUD_RUN.md) を参照してください。

```powershell
.\tools\deploy_cloud_run.ps1 -ProjectId YOUR_GCP_PROJECT_ID
```

## ユーザー同士で1対1対戦する

Node.js（`npx`）が利用できる環境で、次のコマンドを実行します。

```powershell
.\.venv\Scripts\python.exe tools\run_duel.py
```

初回はWranglerのダウンロードに時間がかかることがあります。Cloudflare Quick TunnelのURLが発行されると、ターミナルに「ホスト用URL」が表示されます。

1. ホスト用URLを自分のブラウザで開く。このURLは共有しない。
2. Player 1のデッキを選び、「ルーム作成」を押す。Player 2のデッキは参加者が参加時に選ぶ。
3. 画面に表示された招待リンクをPlayer 2へ共有する。
4. 両者がブラウザから合法手を選択して対戦する。

ホストPC、対戦サーバー、`run_duel.py`を起動したターミナルは対戦終了まで閉じないでください。URLは起動のたびに変わり、プロセスを終了すると無効になります。

「ポート8000はすでに使用中です」と表示された場合は、以前起動した `app/server.py` または `tools/run_duel.py` のターミナルで `Ctrl+C` を押して終了してから、再実行してください。

対戦には `agents/{agent-name}/src/deck.csv` をデッキプリセットとして使用します。現在は1プロセスにつき1ルームのみ利用できます。新しいルームを作成すると、それまでの対戦は終了します。

## ローカルUI

```powershell
.\.venv\Scripts\python.exe app\server.py
```

起動後に表示されるURLをブラウザで開きます。

- 人間 vs AI: `http://127.0.0.1:8000`
- エージェント同士の観戦: `http://127.0.0.1:8000/watch`
- 人間 vs 人間: ターミナルに表示されるホスト用URL

- プレイヤー側のデッキ: `app/deck.csv`
- AI側の実装とデッキ: `agents/{agent-name}/src/`
- カード画像: `docs/cards/`
