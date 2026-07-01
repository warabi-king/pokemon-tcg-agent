# pokemon-tcg-agent

Pokémon TCG AI Battle Challenge Simulation向けのAIエージェント開発用リポジトリです。

Kaggle Dataの `sample_submission/` を参照用に残しつつ、実際に開発・提出するコードは `src/` に集約しています。

## ディレクトリ構成

```text
.
├── src/                 # 開発・提出対象
│   ├── main.py          # Kaggleが呼び出すagent実装
│   ├── deck.csv         # 提出に使う60枚デッキ
│   └── cg/              # cabt SDK本体
├── tools/               # ローカル開発用ツール。提出物には含めない
│   ├── run_local_match.py
│   ├── build_submission.py
│   └── inspect_cards.py
├── sample_submission/   # Kaggle配布sample。基本的に編集しない
├── data/                # Kaggle DataのカードCSV
├── docs/                # Kaggle DataのPDF資料
├── requirements.txt
└── README.md
```

提出物は `src/` 配下の `main.py`、`deck.csv`、`cg/` から作成します。

## セットアップ

Python 3.11で動作確認しています。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` では `kaggle-environments==1.30.2` を使っています。
Competition Overviewには `1.14.10` の記載がありますが、PyPIではそのバージョンが公開されていないため、cabt環境が含まれる公開版を使っています。

## 開発対象

通常は [src/main.py](src/main.py) を編集します。

`agent(obs_dict)` がKaggle/cabtから呼ばれるエントリポイントです。

```python
def agent(obs_dict: dict) -> list[int]:
    ...
```

初回呼び出しでは `obs.select` が `None` になり、このときは60枚デッキを返します。
以降は `obs.select.option` に含まれる合法手のindexを返します。

## デッキ

提出用デッキは [src/deck.csv](src/deck.csv) です。

形式はカードIDを1行1枚で60行です。

```text
721
721
...
3
3
```

カードIDは [data/EN_Card_Data.csv](data/EN_Card_Data.csv) または [data/JP_Card_Data.csv](data/JP_Card_Data.csv) で確認できます。

カード一覧をターミナルで確認する場合:

```bash
python tools/inspect_cards.py
```

## ローカル対戦

`src/main.py` のエージェント同士で1試合実行します。

```bash
python tools/run_local_match.py
```

出力先:

```text
results/result.html
results/result_kaggle.html
```

`results/result.html` はローカル確認用の簡易ビューアです。ステップごとのJSON、最終status、最終rewardを確認できます。

`results/result_kaggle.html` は `kaggle-environments` 標準のHTMLレンダーです。ただしcabtではrenderer未設定のため、ブラウザで空表示になる場合があります。

`results/` はGit追跡対象外です。

## 提出ファイル作成

```bash
python tools/build_submission.py
```

作成されるファイル:

```text
submission.tar.gz
```

アーカイブ直下には以下が入ります。

```text
main.py
deck.csv
cg/
```

Kaggle提出では `main.py` がアーカイブ直下にある必要があります。`src/` ディレクトリごと入れないようにしてください。

中身を確認する場合:

```bash
tar -tzf submission.tar.gz | head -30
```

## Git管理方針

作業ルールの詳細は [AGENTS.md](AGENTS.md) も参照してください。

Gitに含めるもの:

- `src/`
- `tools/`
- `sample_submission/`
- `data/`
- `docs/`
- `requirements.txt`
- `README.md`
- `AGENTS.md`

Gitに含めないもの:

- `.venv/`
- `results/`
- `submission.tar.gz`
- `__pycache__/`
- `.DS_Store`

## 注意

- `sample_submission/` はKaggle配布物の参照用です。通常は編集しません。
- 実装変更は `src/main.py` に集約します。
- `src/cg/` はcabt SDK本体です。SDK更新が必要な場合以外は変更しません。
- Kaggle評価中は外部通信できないため、エージェントは提出アーカイブ内のファイルだけで動く必要があります。
