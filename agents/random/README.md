# random agent

## 概要

`random` は、cabtが提示する合法手からランダムに選択するベースラインagentです。

## 提出対象

```text
agents/random/src/
```

`main.py` の `agent(obs_dict)` がKaggle/cabtから呼び出されます。初回呼び出しでは
`deck.csv` を読み込み、以降は `obs.select.option` のindexをランダムに返します。

## 学習

学習処理はありません。

## ローカル対戦

```bash
python tools/run_local_match.py --agent random
```

## 提出ファイル作成

```bash
python tools/build_submission.py --agent random
```
