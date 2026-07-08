# Cloud Runへのデプロイ

人間同士の対戦をGoogle Cloud Runで公開します。1つのCloud Runインスタンス内に最大5つのcabt子プロセスを起動し、最大10人が同時に対戦できます。

## 構成

- Cloud Runインスタンス: 最大1、最小0
- Cloud Runコンテナ: 1 CPU、2 GiBメモリ、同時リクエスト20
- Gunicorn: 1 worker、20 threads
- cabt: 1ルームにつき独立した子プロセス
- ルーム上限: 5
- 待機中ルーム: 15分で削除
- 無通信ルーム: 30分で削除
- 終了済みルーム: 10分で削除

ルーム情報はコンテナのメモリだけに保存します。Cloud Runがコンテナを再起動した場合、進行中の対戦は終了します。低コストを優先した構成であり、永続化は行いません。

## 初回準備

1. Google Cloudでプロジェクトを作成し、請求先アカウントを関連付ける。
2. [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)をインストールする。
3. ログインする。

```powershell
gcloud auth login
```

## デプロイ

リポジトリのルートで実行します。

```powershell
.\tools\deploy_cloud_run.ps1 -ProjectId YOUR_GCP_PROJECT_ID
```

デフォルトでは東京リージョン `asia-northeast1` に `poketcg-duel` というサービス名でデプロイします。

```powershell
.\tools\deploy_cloud_run.ps1 `
  -ProjectId YOUR_GCP_PROJECT_ID `
  -Region asia-northeast1 `
  -Service poketcg-duel
```

デプロイ完了時に表示されるHTTPS URLを開きます。Cloudflare TunnelやホストPCの常時起動は不要です。

## 費用を抑える設定

- `--min-instances 0`: 未使用時にゼロまで縮退する。
- `--max-instances 1`: 想定外のスケールアウトと課金を防ぎ、インメモリのルームを1か所に保つ。
- Artifact Registryの古いイメージを定期的に削除する。
- GCPの予算アラートを設定する。

Cloud Run、Cloud Build、Artifact Registryには無料枠がありますが、請求先アカウントの登録は必要です。利用量、リージョン、コンテナイメージ容量、通信量によって料金が発生する可能性があります。

## ローカルDocker確認

```powershell
docker build -t poketcg-duel .
docker run --rm -p 8080:8080 -e FLASK_SECRET_KEY=local-test poketcg-duel
```

ブラウザで `http://127.0.0.1:8080` を開きます。
