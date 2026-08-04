# Goal完了監査

監査日: 2026-08-03

## 結論

Strategy Track向けの戦略立案、agent実装、学習、固定条件評価、提出アーカイブ作成、再現手順、詳細解説まで完了した。最終方式は、事前登録した候補Cから悪化要因を除いた `C′: history-consistent single-determinization PUCT` である。

Kaggleへの実アップロードだけは行っていない。これは外部提出前にユーザー承認を得るという境界によるものであり、Stage 3は「未確認」とする。

## 要件別監査

| 要件 | 判定 | 根拠 |
|---|---|---|
| 公式ルールと提出形式の把握 | 完了 | `docs/strategy/experiment-report.md`、`tools/validate_submission.py` |
| 事前登録と設計候補比較 | 完了 | `docs/strategy/preregistration.md` |
| 新agentと学習済みモデル | 完了 | `agents/belief_puct/src/` |
| 学習・resume・時系列split | 完了 | `agents/belief_puct/train/`、`agents/belief_puct/README.md` |
| Stage 1固定評価 | 完了 | 主要5相手100試合、holdout 5相手100試合 |
| Stage 2固定評価 | 完了 | 同一デッキ100試合、各自デッキ100試合 |
| Strategy Trackレポート | 完了 | `docs/strategy/strategy-report.md` |
| 詳細な戦略解説 | 完了 | `docs/strategy/strategy-explained.md` |
| 実験記録と定量結果 | 完了 | `docs/strategy/experiment-report.md`、`results/*.json` |
| Kaggle提出アーカイブ | 完了 | `dist/submission_belief_puct.tar.gz` |
| Kaggle実提出・公開対戦 | 未確認 | ユーザー承認前のため未実施 |

## 固定評価の最終値

| 評価 | 結果 | 95% Wilson区間 | 判定 |
|---|---:|---:|---|
| Stage 1a 主要5相手 | 54-46、54% | 44.26%〜63.44% | 失敗 |
| Stage 1b 未見5相手 | 66-34、66% | 56.28%〜74.54% | 成功 |
| Stage 2a 同一デッキ | 91-9、91% | 83.77%〜95.19% | 成功 |
| Stage 2b 各自デッキ | 40-60、40% | 30.94%〜49.80% | 失敗 |

失敗結果も除外せず、主要5相手への不確実性、cluster 08、相手固有デッキ、後攻テンポを次の課題として記録した。

## 提出物監査

- archive: `dist/submission_belief_puct.tar.gz`
- size: 48,594,554 bytes
- SHA-256: `28dfb60c3cfc97718ca2b9f649a7e872e1e5ca9959d87dd150826bc3667089e6`
- model SHA-256: `28877e029f52666b40dab09046b730c541b9da72d07020f8126b9efc53b95c03`
- deck SHA-256: `01b152e466925c6c5cf4c0aeaaa019da3a7f7ebd967948e3fd5e1a891df156f9`
- member: 23、直下に `main.py`、`deck.csv`、`model.pth`、`cg/`、`rl_mcts/`
- deck: 60枚
- 不要物: `train/`、log、cache、pyc、Git情報なし
- 実行検証: `/kaggle_simulations/agent/`を模した場所から3 seed、先後入替6試合、6完走、error 0
- 作業ソースとの比較: cache以外の差分なし

最終ターンではPython実体への読み取りがsandboxで拒否されたため、動的検証の再実行はできなかった。提出アーカイブ自体は変更せず、直前に成功した `results/submission_validation.json` と、作業ソースとの静的一致を再確認した。

## 環境・境界監査

- worktree参照: `refs/heads/codex-auto`
- 書込み先: `codex-auto` worktree内のみ
- 他worktree: 読取り参照のみ
- 使用量: `codex-auto`全体996 MiB、agent 686 MiB、提出source 54 MiB
- 空き容量: 276 GiB
- device: Apple M5 CPU。CUDA/MPSは利用不可
- Git状態: OS標準Gitとfallback Gitが環境制約で実行できず、worktreeのclean/dirty判定は未確認。branchはworktree管理ファイルで確認した

## 次の人間判断

Kaggleへ提出してValidation Episodeと実環境時間を確認するかを判断する。提出する場合でも、ローカルで失敗した各自デッキ比較を踏まえ、leaderboard勝率を事前に保証しない。
