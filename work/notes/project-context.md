# Pokémon TCG Strategy Agent: 現在の認識

## 目的

Pokémon TCG Live Competition の Strategy Track で belief_puct を多様な相手に対して強くし、再現可能な根拠と提出物を整える。デッキ検索DBは、非公開カードの belief 推定と自分の60枚デッキ構成の最適化に使う。

## 学習の流れ

1. Phase 1: 既存対局ログで模倣学習する。完了済みモデルを `agents/belief_puct/src/model.pth` として使う。
2. Phase 2a: 方策モデルと探索設定を固定し、候補60枚だけを変える。develop_naoki の upsize1/upsize2 16model から選ぶ固定・多様リーグで候補を順位付けする。random baseline は正式リーグに含めない。
3. Phase 2b: 採用デッキを固定し、学習側モデルだけを更新する。自己対戦だけにせず、固定した強い外部相手と過去 checkpoint を混ぜる。
4. Phase 2c: 新しい60枚候補を生成して固定リーグで評価し、重複を統合した相手別成績とともに検索DBへ追加する。
5. Phase 3: holdoutリーグでモデルとデッキを固定して検証し、提出物を作る。

## 現在地と判定方針

- Phase 1 は完了し、`agents/belief_puct/src/model.pth` は模倣学習完了モデルである。
- 以前の Phase 2a は codex-auto 内の rl_mcts / random を相手にしており、正式な採用根拠に使わない。
- Phase 2a は upsize1 固定リーグでやり直す。少数ゲームで各相手の `errors: 0` を確認してから全候補を評価する。
- Phase 2b の外部 upsize 相手の混合と、Phase 2c の候補生成・DB追加は未実装である。
- timeout・未判定は `--errors-as-losses` で候補側の敗戦にする。順位は重み付き勝率、最苦手相手、Wilson下限を残す。
