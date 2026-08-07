"""selfplay時だけ、worker-batched MCTSのroot子ノード事前確率へDirichletノイズを混ぜるpatch。

AlphaZero系の実装にある「ルート探索ノイズ」がこのリポジトリには存在しなかった。
このpatchは ``tools/batched_tournament.py`` の ``_commit_evaluation_rows`` を書き換え、
root nodeの事前確率だけへ ``(1-eps)*p + eps*Dirichlet(alpha)`` を適用する。

「root node」は探索済みのMCTS rootだけでなく、``_prepare_policy_only_root``が作る
探索なしのroot（セットアップ中、相手Activeが裏向きの間の1-step policy判断）も含む
（どちらも ``request.node.parent is None`` で判定されるため）。後者はchildrenが
全てvisit=0なので、このノイズは実質「ノイズ入りNN priorからの直接sampling」になる。

学習側 ``tools/train/train_imitation.py`` はMCTSのvisit分布ではなく、
selfplayで実際に選ばれた一手（chosen_index）だけを正解クラスとする分類学習
（cross-entropy）を行う。そのため、このrootノイズがなければ「ネットワークが
一番良いと思う一手」がほぼ毎回同じ状況で選ばれ続け、学習データ側の
(局面, 選択action)の分布が早期に偏る（mode collapse）。rootノイズはこの分布の
多様性を保つのが主目的で、AlphaZeroのように探索精度そのものを上げるためではない。

``tools/batched_tournament.py`` はworkerサブプロセスがdiskから直接importするため、
温度patch（``_finish_search``）と同じ方式でファイル自体を書き換える。
既に適用済みなら何もしない（冪等）。
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT_NOISE_PATCH_MARKER = "SELFPLAY_ROOT_DIRICHLET_NOISE_PATCH_V1"

_ANCHOR = '''def _commit_evaluation_rows(
    chunk: list[_EvalRequest],
    value_rows: list[list[float]],
    policy_rows: list[list[float]],
) -> None:
    """device評価結果をCPU側のMCTS nodeへ反映する。"""
    for request, value_row, policy_row in zip(
        chunk, value_rows, policy_rows, strict=True
    ):
        value = float(value_row[0])
        if request.node.parent is None:
            request.context.root_sample = BatchedLearnSample(
                value=value,
                policy=[
                    float(policy_row[index])
                    for index in range(len(request.actions))
                ],
                sv_enc=_snapshot_sparse(request.encoder),
                # decoderはforward前にbucket幅までpaddingされているため、
                # 実際の合法手数へ戻して保存する。
                sv_dec=_snapshot_sparse(
                    request.decoder,
                    offset_count=len(request.actions),
                ),
            )
        propagated = value
        if request.node.player_index != request.context.your_index:
            propagated = -propagated
        request.node.value = propagated
        request.node.backprop(propagated)

        probabilities = [
            math.exp(float(policy_row[index]) * 10.0)
            for index in range(len(request.actions))
        ]
        probability_sum = sum(probabilities)
        for action, probability in zip(
            request.actions, probabilities, strict=True
        ):
            if probability_sum > 0:
                probability /= probability_sum
            request.node.children.append(_Child(action, probability))
        if request.reserved_child is not None:
            request.reserved_child.in_flight = False'''

_PATCHED = f'''# {ROOT_NOISE_PATCH_MARKER}
def _sample_dirichlet_noise(count: int, alpha: float) -> list[float]:
    """対称Dirichlet(alpha)分布からcount個の重みをsampleする（合計1.0）。"""
    if count <= 0:
        return []
    samples = [random.gammavariate(alpha, 1.0) for _ in range(count)]
    total = sum(samples)
    if total <= 0.0:
        return [1.0 / count] * count
    return [sample / total for sample in samples]


def _apply_root_dirichlet_noise(probabilities: list[float]) -> list[float]:
    """selfplay時だけ有効な、root事前確率へのDirichletノイズ混合。

    AlphaZero同様 ``(1-eps)*p + eps*noise`` で混合する。実対戦・評価対戦では
    ``SELFPLAY_ROOT_NOISE_ENABLED`` を設定しないため、この関数は入力をそのまま返す。
    """
    if os.environ.get("SELFPLAY_ROOT_NOISE_ENABLED") != "1":
        return probabilities
    if len(probabilities) <= 1:
        return probabilities
    alpha = float(os.environ.get("SELFPLAY_ROOT_NOISE_ALPHA", "0.3"))
    epsilon = float(os.environ.get("SELFPLAY_ROOT_NOISE_EPSILON", "0.25"))
    if epsilon <= 0.0:
        return probabilities
    noise = _sample_dirichlet_noise(len(probabilities), alpha)
    return [
        (1.0 - epsilon) * probability + epsilon * noise_value
        for probability, noise_value in zip(probabilities, noise, strict=True)
    ]


def _commit_evaluation_rows(
    chunk: list[_EvalRequest],
    value_rows: list[list[float]],
    policy_rows: list[list[float]],
) -> None:
    """device評価結果をCPU側のMCTS nodeへ反映する。"""
    for request, value_row, policy_row in zip(
        chunk, value_rows, policy_rows, strict=True
    ):
        value = float(value_row[0])
        if request.node.parent is None:
            request.context.root_sample = BatchedLearnSample(
                value=value,
                policy=[
                    float(policy_row[index])
                    for index in range(len(request.actions))
                ],
                sv_enc=_snapshot_sparse(request.encoder),
                # decoderはforward前にbucket幅までpaddingされているため、
                # 実際の合法手数へ戻して保存する。
                sv_dec=_snapshot_sparse(
                    request.decoder,
                    offset_count=len(request.actions),
                ),
            )
        propagated = value
        if request.node.player_index != request.context.your_index:
            propagated = -propagated
        request.node.value = propagated
        request.node.backprop(propagated)

        probabilities = [
            math.exp(float(policy_row[index]) * 10.0)
            for index in range(len(request.actions))
        ]
        probability_sum = sum(probabilities)
        normalized_probabilities = [
            probability / probability_sum if probability_sum > 0 else probability
            for probability in probabilities
        ]
        if request.node.parent is None:
            normalized_probabilities = _apply_root_dirichlet_noise(
                normalized_probabilities
            )
        for action, probability in zip(
            request.actions, normalized_probabilities, strict=True
        ):
            request.node.children.append(_Child(action, probability))
        if request.reserved_child is not None:
            request.reserved_child.in_flight = False'''


def ensure_root_noise_patch(batched_tournament_path: Path, match_root: Path) -> None:
    """``tools/batched_tournament.py`` にルートDirichletノイズpatchを適用する。

    既に適用済みなら何もしない。期待した元実装を一意に確認できない場合は、
    推測で変更せずエラーで停止する。
    """
    source = batched_tournament_path.read_text(encoding="utf-8")

    if ROOT_NOISE_PATCH_MARKER in source:
        print(f"root noise patch already present: {batched_tournament_path}")
        return

    occurrences = source.count(_ANCHOR)
    if occurrences != 1:
        raise RuntimeError(
            "_commit_evaluation_rows() の期待した元実装を一意に確認できません。"
            f" occurrences={occurrences}. 推測で変更せず停止します。"
        )

    backup_path = (
        match_root / "results" / "_generated" / "batched_tournament_before_root_noise.py"
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if not backup_path.exists():
        shutil.copy2(batched_tournament_path, backup_path)

    patched = source.replace(_ANCHOR, _PATCHED, 1)
    compile(patched, str(batched_tournament_path), "exec")

    temporary = batched_tournament_path.with_name(
        f".{batched_tournament_path.name}.root_noise.tmp"
    )
    try:
        temporary.write_text(patched, encoding="utf-8")
        temporary.replace(batched_tournament_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"root noise patch applied: {batched_tournament_path}")
    print(f"backup: {backup_path}")
