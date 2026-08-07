"""value headをtanh飽和させずに学習するための共通処理。"""

from __future__ import annotations

import os


VALUE_TRAINING_SPACES = frozenset({"bounded", "logit"})


def value_training_space() -> str:
    """value lossを計算する空間を返す。

    ``logit`` は推論時の ``tanh`` より前で損失を計算する。これにより、予測が
    ±1付近にあるときも ``tanh`` の微分が勾配を消さない。``bounded`` は旧挙動。
    """
    space = os.environ.get("SELFPLAY_VALUE_TRAINING_SPACE", "logit")
    if space not in VALUE_TRAINING_SPACES:
        raise ValueError(
            f"SELFPLAY_VALUE_TRAINING_SPACEが不正です: {space!r} "
            f"(有効値: {sorted(VALUE_TRAINING_SPACES)})"
        )
    return space


def value_target_epsilon() -> float:
    """±1の教師を有限logitへ写すための端点epsilon。"""
    epsilon = float(os.environ.get("SELFPLAY_VALUE_TARGET_EPSILON", "1e-3"))
    if not 0.0 < epsilon < 1.0:
        raise ValueError(
            "SELFPLAY_VALUE_TARGET_EPSILONは0より大きく1未満にしてください: "
            f"{epsilon}"
        )
    return epsilon


def value_targets_for_loss(targets, *, space: str | None = None):
    """回帰loss向けに[-1, 1]のvalue教師を学習空間へ変換する。"""
    import torch

    resolved_space = value_training_space() if space is None else space
    if resolved_space == "bounded":
        return targets
    if resolved_space != "logit":
        raise ValueError(f"未対応のvalue学習空間です: {resolved_space!r}")
    epsilon = value_target_epsilon()
    return torch.atanh(targets.clamp(-1.0 + epsilon, 1.0 - epsilon))


def build_value_loss(torch_module, *, kind: str = "bce", huber_delta: float = 0.2):
    """推論値 ``tanh(z)`` と整合するvalue lossを作る。

    ``bce`` では ``tanh(z) = 2 * sigmoid(2z) - 1`` を利用する。勝敗教師を
    ``(value + 1) / 2`` へ写したBCEは勝率を学ぶproper lossであり、二値教師に
    Huberを使ったときの「条件付き中央値が±1へ振り切れる」問題を避けられる。
    """
    if kind == "bce":
        if value_training_space() != "logit":
            raise ValueError(
                "value loss=bceにはSELFPLAY_VALUE_TRAINING_SPACE=logitが必要です。"
            )

        def binary_value_loss(logits, bounded_targets):
            probability_targets = (bounded_targets.clamp(-1.0, 1.0) + 1.0) * 0.5
            return torch_module.nn.functional.binary_cross_entropy_with_logits(
                logits * 2.0,
                probability_targets,
            )

        return binary_value_loss

    if kind == "mse":
        base_loss = torch_module.nn.MSELoss()
    elif kind == "huber":
        base_loss = torch_module.nn.HuberLoss(delta=huber_delta)
    else:
        raise ValueError(f"未対応のvalue lossです: {kind!r}")

    def regression_value_loss(predictions, bounded_targets):
        return base_loss(
            predictions,
            value_targets_for_loss(bounded_targets),
        )

    return regression_value_loss


def forward_for_training(model, *inputs, space: str | None = None):
    """推論互換のpolicyと、value loss用のvalueを返す。

    モデルの通常forwardはvalueとpolicyの両方へtanhを適用する。valueだけは
    encoderの計算グラフからraw logitを直接取り出し、logit学習時の勾配がtanhを
    通らないようにする。policyは既存の学習・MCTS挙動を保つため変更しない。
    """
    import torch

    resolved_space = value_training_space() if space is None else space
    (
        index_encoder,
        value_encoder,
        offset_encoder,
        index_decoder,
        value_decoder,
        offset_decoder,
    ) = inputs

    hidden = model.encoder_bag(index_encoder, offset_encoder, value_encoder)
    hidden = hidden.reshape(
        -1,
        model.num_words_encoder,
        model.d_model,
    ).transpose(0, 1)
    batch_size = hidden.size(1)
    encoder_out = model.encoder(hidden)
    raw_value = model.encoder_fc(encoder_out).mean(0)

    policy = model.decoder_bag(index_decoder, offset_decoder, value_decoder)
    policy = policy.reshape(batch_size, -1, model.d_model).transpose(0, 1)
    for layer in model.decoder:
        policy = layer(policy, encoder_out)
    policy = model.decoder_fc(policy)
    policy = policy.transpose(0, 1).reshape(batch_size, -1)
    bounded_policy = torch.tanh(policy)

    if resolved_space == "logit":
        return raw_value, bounded_policy
    if resolved_space == "bounded":
        return torch.tanh(raw_value), bounded_policy
    raise ValueError(f"未対応のvalue学習空間です: {resolved_space!r}")
