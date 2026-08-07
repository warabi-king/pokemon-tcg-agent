from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn
import torch.nn.functional

from cg.api import SelectContext, all_attack, all_card_data

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000
DECODER_MAIN_FEATURE = 8
DECODER_ATTACK_OFFSET = 14


@lru_cache(maxsize=1)
def card_count() -> int:
    """カードIDを特徴量indexに変換するための語彙サイズを返す。"""
    all_card = all_card_data()
    return max(all_card, key=lambda card: card.cardId).cardId + 1


@lru_cache(maxsize=1)
def attack_count() -> int:
    """攻撃IDを特徴量indexに変換するための語彙サイズを返す。"""
    all_attacks = all_attack()
    return max(all_attacks, key=lambda attack: attack.attackId).attackId + 1


def decoder_size(card_vocab_size: int, attack_vocab_size: int) -> int:
    """decoder側の特徴量語彙サイズを計算する。"""
    decoder_card_offset = DECODER_ATTACK_OFFSET + attack_vocab_size
    return decoder_card_offset + (
        1 + DECODER_MAIN_FEATURE + SelectContext.RECOVER_SPECIAL_CONDITION
    ) * card_vocab_size


class DecoderLayer(torch.nn.Module):
    """局面表現を参照して、候補手表現を更新する層。"""

    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)


class MyModel(torch.nn.Module):
    """局面評価(value)と候補手評価(policy)を出力するTransformer系モデル。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_feedforward: int,
        num_layers_encoder: int,
        num_layers_decoder: int,
        encoder_vocab_size: int,
        decoder_vocab_size: int,
        num_words_encoder: int = NUM_WORDS_ENCODER,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_words_encoder = num_words_encoder

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_vocab_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            d_feedforward,
            dropout=0,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers_encoder,
            enable_nested_tensor=False,
        )
        self.encoder_fc = torch.nn.Linear(d_model, 1)

        self.decoder_bag = torch.nn.EmbeddingBag(decoder_vocab_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_feedforward)
            for _ in range(num_layers_decoder)
        )
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """encoder入力とdecoder入力からvalueとpolicyを計算する。"""
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, self.num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        # policyは意図的にtanhをかけない(raw logits)。
        # 模倣学習(cross_entropy)は非有界logitを前提とする分類損失であり、
        # tanhで±1に制限すると上位候補手が容易に飽和して同値化し、
        # 手同士の優劣情報が消える(推論側のexp(policy*temperature)が
        # 実質one-hotになりMCTSのPUCT探索項を無効化する)。
        # 自己対戦(Huber回帰、教師はclamp(±1)された小さいQ優位度)側は
        # 目標値自体が小さいため、tanhが無くても実質的な挙動は変わらない。
        return v, p


def create_model(
    d_model: int = 128,
    num_heads: int = 2,
    d_feedforward: int = 256,
    num_layers_encoder: int = 1,
    num_layers_decoder: int = 1,
) -> MyModel:
    """現在のカード/攻撃定義に合わせたモデルを作成する。"""
    cards = card_count()
    attacks = attack_count()
    return MyModel(
        d_model=d_model,
        num_heads=num_heads,
        d_feedforward=d_feedforward,
        num_layers_encoder=num_layers_encoder,
        num_layers_decoder=num_layers_decoder,
        encoder_vocab_size=ENCODER_SIZE,
        decoder_vocab_size=decoder_size(cards, attacks),
    )
