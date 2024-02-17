import math
import torch
from torch import Tensor, nn
from constants import *

class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)

class PositionalEncoding(RecursiveDeviceModule):
    def __init__(self, max_sequence_length: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

        self.pe = torch.zeros(max_sequence_length, d_model)
        for pos in range(max_sequence_length):
            for i in range(d_model // 2):
                self.pe[pos, 2 * i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pe[pos, 2 * i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / d_model))
                )
        self._device = DEVICE

    def to(self, device: str):
        self.pe = self.pe.to(device)
        return super().to(device)

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length, d_model)
        return x + self.pe[: x.size(1), :]


class Attention(RecursiveDeviceModule):
    def __init__(self, max_sequence_length: int, d_model: int, num_heads: int):
        super(Attention, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_n = d_model // num_heads

        self.w_q = nn.Linear(d_model, self.d_n)
        self.w_k = nn.Linear(d_model, self.d_n)
        self.w_v = nn.Linear(d_model, self.d_n)

        self.softmax = nn.Softmax(dim=-1)
        self._device = DEVICE


    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        # q, k, v: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)

        q = self.w_q(q)  # (batch_size, max_sequence_length, d_n)
        k = self.w_k(k)  # (batch_size, max_sequence_length, d_n)
        v = self.w_v(v)  # (batch_size, max_sequence_length, d_n)

        attention_score = (
            torch.matmul(q, torch.transpose(k, 1, 2)) / math.sqrt(self.d_n)
        ) + mask * -1e9  # (batch_size, max_sequence_length, max_sequence_length)

        return torch.matmul(
            self.softmax(attention_score), v
        )  # (batch_size, max_sequence_length, d_n)

class MultiHeadAttention(RecursiveDeviceModule):
    def __init__(self, max_sequence_length: int, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_n = d_model // num_heads

        self.attentions = nn.ModuleList(
            [Attention(max_sequence_length, d_model, num_heads) for _ in range(num_heads)]
        )

        self.w_o = nn.Linear(d_model, d_model)
        self._device = DEVICE


    def forward(self, x: Tensor, mask: Tensor, encoder_output: Tensor | None = None):
        # x, encoder_output: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)

        # splitted_x_list = torch.split(x, self.num_heads, dim=-2)
        attention_outputs: list[Tensor] = []
        for attention in self.attentions:
            if encoder_output is not None:
                attention_outputs.append(
                    attention(x, encoder_output, encoder_output, mask)
                )
                continue
            attention_outputs.append(attention(x, x, x, mask))

        attention_outputs = torch.cat(attention_outputs, dim=-1)

        return self.w_o(attention_outputs)  # (batch_size, max_sequence_length, d_model)


class PositionWiseFeedForward(RecursiveDeviceModule):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self._device = DEVICE


    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length, d_model)

        return self.w_2(torch.relu(self.w_1(x)))  # (batch_size, max_sequence_length, d_model)


class EncoderLayer(RecursiveDeviceModule):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_dropout: float,
    ):
        super(EncoderLayer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        self.multi_head_attention = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff        
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
        self._device = DEVICE


    def forward(self, x: Tensor, mask: Tensor):
        # x: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)

        x = self.layer_norm_1(
            x + self.dropout(self.multi_head_attention(x, mask))
        )  # (batch_size, max_sequence_length, d_model)
        x = self.layer_norm_2(
            x + self.dropout(self.position_wise_feed_forward(x))
        )  # (batch_size, max_sequence_length, d_model)

        return x  # (batch_size, max_sequence_length, d_model)


class PaddingMask(RecursiveDeviceModule):
    def __init__(self, pad_idx: int):
        super(PaddingMask, self).__init__()
        self.pad_idx = pad_idx
        self._device = DEVICE

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length)

        return torch.eq(x, self.pad_idx).unsqueeze(1).repeat(1, x.size(1), 1)


class LookAheadMask(RecursiveDeviceModule):
    def __init__(self):
        super(LookAheadMask, self).__init__()
        self._device = DEVICE

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length, max_sequence_length)
        x = x.clone().detach()
        triu_indices = torch.triu_indices(x.size(1), x.size(1), offset=1)

        for i in range(len(triu_indices[0])):
            idx_x, idx_y = triu_indices[:, i]
            x[:, idx_x.item(), idx_y.item()] = 1
        return x


class Encoder(RecursiveDeviceModule):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_dropout: float,
        num_layers: int,
        embedding_layer: nn.Embedding,
    ):
        super(Encoder, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self.embedding_layer = embedding_layer

        self.padding_mask = PaddingMask(pad_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)
        self.dropout = nn.Dropout(p_dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    vocab_size=vocab_size,
                    max_sequence_length=max_sequence_length,
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    p_dropout=p_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self._device = DEVICE


    def forward(self, x: Tensor, padding_mask: Tensor):
        # x: (batch_size, max_sequence_length)
        # padding_mask: (batch_size, max_sequence_length, max_sequence_length)

        x = self.embedding_layer(x)  # (batch_size, max_sequence_length, d_model)
        x = self.dropout(self.positional_encoding(x))  # (batch_size, max_sequence_length, d_model)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, padding_mask)

        return x


class DecoderLayer(RecursiveDeviceModule):
    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_dropout: float,
    ):
        super(DecoderLayer, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads

        self.multi_head_attention_1 = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.multi_head_attention_2 = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff
        )
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
        self._device = DEVICE


    def forward(
        self, x: Tensor, mask: Tensor, lookahead_mask: Tensor, encoder_output: Tensor
    ):
        # x, encoder_output: (batch_size, max_sequence_length, d_model)
        # mask, lookahead_mask: (batch_size, max_sequence_length, max_sequence_length)

        x = self.layer_norm_1(
            x + self.dropout(self.multi_head_attention_1(x, lookahead_mask))
        )  # (batch_size, max_sequence_length, d_model)
        x = self.layer_norm_2(
            x + self.dropout(self.multi_head_attention_2(x, mask, encoder_output))
        )  # (batch_size, max_sequence_length, d_model)
        x = self.layer_norm_3(
            x + self.dropout(self.position_wise_feed_forward(x))
        )  # (batch_size, max_sequence_length, d_model)

        return x  # (batch_size, max_sequence_length, d_model)


class Decoder(RecursiveDeviceModule):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        max_sequence_length: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        p_dropout: float,
        num_layers: int,
        embedding_layer: nn.Embedding,
    ):
        super(Decoder, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self.embedding_layer = embedding_layer

        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)
        self.dropout = nn.Dropout(p_dropout)

        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    vocab_size=vocab_size,
                    max_sequence_length=max_sequence_length,
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    p_dropout=p_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self._device = DEVICE


    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor,
        lookahead_mask: Tensor,
        encoder_output: Tensor,
    ):
        # x: (batch_size, max_sequence_length)
        # encoder_output: (batch_size, max_sequence_length, d_model)
        # padding_mask, lookahead_mask: (batch_size, max_sequence_length, max_sequence_length)

        x = self.embedding_layer(x)
        x = self.dropout(self.positional_encoding(x))  # (batch_size, max_sequence_length, d_model)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, padding_mask, lookahead_mask, encoder_output)

        return x

class Transformer(RecursiveDeviceModule):
    def __init__(
        self,
        pad_idx: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_sequence_length: int,
        p_dropout: float,
    ):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, d_model)

        self.padding_mask = PaddingMask(pad_idx=pad_idx)
        self.lookahead_mask = LookAheadMask()
        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)

        self.encoder = Encoder(
            pad_idx=pad_idx,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            p_dropout=p_dropout,
            num_layers=num_layers,
            embedding_layer=self.embedding_layer,
        )

        self.decoder = Decoder(
            pad_idx=pad_idx,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            p_dropout=p_dropout,
            num_layers=num_layers,
            embedding_layer=self.embedding_layer,
        )

        self.linear = nn.Linear(d_model, vocab_size)
        self._device = DEVICE


    def forward(self, x: Tensor, y: Tensor):
        # x, y: (batch_size, max_sequence_length)

        lookahead_mask = self.lookahead_mask(
            self.padding_mask(y)
        )  # (batch_size, max_sequence_length, max_sequence_length)

        padding_mask = self.padding_mask(x)  # (batch_size, max_sequence_length, max_sequence_length)

        encoder_output = self.encoder(x, padding_mask)
        decoder_output = self.decoder(y, padding_mask, lookahead_mask, encoder_output)

        return self.linear(decoder_output)  # (batch_size, max_sequence_length, vocab_size)

