import math
import torch
from torch import Tensor, nn
from constants import *

# special thanks to 경택. without this suggestion, I did not figure out to avoid divice mismatch error 
class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)

class PositionalEncoding(RecursiveDeviceModule):
    def __init__(self, max_sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        # Initialize parameters
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
        # Initialize positional encoding matrix
        self.pe = self.generate_positional_encoding(max_sequence_length, d_model)
        self._device = DEVICE
    
    def generate_positional_encoding(self, max_sequence_length, d_model):
        pe = torch.zeros(max_sequence_length, d_model)
        for pos in range(max_sequence_length):
            for i in range(d_model // 2):
                pe[pos, 2 * i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, 2 * i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        return pe
    
    def to(self, device):
        # Move positional encoding matrix to specified device
        self.pe = self.pe.to(device)
        return super().to(device)

    def forward(self, x: Tensor):
        # Add positional encoding to input tensor
        return x + self.pe[: x.size(1), :]


class Attention(RecursiveDeviceModule):
    def __init__(self, max_sequence_length, d_model, num_heads):
        super(Attention, self).__init__()
        # Initialize parameters
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Calculate dimensionality of each head
        self.d_n = d_model // num_heads
        
        # Linear transformations for query, key, and value
        self.w_q = nn.Linear(d_model, self.d_n)
        self.w_k = nn.Linear(d_model, self.d_n)
        self.w_v = nn.Linear(d_model, self.d_n)
        
        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)
        self._device = DEVICE

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        # q, k, v: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)
        
        # Linear transformations
        q = self.w_q(q)  # (batch_size, max_sequence_length, d_n)
        k = self.w_k(k)  # (batch_size, max_sequence_length, d_n)
        v = self.w_v(v)  # (batch_size, max_sequence_length, d_n)
        
        # Compute attention scores
        attention_score = torch.matmul(q, torch.transpose(k, 1, 2)) / math.sqrt(self.d_n) + mask * -1e9
        # (batch_size, max_sequence_length, max_sequence_length)
        
        # Apply softmax and compute weighted sum
        return torch.matmul(self.softmax(attention_score), v)
        # (batch_size, max_sequence_length, d_n)


class MultiHeadAttention(RecursiveDeviceModule):
    def __init__(self, max_sequence_length, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Initialize parameters
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Calculate dimensionality of each head
        self.d_n = d_model // num_heads
        
        # Initialize multiple attention heads
        self.attentions = nn.ModuleList([
            Attention(max_sequence_length, d_model, num_heads) for _ in range(num_heads)
        ])
        
        # Linear transformation for output
        self.w_o = nn.Linear(d_model, d_model)
        self._device = DEVICE

    def forward(self, x: Tensor, mask: Tensor, encoder_output: Tensor | None = None):
        # x, encoder_output: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)
        
        attention_outputs: list[Tensor] = []
        for attention in self.attentions:
            if encoder_output is not None:
                # If encoder_output is provided, use it as key and value
                attention_outputs.append(attention(x, encoder_output, encoder_output, mask))
                continue
            # Otherwise, use x as key, query, and value
            attention_outputs.append(attention(x, x, x, mask))
        
        # Concatenate attention outputs from all heads
        attention_outputs = torch.cat(attention_outputs, dim=-1)

        # Apply linear transformation
        return self.w_o(attention_outputs)
        # (batch_size, max_sequence_length, d_model)


class PositionWiseFeedForward(RecursiveDeviceModule):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        # Initialize parameters
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Linear transformations
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self._device = DEVICE

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length, d_model)
        
        # Apply first linear transformation and activation function
        intermediate_output = torch.relu(self.w_1(x))
        
        # Apply second linear transformation
        return self.w_2(intermediate_output)
        # (batch_size, max_sequence_length, d_model)


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
        # Initialize parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Components of encoder layer
        self.multi_head_attention = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff
        )
        
        # Layer normalization and dropout
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_dropout)
        self._device = DEVICE

    def forward(self, x: Tensor, mask: Tensor):
        # x: (batch_size, max_sequence_length, d_model)
        # mask: (batch_size, max_sequence_length, max_sequence_length)
        
        # Multi-head attention
        attention_output = self.multi_head_attention(x, mask)
        # Apply dropout and residual connection, then layer normalization
        x = self.layer_norm_1(x + self.dropout(attention_output))
        
        # Position-wise feed-forward network
        ff_output = self.position_wise_feed_forward(x)
        # Apply dropout and residual connection, then layer normalization
        x = self.layer_norm_2(x + self.dropout(ff_output))
        
        return x  # (batch_size, max_sequence_length, d_model)


class PaddingMask(RecursiveDeviceModule):
    def __init__(self, pad_idx: int):
        super(PaddingMask, self).__init__()
        # Initialize parameters
        self.pad_idx = pad_idx
        self._device = DEVICE

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length)
        # Create mask by comparing with pad_idx and repeating it along the second dimension
        return torch.eq(x, self.pad_idx).unsqueeze(1).repeat(1, x.size(1), 1)


class LookAheadMask(RecursiveDeviceModule):
    def __init__(self):
        super(LookAheadMask, self).__init__()
        self._device = DEVICE

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length, max_sequence_length)
        # Create a copy of input tensor
        x = x.clone().detach()
        # Get upper triangular indices
        triu_indices = torch.triu_indices(x.size(1), x.size(1), offset=1)

        # Set elements above the diagonal to 1
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
        # Initialize parameters
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self.embedding_layer = embedding_layer
        
        # Components of encoder
        self.padding_mask = PaddingMask(pad_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)
        self.dropout = nn.Dropout(p_dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                p_dropout=p_dropout,
            ) for _ in range(num_layers)
        ])
        self._device = DEVICE

    def forward(self, x: Tensor, padding_mask: Tensor):
        # x: (batch_size, max_sequence_length)
        # padding_mask: (batch_size, max_sequence_length, max_sequence_length)
        
        # Embedding and positional encoding
        x = self.embedding_layer(x)  # (batch_size, max_sequence_length, d_model)
        x = self.dropout(self.positional_encoding(x))  # (batch_size, max_sequence_length, d_model)
        
        # Pass through each encoder layer
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
        # Initialize parameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Components of decoder layer
        self.multi_head_attention_1 = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.multi_head_attention_2 = MultiHeadAttention(
            max_sequence_length=max_sequence_length, d_model=d_model, num_heads=num_heads
        )
        self.position_wise_feed_forward = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff
        )
        
        # Layer normalization and dropout
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
        
        # First multi-head attention layer with self-attention
        attention_output_1 = self.multi_head_attention_1(x, lookahead_mask)
        # Apply dropout and residual connection, then layer normalization
        x = self.layer_norm_1(x + self.dropout(attention_output_1))
        
        # Second multi-head attention layer with encoder-decoder attention
        attention_output_2 = self.multi_head_attention_2(x, mask, encoder_output)
        # Apply dropout and residual connection, then layer normalization
        x = self.layer_norm_2(x + self.dropout(attention_output_2))
        
        # Position-wise feed-forward network
        ff_output = self.position_wise_feed_forward(x)
        # Apply dropout and residual connection, then layer normalization
        x = self.layer_norm_3(x + self.dropout(ff_output))
        
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
        # Initialize parameters
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        self.num_layers = num_layers
        self.embedding_layer = embedding_layer
        
        # Components of decoder
        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)
        self.dropout = nn.Dropout(p_dropout)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                p_dropout=p_dropout,
            ) for _ in range(num_layers)
        ])
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
        
        # Embedding and positional encoding
        x = self.embedding_layer(x)
        x = self.dropout(self.positional_encoding(x))  # (batch_size, max_sequence_length, d_model)
        
        # Pass through each decoder layer
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
        # Initialize parameters
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.p_dropout = p_dropout
        
        # Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        
        # Masks and positional encoding
        self.padding_mask = PaddingMask(pad_idx=pad_idx)
        self.lookahead_mask = LookAheadMask()
        self.positional_encoding = PositionalEncoding(max_sequence_length=max_sequence_length, d_model=d_model)
        
        # Encoder and Decoder
        self.encoder = self.initialize_encoder_decoder(Encoder)
        self.decoder = self.initialize_encoder_decoder(Decoder)
        
        # Output linear layer
        self.linear = nn.Linear(d_model, vocab_size)
        self._device = DEVICE
    
    def initialize_encoder_decoder(self, model_class):
        return model_class(
            pad_idx=self.pad_idx,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            p_dropout=self.p_dropout,
            num_layers=self.num_layers,
            embedding_layer=self.embedding_layer,
        )
        
    def forward(self, x: Tensor, y: Tensor):
        # x, y: (batch_size, max_sequence_length)
        
        # Generate masks
        lookahead_mask = self.lookahead_mask(self.padding_mask(y))  # (batch_size, max_sequence_length, max_sequence_length)
        padding_mask = self.padding_mask(x)  # (batch_size, max_sequence_length, max_sequence_length)
        
        # Encoder-Decoder computation
        encoder_output = self.encoder(x, padding_mask)
        decoder_output = self.decoder(y, padding_mask, lookahead_mask, encoder_output)
        
        # Linear layer for output
        return self.linear(decoder_output)  # (batch_size, max_sequence_length, vocab_size)
