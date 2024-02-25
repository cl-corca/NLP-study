import math
from typing import Generator
import torch
from torch import Tensor, nn
from constants import *

# special thanks to 경택. 
class RecursiveDeviceModule(nn.Module):
    def to(self, device: str):
        self._device = device
        for module in self.children():
            module.to(device)

class LookAheadMask(RecursiveDeviceModule):
    def __init__(self):
        super(LookAheadMask, self).__init__()

    def forward(self, x: Tensor):
        x = x.clone().detach()
        # Get upper triangular indices
        triu_indices = torch.triu_indices(x.size(1), x.size(1), offset=1)

        # Set elements above the diagonal to 1
        for i in range(len(triu_indices[0])):
            idx_x, idx_y = triu_indices[:, i]
            x[:, idx_x.item(), idx_y.item()] = 1
        return x

class PaddingMask(RecursiveDeviceModule):
    def __init__(self, pad_idx: int):
        super(PaddingMask, self).__init__()
        # Initialize parameters
        self.pad_idx = PAD_IDX

    def forward(self, x: Tensor):
        # x: (batch_size, max_sequence_length)
        # Create mask by comparing with pad_idx and repeating it along the second dimension
        return torch.eq(x, self.pad_idx).unsqueeze(1).repeat(1, x.size(1), 1)


class Attention(RecursiveDeviceModule):
    def __init__(self, d_model, num_heads, p_dropout):
        super(Attention, self).__init__()
        # Initialize parameters
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
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):        
        # Linear transformations
        q = self.w_q(q)  # (batch_size, max_sequence_length, d_n)
        k = self.w_k(k)  # (batch_size, max_sequence_length, d_n)
        v = self.w_v(v)  # (batch_size, max_sequence_length, d_n)
        
        # Compute attention scores
        attention_score = torch.matmul(q, torch.transpose(k, 1, 2)) / math.sqrt(self.d_n) + mask * -1e9
        # (batch_size, max_sequence_length, max_sequence_length)
        attention_score = self.softmax(attention_score)
        return torch.matmul(self.dropout(attention_score), v)


class MultiHeadAttention(RecursiveDeviceModule):
    def __init__(self, d_model, num_heads, prob_attention_dropout, prob_residual_dropout):
        super(MultiHeadAttention, self).__init__()
        # Initialize parameters
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_n = d_model // num_heads
        self.attentions = nn.ModuleList([
            Attention(d_model, num_heads, prob_attention_dropout) for _ in range(num_heads)
        ])
        
        # Linear transformation for output
        self.residual_dropout = nn.Dropout(p=prob_residual_dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor, encoder_output: Tensor | None = None):
        attention_outputs: list[Tensor] = []
        for attention in self.attentions:
            attention_outputs.append(attention(x, x, x, mask))
        attention_outputs = torch.cat(attention_outputs, dim=-1)
        return self.residual_dropout(self.proj(attention_outputs))


class DecoderLayer(RecursiveDeviceModule):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.attention = MultiHeadAttention(
            d_model=EMBEDDING_DIM,
            num_heads=HEAD,
            prob_attention_dropout=ATTENTION_DROPOUT,
            prob_residual_dropout=RESIDUAL_DROPOUT,
        )

        self.layer_norm_1 = nn.LayerNorm(EMBEDDING_DIM)
        self.layer_norm_2 = nn.LayerNorm(EMBEDDING_DIM)
        self.fc_1 = nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM)
        self.proj = nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(RESIDUAL_DROPOUT)

    def forward(self, x: Tensor, mask: Tensor):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)

        x = x + self.attention(self.layer_norm_1(x), mask)
        x = x + self.dropout(self.proj(self.gelu(self.fc_1(self.layer_norm_2(x)))))
        return x

class GPT(RecursiveDeviceModule):
    def __init__(self):
        super(GPT, self).__init__()
        # Initialize parameters
        self.word_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.embedding_dropout = nn.Dropout(EMBEDDING_DROPOUT)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer() for _ in range(LAYER)]
        )

        self.layer_norm = nn.LayerNorm(EMBEDDING_DIM)
        self.fc_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE, bias=False)
        self.block_size = BLOCK_SIZE
        self.padding_mask = PaddingMask(PAD_IDX)
        self.look_ahead_mask = LookAheadMask()
        self.weight_std = WEIGHT_STD
        self.weight_decay = WEIGHT_DECAY
        self.lr = LR
        self.adam_beta1 = ADAM_BETA1
        self.adam_beta2 = ADAM_BETA2


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                torch.nn.init.normal_(
                    p,
                    mean=0.0,
                    std=self.weight_std / math.sqrt(2 * LAYER),
                )
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.weight_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.weight_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_optimizer(self):
        decay_set: set[str] = set()
        no_decay_set: set[str] = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = ".".join([mn, pn]) if mn else pn
                if pn.endswith("bias"):
                    no_decay_set.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay_set.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay_set.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay_set & no_decay_set
        union_params = decay_set | no_decay_set
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay_set))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay_set))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            betas=(self.adam_beta1, self.adam_beta2),
            lr=self.lr,
        )
        return optimizer
            
    def forward(self, x: Tensor):
        # x: (batch_size, seq_len)

        mask = self.look_ahead_mask(
            self.padding_mask(x)
        )  # (batch_size, seq_len, seq_len)

        embeddings = self.word_embedding(x)  # (batch_size, seq_len, d_model)

        positions = torch.arange(embeddings.size(1), device=self._device).expand(
            embeddings.size(0), embeddings.size(1)
        )

        positions = self.position_embedding(positions)  # (batch_size, seq_len, d_model)

        x = self.embedding_dropout(
            embeddings + positions
        )  # (batch_size, seq_len, d_model)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        x = self.layer_norm(x)
        return self.fc_head(x)  # (batch_size, seq_len, VOCAB_SIZE)
    
    def _generate_next_token(
        self, x: Tensor, temperature: float, top_k: int | None
    ) -> Tensor:
        # x: (batch_size, seq_len)

        logits = self.forward(x)  # (batch_size, seq_len, VOCAB_SIZE)
        logits = logits[:, -1, :] / temperature  # (batch_size, VOCAB_SIZE)

        if top_k is not None:
            logits_topk, _indices = logits.topk(top_k, dim=-1)
            logits[logits < torch.min(logits_topk)] = -1e9
            probs = torch.softmax(logits, dim=-1)

            return torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        next_token = torch.argmax(logits, dim=-1)
        return next_token.unsqueeze(-1)

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        max_iter: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        # x: (1, seq_len)
        if x.size(0) != 1:
            raise ValueError("batch size should be 1")

        if str(x.device) != self._device:
            x = x.to(self._device)

        for _ in range(max_iter):
            x = (
                x
                if x.size(1) <= self.block_size
                else x[:, -self.block_size :]
            )
            next_token = self._generate_next_token(x, temperature, top_k)
            x = torch.cat([x, next_token], dim=-1)

        return x[:, -max_iter:]

    @torch.no_grad()
    def stream(
        self,
        x: Tensor,
        max_iter: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Generator[Tensor, None, None]:
        # x: (1, seq_len)
        if x.size(0) != 1:
            raise ValueError("batch size should be 1")

        if str(x.device) != self._device:
            x = x.to(self._device)

        for _ in range(max_iter):
            x = (
                x
                if x.size(1) <= self.block_size
                else x[:, -self.block_size :]
            )
            next_token = self._generate_next_token(x, temperature, top_k)
            yield next_token
            x = torch.cat([x, next_token], dim=-1)

if __name__ == "__main__":
    model = GPT()
    print(model)