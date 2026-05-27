import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(
            self.conv2(self.dropout1(self.relu(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        return outputs + inputs


class TiSASRecAttention(nn.Module):
    """Self-attention with absolute position and time-interval K/V (TiSASRec)."""

    def __init__(
        self,
        hidden_units: int,
        num_heads: int,
        maxlen: int,
        time_span: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if hidden_units % num_heads != 0:
            raise ValueError("hidden_units must be divisible by num_heads")
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.maxlen = maxlen

        self.q_linear = nn.Linear(hidden_units, hidden_units)
        self.k_linear = nn.Linear(hidden_units, hidden_units)
        self.v_linear = nn.Linear(hidden_units, hidden_units)
        self.out_linear = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        self.abs_pos_K = nn.Embedding(maxlen, hidden_units)
        self.abs_pos_V = nn.Embedding(maxlen, hidden_units)
        self.time_emb_K = nn.Embedding(time_span + 1, hidden_units)
        self.time_emb_V = nn.Embedding(time_span + 1, hidden_units)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        time_matrix: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = queries.shape
        q = self.q_linear(queries)
        k = self.k_linear(keys)
        v = self.v_linear(keys)

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        positions = torch.arange(seq_len, device=queries.device)
        abs_k = split_heads(self.abs_pos_K(positions).unsqueeze(0).expand(batch_size, -1, -1))
        abs_v = split_heads(self.abs_pos_V(positions).unsqueeze(0).expand(batch_size, -1, -1))

        time_k = self.time_emb_K(time_matrix)
        time_v = self.time_emb_V(time_matrix)
        time_k = time_k.view(
            batch_size, seq_len, seq_len, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)
        time_v = time_v.view(
            batch_size, seq_len, seq_len, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        scores = scores + torch.matmul(q, abs_k.transpose(-2, -1))
        scores = scores + torch.einsum("bhid,bhijd->bhij", q, time_k)
        scores = scores.masked_fill(attn_mask.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out + torch.matmul(attn, abs_v)
        out = out + torch.einsum("bhij,bhijd->bhid", attn, time_v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        return self.out_linear(out)


class TiSASRec(nn.Module):
    """End-to-end time-aware sequential recommender."""

    def __init__(
        self,
        num_items: int,
        hidden_units: int = 128,
        maxlen: int = 100,
        time_span: int = 256,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.maxlen = maxlen
        self.time_span = time_span

        self.item_emb = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attn_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)]
        )
        self.attn_layers = nn.ModuleList(
            [
                TiSASRecAttention(hidden_units, num_heads, maxlen, time_span, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.forward_layernorms = nn.ModuleList(
            [nn.LayerNorm(hidden_units, eps=1e-8) for _ in range(num_blocks)]
        )
        self.forward_layers = nn.ModuleList(
            [PointWiseFeedForward(hidden_units, dropout_rate) for _ in range(num_blocks)]
        )
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_units)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    def _attention_mask(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_keys = seq.eq(0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        pad_queries = seq.eq(0).unsqueeze(-1).expand(batch_size, seq_len, seq_len)
        return causal.unsqueeze(0) | pad_keys | pad_queries

    def encode(self, seq: torch.Tensor, time_matrix: torch.Tensor) -> torch.Tensor:
        seqs = self.item_emb(seq) * math.sqrt(self.hidden_units)
        seqs = self.emb_dropout(seqs)

        pad_mask = seq.eq(0)
        attn_mask = self._attention_mask(seq)

        for i in range(len(self.attn_layers)):
            x = self.attn_layernorms[i](seqs)
            attn_out = self.attn_layers[i](x, x, time_matrix, attn_mask)
            seqs = seqs + attn_out
            seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            seqs = seqs.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return self.last_layernorm(seqs)

    def forward(
        self,
        seq: torch.Tensor,
        time_matrix: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
    ) -> torch.Tensor:
        seqs = self.encode(seq, time_matrix)
        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)

        pos_logits = (seqs * pos_emb).sum(dim=-1)
        neg_logits = (seqs * neg_emb).sum(dim=-1)

        loss_mask = pos.ne(0).float()
        loss = -(
            F.logsigmoid(pos_logits) + F.logsigmoid(-neg_logits)
        ) * loss_mask
        denom = loss_mask.sum()
        if denom.item() == 0:
            return loss.sum() * 0.0
        return loss.sum() / denom

    @torch.no_grad()
    def predict_next(
        self,
        seq: torch.Tensor,
        time_matrix: torch.Tensor,
        candidate_items: torch.Tensor,
    ) -> torch.Tensor:
        seqs = self.encode(seq, time_matrix)
        lengths = seq.ne(0).sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(seq.size(0), device=seq.device)
        last_repr = seqs[batch_idx, lengths]
        cand_emb = self.item_emb(candidate_items)
        scores = torch.matmul(last_repr.unsqueeze(1), cand_emb.transpose(1, 2)).squeeze(1)
        return scores
