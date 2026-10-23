"""Causal self-attention with padding masks and an incremental KV cache."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Pre-compute the lower-triangular causal mask once.  Buffers move with
        # the model to GPU/CPU automatically and are not recreated each forward.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones((config.block_size, config.block_size), dtype=torch.bool)),
            persistent=False,
        )

    def _split_heads(self, tensor: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        return tensor.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, channels = x.shape
        if channels != self.n_embd:
            raise ValueError(f"Expected hidden size {self.n_embd}, got {channels}")
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = self._split_heads(q, batch, length)
        k = self._split_heads(k, batch, length)
        v = self._split_heads(v, batch, length)

        past_length = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            if past_k.shape[0] != batch or past_k.shape[1] != self.n_head:
                raise ValueError("Cached key/value shapes do not match the current batch")
            past_length = past_k.shape[2]
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        total_length = past_length + length

        mask: torch.Tensor | None = None
        if length > 1:
            if past_length == 0:
                causal = self.causal_mask[:length, :length]
            else:
                # causal[i, j] = (j <= i + past_length) is equivalent to
                # base_mask[i + past_length, j] where base_mask[a, b] = (a >= b).
                causal = self.causal_mask[
                    past_length : past_length + length, :total_length
                ]
            mask = causal[None, None, :, :].to(device=x.device)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (batch, length):
                raise ValueError(
                    f"attention_mask must have shape {(batch, length)}, "
                    f"got {tuple(attention_mask.shape)}"
                )
            key_mask = attention_mask.to(dtype=torch.bool, device=x.device)
            if past_length:
                key_mask = torch.cat(
                    (
                        torch.ones((batch, past_length), dtype=torch.bool, device=x.device),
                        key_mask,
                    ),
                    dim=1,
                )
            key_mask = key_mask[:, None, None, :]
            mask = key_mask if mask is None else mask & key_mask

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.c_proj(output)
        present = (k, v) if use_cache else None
        return output, present


def causal_mask_reference(length: int, device: torch.device | None = None) -> torch.Tensor:
    """Explicit lower-triangular mask, useful in tests and diagnostics."""
    return torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))


def scaled_dot_product_attention_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    length = q.shape[-2]
    scores = scores.masked_fill(~causal_mask_reference(length, q.device), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v
