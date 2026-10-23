"""Decoder-only Transformer in the style of GPT-1/GPT-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attention import CausalSelfAttention
from .initialization import initialize_weights, scale_residual_projections


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attn(
            self.ln_1(x), attention_mask=attention_mask, past_kv=past_kv, use_cache=use_cache
        )
        x = x + self.resid_dropout(attn_out)
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x, present


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(lambda module: initialize_weights(module, config))
        scale_residual_projections(self, config)
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer["wte"].weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, time], got {tuple(input_ids.shape)}")
        batch, length = input_ids.shape
        if length < 1:
            raise ValueError("input_ids must contain at least one token")
        if length > self.config.block_size:
            raise ValueError(
                f"Sequence length {length} exceeds block_size {self.config.block_size}"
            )
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layer:
                raise ValueError("past_key_values must contain one entry per layer")
            past_length = past_key_values[0][0].shape[-2]

        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + length, device=input_ids.device
            ).unsqueeze(0).expand(batch, -1)
        else:
            if position_ids.ndim == 1:
                position_ids = position_ids.unsqueeze(0).expand(batch, -1)
            if position_ids.shape != (batch, length):
                raise ValueError(
                    f"position_ids must have shape {(batch, length)}, got {tuple(position_ids.shape)}"
                )
        if int(position_ids.max()) >= self.config.block_size:
            raise ValueError("position id exceeds block_size")

        x = self.transformer["wte"](input_ids) + self.transformer["wpe"](position_ids)
        x = self.transformer["drop"](x)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []

        use_checkpoint = (
            self.config.gradient_checkpointing
            and self.training
            and past_key_values is None
        )

        for index, block in enumerate(self.transformer["h"]):
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    lambda inp, attn_mask: block(
                        inp,
                        attention_mask=attn_mask,
                        past_kv=None,
                        use_cache=False,
                    )[0],
                    x,
                    attention_mask,
                    use_reentrant=True,
                )
            else:
                past = past_key_values[index] if past_key_values is not None else None
                x, present = block(
                    x,
                    attention_mask=attention_mask,
                    past_kv=past,
                    use_cache=use_cache,
                )
                if use_cache and present is not None:
                    presents.append(present)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels must match input_ids shape, got {tuple(labels.shape)}"
                )
            if not torch.any(labels != -100):
                raise ValueError("Batch contains no supervised target tokens")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if use_cache else None,
        )

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(parameter.numel() for parameter in self.parameters())
        if non_embedding:
            total -= self.transformer["wte"].weight.numel()
            total -= self.transformer["wpe"].weight.numel()
        return total

    def parameter_summary(self) -> dict[str, int]:
        embedding = self.transformer["wte"].weight.numel()
        positions = self.transformer["wpe"].weight.numel()
        total = self.num_parameters()
        return {
            "total": total,
            "token_embeddings": embedding,
            "position_embeddings": positions,
            "non_embedding": self.num_parameters(non_embedding=True),
            "output_head_tied": int(self.config.tie_embeddings),
        }
