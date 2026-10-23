"""Transformer weight initialization."""

from __future__ import annotations

import math

import torch.nn as nn

from ..config import ModelConfig


def initialize_weights(module: nn.Module, config: ModelConfig) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def scale_residual_projections(model: nn.Module, config: ModelConfig) -> None:
    """GPT-2 style 1/sqrt(2N) scaling for residual output projections."""
    std = config.init_std / math.sqrt(2 * config.n_layer)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(("attn.c_proj", "mlp.c_proj")):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1
    if count != 2 * config.n_layer:
        raise RuntimeError(
            f"Expected {2 * config.n_layer} residual projections, scaled {count}. "
            "Module naming changed?"
        )
