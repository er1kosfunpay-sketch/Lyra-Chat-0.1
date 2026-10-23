"""AdamW parameter groups."""

from __future__ import annotations

from torch import nn
from torch.optim import AdamW

from ..config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        # Biases and normalization scales do not benefit from weight decay.
        if parameter.ndim < 2 or name.endswith(".bias") or "ln_" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay:
        raise RuntimeError("No weight-decay parameters found; check the model")
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )
