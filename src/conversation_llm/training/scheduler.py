"""Learning-rate schedule: linear warmup followed by cosine decay."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_lr(step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    if max_steps <= warmup_steps:
        return max(min_lr_ratio, 0.0)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr(step, warmup_steps, max_steps, min_lr_ratio),
    )
