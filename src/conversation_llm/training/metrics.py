"""Metric helpers used by the trainer and evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.count += weight

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def perplexity(loss: float) -> float | None:
    if not math.isfinite(loss) or loss > 50:
        return None
    return math.exp(loss)


def tokens_per_second(tokens: int, seconds: float) -> float:
    return tokens / seconds if seconds > 0 else 0.0


def format_metrics(metrics: dict[str, float | int | None]) -> str:
    parts: list[str] = []
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


@dataclass
class MetricHistory:
    records: list[dict[str, float | int]] = field(default_factory=list)

    def add(self, record: dict[str, float | int]) -> None:
        self.records.append(record)

    def values(self, key: str) -> list[float]:
        return [float(record[key]) for record in self.records if key in record]

    def summary(self, keys: Iterable[str]) -> dict[str, dict[str, float] | None]:
        out: dict[str, dict[str, float] | None] = {}
        for key in keys:
            values = self.values(key)
            if not values:
                out[key] = None
            else:
                out[key] = {
                    "first": values[0],
                    "last": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }
        return out
