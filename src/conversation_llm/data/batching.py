"""DataLoader construction and padding collators."""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.reproducibility import seed_worker


def collate_pretrain(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def make_sft_collator(pad_id: int) -> Callable[[Sequence[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    def collate(batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    return collate


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable | None = None,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    seed: int = 42,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if os.name == "nt" and num_workers > 0:
        # Windows spawn workers work, but make the default predictable.
        num_workers = max(0, num_workers)
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, prefetch_factor)
    loader = DataLoader(dataset, **kwargs)
    if shuffle and len(loader) == 0:
        raise ValueError("DataLoader has zero batches; reduce batch_size or check the dataset")
    return loader
