"""Atomic checkpoints with enough state to truly resume training."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import ModelConfig, TrainingConfig

CHECKPOINT_VERSION = 2


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    tokens_seen: int,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    best_val_loss: float | None = None,
    scaler: Any = None,
    rng_state: dict[str, Any] | None = None,
    data_generator_state: dict[str, Any] | torch.Tensor | None = None,
    tokenizer_fingerprint: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_VERSION,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "best_val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "metrics": metrics or {},
        "saved_at": time.time(),
    }
    if rng_state is not None:
        payload["rng_state"] = rng_state
    if data_generator_state is not None:
        payload["data_generator_state"] = data_generator_state
    _atomic_torch_save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{path} is not a conversation_llm checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        if payload.get("optimizer") is None:
            raise ValueError("Checkpoint has no optimizer state; cannot resume exactly")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        from ..utils.reproducibility import restore_rng_state

        restore_rng_state(payload.get("rng_state"))
    return payload


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler", "scaler", "rng_state"}
    }


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    last = directory / "last.pt"
    if last.exists():
        return last
    candidates = sorted(
        directory.glob("step_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
    )
    return candidates[-1] if candidates else None


def prune_checkpoints(directory: str | Path, keep: int = 3) -> list[Path]:
    directory = Path(directory)
    if keep < 1:
        return []
    candidates = sorted(
        directory.glob("step_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
    )
    removed: list[Path] = []
    for path in candidates[:-keep]:
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            pass
    return removed


def write_metrics_summary(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
