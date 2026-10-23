"""Training loop with AMP, accumulation, validation, checkpoints and resume."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from ..config import ModelConfig, TrainingConfig
from ..utils.device import amp_settings, describe_device
from ..utils.logging import JsonlLogger, setup_logger
from ..utils.reproducibility import capture_rng_state, set_seed
from .checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    write_metrics_summary,
)
from .metrics import RunningAverage, perplexity, tokens_per_second
from .optimizer import build_optimizer
from .scheduler import build_scheduler


@dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    supervised_tokens: int = 0
    best_val_loss: float | None = None
    skipped_steps: int = 0
    resumed_from: str | None = None
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        run_dir: str | Path,
        tokenizer_fingerprint: str | None = None,
        data_meta: dict[str, Any] | None = None,
        logger: Any = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / training_config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.data_meta = data_meta or {}
        self.logger = logger or setup_logger(
            "conversation_llm.trainer", self.run_dir / "train.log"
        )
        self.jsonl = JsonlLogger(self.run_dir / "train.jsonl") if training_config.log_jsonl else None

        self.amp_enabled, self.amp_dtype, scaler_needed = amp_settings(device, training_config.amp)
        self.scaler = (
            torch.amp.GradScaler(device.type, enabled=True) if scaler_needed else None
        )
        self.optimizer = build_optimizer(model, training_config)
        self.scheduler = build_scheduler(
            self.optimizer,
            training_config.warmup_steps,
            training_config.max_steps,
            training_config.min_lr_ratio,
        )
        if training_config.compile and device.type == "cuda":
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")
        self.state = TrainState()
        self.session_start_tokens = 0
        self._iterator: Any = None
        self._stop_reason = "max_steps"
        # Deterministic, resumable data order.  A DataLoader restart after a
        # mid-epoch checkpoint must replay the same permutation and skip the
        # batches that were already consumed, not silently jump to the next
        # epoch (which would skip thousands of examples).
        self._epoch = 0
        self._batch_in_epoch = 0
        self._resume_skip_batches = 0

        config_dump = {
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "device": describe_device(device),
            "amp": {"enabled": self.amp_enabled, "dtype": str(self.amp_dtype)},
            "parameters": self._base_model().num_parameters()
            if hasattr(self._base_model(), "num_parameters")
            else None,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "data": self.data_meta,
        }
        write_metrics_summary(self.run_dir / "run_config.json", config_dump)
        self.logger.info(
            "Trainer ready: device=%s amp=%s params=%s",
            device,
            self.amp_dtype if self.amp_enabled else "off",
            f"{self._base_model().num_parameters():,}"
            if hasattr(self._base_model(), "num_parameters")
            else "n/a",
        )

    # ------------------------------------------------------------------ resume
    def _base_model(self) -> nn.Module:
        return getattr(self.model, "_orig_mod", self.model)

    def resume(self, path: str | Path | None = "auto") -> bool:
        if path in {None, "auto"}:
            checkpoint = find_latest_checkpoint(self.checkpoint_dir)
            if checkpoint is None:
                if path == "auto":
                    self.logger.info("No checkpoint found; starting from scratch")
                    return False
                raise FileNotFoundError(f"No checkpoint in {self.checkpoint_dir}")
        else:
            checkpoint = Path(path)
            if not checkpoint.is_absolute():
                checkpoint = self.run_dir / checkpoint
        payload = load_checkpoint(
            checkpoint,
            self._base_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            map_location=self.device,
        )
        saved_model = payload.get("model_config", {})
        for key in ("n_layer", "n_head", "n_embd", "vocab_size", "block_size"):
            if saved_model.get(key) != getattr(self.model_config, key):
                raise ValueError(
                    f"Checkpoint architecture mismatch for {key}: "
                    f"{saved_model.get(key)} != {getattr(self.model_config, key)}"
                )
        if self.tokenizer_fingerprint and payload.get("tokenizer_fingerprint"):
            if payload["tokenizer_fingerprint"] != self.tokenizer_fingerprint:
                raise RuntimeError(
                    "Checkpoint was trained with a different tokenizer. "
                    "Refusing to resume with incompatible token ids."
                )
        self.state.step = int(payload.get("step", 0))
        self.state.tokens_seen = int(payload.get("tokens_seen", 0))
        self.state.best_val_loss = payload.get("best_val_loss")
        self.state.resumed_from = str(checkpoint)
        self.session_start_tokens = self.state.tokens_seen
        data_state = payload.get("data_generator_state")
        if isinstance(data_state, dict) and "epoch" in data_state:
            saved_seed = data_state.get("seed")
            saved_batches = data_state.get("batches_per_epoch")
            if saved_seed is not None and int(saved_seed) != int(self.config.seed):
                raise ValueError(
                    "Checkpoint data cursor was created with seed "
                    f"{saved_seed}, but the current seed is {self.config.seed}. "
                    "Use the original seed or start a new run."
                )
            if saved_batches is not None and int(saved_batches) != len(self.train_loader):
                raise ValueError(
                    "Checkpoint data cursor expects "
                    f"{saved_batches} batches per epoch, but the current loader has "
                    f"{len(self.train_loader)}. Keep batch_size and the dataset "
                    "unchanged when resuming, or start a new run."
                )
            self._epoch = int(data_state.get("epoch", 0))
            self._batch_in_epoch = int(data_state.get("batch_in_epoch", 0))
            self._resume_skip_batches = self._batch_in_epoch
            self.logger.info(
                "Resuming data stream at epoch %d, batch %d",
                self._epoch,
                self._batch_in_epoch,
            )
        elif data_state is not None and getattr(self.train_loader, "generator", None) is not None:
            # Backward compatibility with checkpoints written before the
            # explicit data cursor was introduced.
            self.train_loader.generator.set_state(data_state)
        self._iterator = None
        self.logger.info(
            "Resumed from %s at step %d (%d tokens seen)",
            checkpoint,
            self.state.step,
            self.state.tokens_seen,
        )
        return True

    # --------------------------------------------------------------- data utils
    def _cycle_batches(self):
        skip = self._resume_skip_batches
        self._resume_skip_batches = 0
        while True:
            generator = getattr(self.train_loader, "generator", None)
            if generator is not None:
                # Same permutation on replay of a partially consumed epoch.
                generator.manual_seed(self.config.seed + self._epoch)
            self._batch_in_epoch = 0
            for index, batch in enumerate(self.train_loader):
                if index < skip:
                    continue
                self._batch_in_epoch = index + 1
                yield batch
            self._epoch += 1
            skip = 0

    def _next_batch(self):
        if self._iterator is None:
            self._iterator = iter(self._cycle_batches())
        return next(self._iterator)

    def _move(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }

    def _forward_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        targets = int((labels != -100).sum().item())
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        if output.loss is None:
            raise RuntimeError("Model did not return a loss")
        return output.loss, targets

    # --------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self) -> dict[str, float | int | None]:
        self.model.eval()
        total_loss = 0.0
        total_targets = 0
        batches = 0
        for batch in self.val_loader:
            batch = self._move(batch)
            loss, targets = self._forward_loss(batch)
            value = float(loss.item())
            if not math.isfinite(value):
                self.logger.warning(
                    "Validation loss is not finite: %s; skipping this validation batch",
                    value,
                )
                continue
            total_loss += value * targets
            total_targets += targets
            batches += 1
            if batches >= self.config.eval_batches:
                break
        self.model.train()
        if total_targets == 0:
            return {"val_loss": None, "val_perplexity": None, "val_batches": batches}
        value = total_loss / total_targets
        result = {
            "val_loss": value,
            "val_perplexity": perplexity(value),
            "val_batches": batches,
            "val_target_tokens": total_targets,
        }
        if result["val_perplexity"] is not None:
            self.logger.info(
                "Validation: loss=%.4f ppl=%.2f (%d target tokens)",
                value,
                result["val_perplexity"],
                total_targets,
            )
        else:
            self.logger.info("Validation: loss=%.4f (%d target tokens)", value, total_targets)
        return result

    # -------------------------------------------------------------------- saving
    def _data_generator_state(self) -> dict[str, int] | None:
        if self.train_loader is None:
            return None
        return {
            "epoch": int(self._epoch),
            "batch_in_epoch": int(self._batch_in_epoch),
            "seed": int(self.config.seed),
            "batches_per_epoch": int(len(self.train_loader)),
        }

    def save(self, filename: str, best: bool = False) -> Path:
        path = self.checkpoint_dir / filename
        save_checkpoint(
            path,
            model=self._base_model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.state.step,
            tokens_seen=self.state.tokens_seen,
            best_val_loss=self.state.best_val_loss,
            model_config=self.model_config,
            training_config=self.config,
            rng_state=capture_rng_state(),
            data_generator_state=self._data_generator_state(),
            tokenizer_fingerprint=self.tokenizer_fingerprint,
            metrics={
                "train_losses": self.state.train_losses[-20:],
                "val_losses": self.state.val_losses[-20:],
                "supervised_tokens": self.state.supervised_tokens,
                "skipped_steps": self.state.skipped_steps,
                "best": best,
            },
        )
        return path

    def _save_periodic(self) -> None:
        self.save("last.pt")
        self.save(f"step_{self.state.step:07d}.pt")
        prune_checkpoints(self.checkpoint_dir, keep=self.config.keep_last_checkpoints)

    # --------------------------------------------------------------------- train
    def train(self) -> dict[str, Any]:
        start = time.time()
        limit_seconds = self.config.max_minutes * 60.0
        if self.config.stop_after_seconds is not None:
            limit_seconds = min(limit_seconds, self.config.stop_after_seconds)
        deadline = start + limit_seconds
        self.model.train()
        window = RunningAverage()
        micro_loss = 0.0
        micro_count = 0
        nonfinite_streak = 0
        log_tokens = self.state.tokens_seen
        log_time = start

        self.logger.info(
            "Starting training at step %d/%d, deadline in %.1f min",
            self.state.step,
            self.config.max_steps,
            limit_seconds / 60.0,
        )
        try:
            while self.state.step < self.config.max_steps:
                if time.time() >= deadline:
                    self._stop_reason = "time_limit"
                    self.logger.info("Time budget reached; stopping gracefully")
                    break
                self.optimizer.zero_grad(set_to_none=True)
                micro_loss = 0.0
                micro_count = 0
                batch_tokens = 0
                supervised = 0
                step_ok = True
                try:
                    for _ in range(self.config.grad_accum_steps):
                        batch = self._move(self._next_batch())
                        loss, targets = self._forward_loss(batch)
                        loss_value = float(loss.item())
                        if not math.isfinite(loss_value):
                            step_ok = False
                            break
                        scaled = loss / self.config.grad_accum_steps
                        if self.scaler is not None:
                            self.scaler.scale(scaled).backward()
                        else:
                            scaled.backward()
                        micro_loss += loss_value
                        micro_count += 1
                        batch_tokens += int(batch["input_ids"].numel())
                        supervised += targets

                    if step_ok:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                        if not torch.isfinite(grad_norm):
                            step_ok = False
                        else:
                            if self.scaler is not None:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()
                            self.scheduler.step()
                            self.state.step += 1
                            self.state.tokens_seen += batch_tokens
                            self.state.supervised_tokens += supervised
                            window.update(micro_loss / max(micro_count, 1))
                            self.state.train_losses.append(micro_loss / max(micro_count, 1))
                            nonfinite_streak = 0
                    if not step_ok:
                        if self.scaler is not None:
                            self.scaler.update()
                        self.state.skipped_steps += 1
                        nonfinite_streak += 1
                        self.logger.warning(
                            "Skipping step %d: non-finite loss or gradients (streak=%d)",
                            self.state.step,
                            nonfinite_streak,
                        )
                        if nonfinite_streak >= 5:
                            raise FloatingPointError(
                                "Five consecutive non-finite training steps. "
                                "Lower the learning rate or check data/tokenizer alignment."
                            )
                        continue
                except torch.cuda.OutOfMemoryError as exc:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA out of memory during training. Reduce batch_size "
                        f"(currently {self.config.batch_size}), increase grad_accum_steps, "
                        "or reduce block_size. No silent fallback was applied."
                    ) from exc

                if self.state.step % self.config.log_interval == 0:
                    now = time.time()
                    elapsed = now - log_time
                    session_tokens = self.state.tokens_seen - log_tokens
                    lr = self.scheduler.get_last_lr()[0]
                    record = {
                        "step": self.state.step,
                        "train_loss": round(window.mean, 6),
                        "lr": lr,
                        "tokens_seen": self.state.tokens_seen,
                        "tokens_per_sec": round(tokens_per_second(session_tokens, elapsed), 1),
                        "elapsed_sec": round(now - start, 1),
                    }
                    self.logger.info(
                        "step %d/%d loss=%.4f lr=%.2e tok/s=%.0f tokens=%d",
                        self.state.step,
                        self.config.max_steps,
                        window.mean,
                        lr,
                        record["tokens_per_sec"],
                        self.state.tokens_seen,
                    )
                    if self.jsonl:
                        self.jsonl.log({"type": "train", **record})
                    log_tokens = self.state.tokens_seen
                    log_time = now

                if self.state.step % self.config.eval_interval == 0:
                    metrics = self.evaluate()
                    if metrics["val_loss"] is not None:
                        value = float(metrics["val_loss"])
                        self.state.val_losses.append(value)
                        if self.state.best_val_loss is None or value < self.state.best_val_loss:
                            self.state.best_val_loss = value
                            if self.config.save_best:
                                self.save("best.pt", best=True)
                                self.logger.info("New best validation loss: %.4f", value)
                    if self.jsonl:
                        self.jsonl.log({"type": "val", "step": self.state.step, **metrics})

                if (
                    self.state.step % self.config.checkpoint_interval == 0
                    or time.time() >= deadline
                ):
                    self._save_periodic()

            self._stop_reason = (
                "max_steps" if self.state.step >= self.config.max_steps else self._stop_reason
            )
        except KeyboardInterrupt:
            self._stop_reason = "interrupted"
            self.logger.warning("Training interrupted by user; saving checkpoint")
        except RuntimeError as exc:
            self._stop_reason = "error"
            self.logger.error("Training stopped: %s", exc)

        try:
            final_metrics = self.evaluate()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Final evaluation skipped: %s", exc)
            final_metrics = {"val_loss": None, "val_perplexity": None, "val_batches": 0}
        if final_metrics["val_loss"] is not None:
            value = float(final_metrics["val_loss"])
            if self.state.best_val_loss is None or value < self.state.best_val_loss:
                self.state.best_val_loss = value
                if self.config.save_best:
                    self.save("best.pt", best=True)
        self.save("last.pt")
        elapsed = time.time() - start
        report = {
            "stage": self.config.stage,
            "stop_reason": self._stop_reason,
            "steps": self.state.step,
            "max_steps": self.config.max_steps,
            "tokens_seen": self.state.tokens_seen,
            "supervised_tokens": self.state.supervised_tokens,
            "session_tokens": self.state.tokens_seen - self.session_start_tokens,
            "elapsed_sec": round(elapsed, 2),
            "tokens_per_sec": round(
                tokens_per_second(self.state.tokens_seen - self.session_start_tokens, elapsed), 1
            ),
            "best_val_loss": self.state.best_val_loss,
            "final_train_loss": self.state.train_losses[-1] if self.state.train_losses else None,
            "final_val_loss": final_metrics["val_loss"],
            "final_val_perplexity": final_metrics["val_perplexity"],
            "skipped_steps": self.state.skipped_steps,
            "resumed_from": self.state.resumed_from,
            "device": describe_device(self.device),
            "amp": {"enabled": self.amp_enabled, "dtype": str(self.amp_dtype)},
            "parameters": self._base_model().num_parameters()
            if hasattr(self._base_model(), "num_parameters")
            else None,
            "training_config": self.config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "data": self.data_meta,
            "train_loss_history": self.state.train_losses,
            "val_loss_history": self.state.val_losses,
        }
        write_metrics_summary(self.run_dir / "training_report.json", report)
        if self.jsonl:
            self.jsonl.log({"type": "final", **{k: v for k, v in report.items() if k != "train_loss_history"}})
        self.logger.info(
            "Training finished (%s): step=%d tokens=%d best_val=%s",
            self._stop_reason,
            self.state.step,
            self.state.tokens_seen,
            f"{self.state.best_val_loss:.4f}" if self.state.best_val_loss is not None else None,
        )
        return report

    def report_path(self) -> Path:
        return self.run_dir / "training_report.json"
