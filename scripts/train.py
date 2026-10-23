"""Train the causal language model (pretraining or response-only SFT)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conversation_llm.utils.environment import ensure_dependencies  # noqa: E402

ensure_dependencies(
    ["torch", "tokenizers", "numpy", "yaml", "tqdm", "safetensors", "datasets"]
)

import torch  # noqa: E402

from conversation_llm.config import (  # noqa: E402
    DataConfig,
    ModelConfig,
    TrainingConfig,
)
from conversation_llm.data.batching import (  # noqa: E402
    build_dataloader,
    make_sft_collator,
)
from conversation_llm.data.dataset import (  # noqa: E402
    PretrainDataset,
    SFTDataset,
    load_tokenized_meta,
)
from conversation_llm.model import GPT  # noqa: E402
from conversation_llm.tokenizer import ConversationTokenizer  # noqa: E402
from conversation_llm.training.checkpoint import find_latest_checkpoint  # noqa: E402
from conversation_llm.training.trainer import Trainer  # noqa: E402
from conversation_llm.utils.device import describe_device, resolve_device  # noqa: E402
from conversation_llm.utils.logging import setup_logger  # noqa: E402
from conversation_llm.utils.reproducibility import set_seed  # noqa: E402

LOGGER = setup_logger("conversation_llm.train")


def _run_dir(root: Path, stage: str, requested: str | None, resume: str) -> Path:
    if requested:
        requested_path = Path(requested)
        return requested_path if requested_path.is_absolute() else root / requested_path
    if resume not in {"none", "auto"}:
        checkpoint = Path(resume)
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent
        return checkpoint.parent
    if resume == "auto":
        base = root / "artifacts" / "runs"
        candidates = sorted(
            (
                path
                for path in base.glob(f"{stage}-*")
                if (path / "checkpoints" / "last.pt").exists()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        if candidates:
            return candidates[-1]
    return root / "artifacts" / "runs" / f"{stage}-{time.strftime('%Y%m%d-%H%M%S')}"


def _smoke_overrides(config: TrainingConfig) -> TrainingConfig:
    values = config.to_dict()
    values.update(
        {
            "max_steps": 4,
            "max_minutes": 10,
            "warmup_steps": 1,
            "eval_interval": 2,
            "checkpoint_interval": 2,
            "log_interval": 1,
            "eval_batches": 2,
            "batch_size": 2,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "stop_after_seconds": 300,
            "save_best": True,
            "amp": "auto",
        }
    )
    return TrainingConfig(**values)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Russian conversation model")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--stage", choices=["pretrain", "sft"], default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto | none | path/to/checkpoint (default: auto)",
    )
    parser.add_argument("--init-checkpoint", default=None, help="Load weights only (e.g. pretrain -> SFT)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true", help="Short end-to-end training run")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save VRAM (slower step, lower memory)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_config = ModelConfig.load(root / args.model_config)
    data_config = DataConfig.load(root / args.data_config)
    training_config = TrainingConfig.load(root / args.training_config)
    if args.stage:
        training_config = replace(training_config, stage=args.stage)
    if args.smoke:
        training_config = _smoke_overrides(training_config)

    overrides = {
        "max_steps": args.max_steps,
        "max_minutes": args.max_minutes,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "eval_interval": args.eval_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "seed": args.seed,
        "compile": args.compile or None,
    }
    if args.gradient_checkpointing:
        model_config = replace(model_config, gradient_checkpointing=True)

    training_config = training_config.with_overrides(**overrides)
    if os.name == "nt" and training_config.num_workers != 0 and args.num_workers is None:
        LOGGER.warning("Windows detected: using num_workers=0 for DataLoader stability")
        training_config = replace(training_config, num_workers=0)
    if training_config.warmup_steps >= training_config.max_steps:
        adjusted = min(training_config.warmup_steps, max(1, training_config.max_steps // 10))
        LOGGER.warning("warmup_steps >= max_steps; setting warmup_steps=%d", adjusted)
        training_config = replace(training_config, warmup_steps=adjusted)

    device = resolve_device(args.device)
    if device.type == "cpu" and not (args.allow_cpu or training_config.allow_cpu):
        raise SystemExit("CPU training refused. Pass --allow-cpu to override.")
    set_seed(training_config.seed, deterministic=False)

    run_dir = _run_dir(root, training_config.stage, args.run_dir, args.resume)
    if args.resume == "auto" and not (run_dir / "checkpoints").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Run directory: %s", run_dir)

    tokenizer_dir = data_config.resolve(root).tokenizer_dir
    if not (tokenizer_dir / "tokenizer.json").exists():
        raise FileNotFoundError(
            f"Tokenizer not found in {tokenizer_dir}. Run scripts/prepare_data.py first."
        )
    tokenizer = ConversationTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != model_config.vocab_size:
        LOGGER.warning(
            "Model vocab_size %d -> using tokenizer vocab_size %d",
            model_config.vocab_size,
            tokenizer.vocab_size,
        )
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    if training_config.stage == "pretrain":
        data_dir = Path(training_config.pretrain_data_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        train_meta = load_tokenized_meta(data_dir, "train")
        block_size = model_config.block_size
        train_dataset = PretrainDataset(data_dir / "train" / "tokens.bin", block_size)
        val_dataset = PretrainDataset(data_dir / "val" / "tokens.bin", block_size)
        collate = None
    else:
        data_dir = Path(training_config.sft_data_dir)
        if not data_dir.is_absolute():
            data_dir = root / data_dir
        train_meta = load_tokenized_meta(data_dir, "train")
        if train_meta["max_len"] > model_config.block_size:
            raise ValueError(
                f"SFT examples are up to {train_meta['max_len']} tokens but the model "
                f"block_size is {model_config.block_size}"
            )
        train_dataset = SFTDataset(data_dir / "train")
        val_dataset = SFTDataset(data_dir / "val")
        collate = make_sft_collator(tokenizer.pad_id)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=training_config.num_workers,
        prefetch_factor=training_config.prefetch_factor,
        seed=training_config.seed,
        pin_memory=device.type == "cuda",
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=max(1, min(4, training_config.batch_size)),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        seed=training_config.seed,
        pin_memory=device.type == "cuda",
    )

    LOGGER.info("Device: %s", json.dumps(describe_device(device), ensure_ascii=False))
    LOGGER.info(
        "Data: stage=%s train_examples=%d val_examples=%d train_tokens=%s",
        training_config.stage,
        len(train_dataset),
        len(val_dataset),
        f"{train_meta.get('tokens', 0):,}",
    )
    model = GPT(model_config)
    model.to(device)
    LOGGER.info("Model parameters: %s", f"{model.num_parameters():,}")

    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = root / init_path
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        payload = torch.load(init_path, map_location="cpu", weights_only=False)
        if payload.get("tokenizer_fingerprint") not in {None, tokenizer.fingerprint()}:
            raise RuntimeError("Initialization checkpoint uses a different tokenizer")
        model.load_state_dict(payload["model"], strict=True)
        LOGGER.info("Initialized weights from %s (weights only)", init_path)

    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        tokenizer_fingerprint=tokenizer.fingerprint(),
        data_meta={"stage": training_config.stage, "train": train_meta},
    )

    resumed = False
    if args.resume != "none":
        resume_arg = args.resume
        if args.resume not in {"auto"}:
            resume_path = Path(args.resume)
            if not resume_path.is_absolute():
                resume_path = root / resume_path
            resume_arg = str(resume_path)
        has_checkpoint = (
            find_latest_checkpoint(run_dir / training_config.checkpoint_dir) is not None
            if args.resume == "auto"
            else True
        )
        if has_checkpoint:
            resumed = trainer.resume(resume_arg)
    report = trainer.train()
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "resumed": resumed,
                "stop_reason": report["stop_reason"],
                "steps": report["steps"],
                "tokens_seen": report["tokens_seen"],
                "best_val_loss": report["best_val_loss"],
                "final_train_loss": report["final_train_loss"],
                "final_val_loss": report["final_val_loss"],
                "report": str(trainer.report_path()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
