from __future__ import annotations

import json
from pathlib import Path

import torch

from conversation_llm.config import TrainingConfig
from conversation_llm.data.batching import build_dataloader, make_sft_collator
from conversation_llm.data.dataset import PretrainDataset, SFTDataset
from conversation_llm.model import GPT
from conversation_llm.training.trainer import Trainer


def make_config(**kwargs) -> TrainingConfig:
    values = dict(
        seed=5,
        stage="sft",
        batch_size=2,
        grad_accum_steps=1,
        max_steps=2,
        max_minutes=1,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0.01,
        grad_clip=1.0,
        num_workers=0,
        amp="off",
        keep_last_checkpoints=2,
        stop_after_seconds=60,
        allow_cpu=True,
    )
    values.update(kwargs)
    return TrainingConfig(**values)


def make_loaders(tokenizer, config, sft_dir: Path):
    train_dataset = SFTDataset(sft_dir / "train")
    val_dataset = SFTDataset(sft_dir / "val")
    collate = make_sft_collator(tokenizer.pad_id)
    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        seed=config.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        seed=config.seed,
    )
    return train_loader, val_loader


def test_short_run_clamps_warmup() -> None:
    config = TrainingConfig(stage="sft", max_steps=100, warmup_steps=1)
    adjusted = config.with_overrides(max_steps=1)
    assert adjusted.max_steps == 1
    assert adjusted.warmup_steps == 0
    longer = config.with_overrides(max_steps=7)
    assert 0 <= longer.warmup_steps < longer.max_steps


def test_sft_dataset_labels_only_targets(tokenizer, tiny_sft_dir: Path) -> None:
    dataset = SFTDataset(tiny_sft_dir / "train")
    item = dataset[0]
    assert item["input_ids"].shape == item["labels"].shape
    supervised = item["labels"] != -100
    assert supervised.any()
    assert not supervised[0]
    assert item["input_ids"][supervised][-1].item() == tokenizer.turn_end_id
    loader = build_dataloader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=make_sft_collator(tokenizer.pad_id),
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()


def test_pretrain_dataset_shifts_targets(tokenizer, tiny_pretrain_dir: Path) -> None:
    dataset = PretrainDataset(tiny_pretrain_dir / "train" / "tokens.bin", block_size=64)
    item = dataset[0]
    assert item["input_ids"].shape == (64,)
    assert item["labels"].shape == (64,)
    assert torch.equal(item["input_ids"][1:], item["labels"][:-1])
    assert len(dataset) > 1


def test_one_training_step_updates_weights_and_checkpoints(
    tokenizer, model_config, tiny_sft_dir: Path, tmp_path: Path
) -> None:
    model = GPT(model_config)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    config = make_config()
    train_loader, val_loader = make_loaders(tokenizer, config, tiny_sft_dir)
    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        run_dir=tmp_path / "run",
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    report = trainer.train()
    assert report["steps"] == 2
    assert report["final_train_loss"] is not None
    assert torch.isfinite(torch.tensor(report["final_train_loss"]))
    assert report["final_val_loss"] is not None
    changed = any(
        not torch.equal(before[name], tensor) for name, tensor in model.state_dict().items()
    )
    assert changed
    checkpoints = tmp_path / "run" / "checkpoints"
    assert (checkpoints / "last.pt").exists()
    assert (checkpoints / "best.pt").exists()
    assert (tmp_path / "run" / "training_report.json").exists()
    assert (tmp_path / "run" / "train.jsonl").exists()


def test_resume_restores_step_optimizer_and_weights(
    tokenizer, model_config, tiny_sft_dir: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run-resume"
    config = make_config(max_steps=1)
    model = GPT(model_config)
    trainer = Trainer(
        model,
        model_config,
        config,
        *make_loaders(tokenizer, config, tiny_sft_dir),
        torch.device("cpu"),
        run_dir,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    report = trainer.train()
    assert report["steps"] == 1
    assert trainer.optimizer.state_dict()["state"], "optimizer state must not be empty"
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    saved_optimizer = trainer.optimizer.state_dict()

    config2 = make_config(max_steps=3)
    model2 = GPT(model_config)
    trainer2 = Trainer(
        model2,
        model_config,
        config2,
        *make_loaders(tokenizer, config2, tiny_sft_dir),
        torch.device("cpu"),
        run_dir,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    assert trainer2.resume("auto")
    assert trainer2.state.step == 1
    for key, value in saved.items():
        assert torch.equal(value, model2.state_dict()[key])
    restored = trainer2.optimizer.state_dict()["state"]
    assert restored.keys() == saved_optimizer["state"].keys()
    for key in restored:
        assert torch.equal(restored[key]["exp_avg"], saved_optimizer["state"][key]["exp_avg"])
        assert torch.equal(restored[key]["exp_avg_sq"], saved_optimizer["state"][key]["exp_avg_sq"])
    continued = trainer2.train()
    assert continued["steps"] >= 2


def test_resume_replays_exact_data_cursor(
    tokenizer, model_config, tiny_sft_dir: Path, tmp_path: Path
) -> None:
    """A mid-epoch checkpoint must continue with the next unseen batch."""
    config = make_config(max_steps=10, batch_size=2)
    run_dir = tmp_path / "run-cursor"
    trainer = Trainer(
        GPT(model_config),
        model_config,
        config,
        *make_loaders(tokenizer, config, tiny_sft_dir),
        torch.device("cpu"),
        run_dir,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    consumed = [trainer._next_batch(), trainer._next_batch()]  # noqa: SLF001
    trainer.save("cursor.pt")
    expected = trainer._next_batch()  # noqa: SLF001

    resumed = Trainer(
        GPT(model_config),
        model_config,
        config,
        *make_loaders(tokenizer, config, tiny_sft_dir),
        torch.device("cpu"),
        run_dir,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    assert resumed.resume("checkpoints/cursor.pt")
    actual = resumed._next_batch()  # noqa: SLF001
    assert torch.equal(expected["input_ids"], actual["input_ids"])
    assert torch.equal(expected["labels"], actual["labels"])
    assert consumed[0]["input_ids"].shape[0] == 2


def test_resume_rejects_wrong_tokenizer(
    tokenizer, model_config, tiny_sft_dir: Path, tmp_path: Path
) -> None:
    config = make_config()
    model = GPT(model_config)
    trainer = Trainer(
        model,
        model_config,
        config,
        *make_loaders(tokenizer, config, tiny_sft_dir),
        torch.device("cpu"),
        tmp_path / "run-wrong-tok",
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    trainer.save("last.pt")
    trainer2 = Trainer(
        GPT(model_config),
        model_config,
        config,
        *make_loaders(tokenizer, config, tiny_sft_dir),
        torch.device("cpu"),
        tmp_path / "run-wrong-tok",
        tokenizer_fingerprint="different",
    )
    try:
        trainer2.resume("auto")
    except RuntimeError as exc:
        assert "tokenizer" in str(exc).lower()
    else:
        raise AssertionError("resume should reject a different tokenizer")
