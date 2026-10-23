from __future__ import annotations

from pathlib import Path

import torch

from conversation_llm.config import TrainingConfig
from conversation_llm.model import GPT
from conversation_llm.training.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)


def test_weights_are_identical_after_save_load(
    tokenizer, model_config, tmp_path: Path
) -> None:
    torch.manual_seed(11)
    model = GPT(model_config).eval()
    ids = torch.randint(0, model_config.vocab_size, (2, 9))
    with torch.no_grad():
        expected = model(ids).logits

    path = tmp_path / "checkpoints" / "best.pt"
    save_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        step=10,
        tokens_seen=999,
        model_config=model_config,
        training_config=TrainingConfig(max_steps=20, warmup_steps=1),
        best_val_loss=3.25,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    loaded = GPT(model_config).eval()
    payload = load_checkpoint(path, loaded, map_location="cpu")
    assert payload["step"] == 10
    assert payload["tokens_seen"] == 999
    assert payload["best_val_loss"] == 3.25
    with torch.no_grad():
        actual = loaded(ids).logits
    assert torch.equal(expected, actual)


def test_checkpoint_rotation_and_latest(tokenizer, model_config, tmp_path: Path) -> None:
    model = GPT(model_config)
    directory = tmp_path / "ckpt"
    for step in (1, 2, 3, 4):
        save_checkpoint(
            directory / f"step_{step:07d}.pt",
            model,
            optimizer=None,
            scheduler=None,
            step=step,
            tokens_seen=step * 10,
            model_config=model_config,
            training_config=TrainingConfig(max_steps=10, warmup_steps=1),
        )
    assert find_latest_checkpoint(directory).name == "step_0000004.pt"
    removed = prune_checkpoints(directory, keep=2)
    assert len(removed) == 2
    assert find_latest_checkpoint(directory).name == "step_0000004.pt"


def test_load_missing_checkpoint_message(tmp_path: Path) -> None:
    model = GPT.__new__(GPT)
    try:
        load_checkpoint(tmp_path / "missing.pt", model)
    except FileNotFoundError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
