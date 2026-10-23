"""Shared fixtures for fast CPU-only tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conversation_llm.config import ModelConfig  # noqa: E402
from conversation_llm.tokenizer import (  # noqa: E402
    SPECIAL_TOKENS,
    ConversationTokenizer,
    train_tokenizer,
)

CORPUS = [
    "Привет! Как дела?",
    "Нормально, а у тебя?",
    "Всё хорошо, спасибо 😊",
    "Что делаешь сегодня вечером?",
    "Смотрю сериал и пью чай.",
    "Здарова, чё как?",
    "Да нормально, работаю.",
    "Hello world 123, mixed text.",
    "Очень длинноесловобезпробеловдляпроверкитокенизации",
    "У меня сегодня ужасный день...",
    "Не переживай, всё наладится!",
    "Как тебя зовут?",
    "Меня зовут Аня.",
    "Кстати, ты любишь кофе?",
    "Обожаю кофе, особенно утром.",
]


@pytest.fixture(scope="session")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> ConversationTokenizer:
    directory = tmp_path_factory.mktemp("tokenizer")
    texts = [text for _ in range(80) for text in CORPUS]
    raw = train_tokenizer(iter(texts), vocab_size=512)
    wrapper = ConversationTokenizer(
        raw,
        raw.get_vocab_size(),
        {token: raw.token_to_id(token) for token in SPECIAL_TOKENS},
    )
    wrapper.save(directory)
    return ConversationTokenizer.load(directory)


@pytest.fixture()
def model_config(tokenizer: ConversationTokenizer) -> ModelConfig:
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=64,
        dropout=0.0,
    )


@pytest.fixture()
def tiny_sft_dir(tmp_path: Path, tokenizer: ConversationTokenizer) -> Path:
    directory = tmp_path / "sft" / "train"
    directory.mkdir(parents=True)
    ids_path = directory / "ids.bin"
    labels_path = directory / "labels.bin"
    offsets: list[tuple[int, int]] = []
    total = 0
    with ids_path.open("wb") as ids_file, labels_path.open("wb") as labels_file:
        for index in range(16):
            prompt = [tokenizer.bos_id]
            prompt += tokenizer.encode_turn("user", f"Привет, это сообщение номер {index}")
            prompt.append(tokenizer.assistant_id)
            target = tokenizer.encode_text(f"Ответ номер {index}, всё хорошо.") + [tokenizer.turn_end_id]
            ids = prompt + target
            labels = [-100] * len(prompt) + target
            np.asarray(ids, dtype=np.int32).tofile(ids_file)
            np.asarray(labels, dtype=np.int32).tofile(labels_file)
            offsets.append((total, total + len(ids)))
            total += len(ids)
    np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    (directory / "meta.json").write_text(
        json.dumps({"examples": 16, "tokens": total, "max_len": 64}),
        encoding="utf-8",
    )
    val = tmp_path / "sft" / "val"
    val.mkdir()
    for name in ("ids.bin", "labels.bin", "offsets.npy"):
        (val / name).write_bytes((directory / name).read_bytes())
    (val / "meta.json").write_text(
        json.dumps({"examples": 16, "tokens": total, "max_len": 64}), encoding="utf-8"
    )
    return tmp_path / "sft"


@pytest.fixture()
def tiny_pretrain_dir(tmp_path: Path, tokenizer: ConversationTokenizer) -> Path:
    tokens: list[int] = []
    for index in range(40):
        tokens.extend(
            tokenizer.encode_dialogue(
                [f"Привет {index}", f"Привет, как дела {index}?"],
                ["user", "assistant"],
            )
        )
    directory = tmp_path / "pretrain"
    for split in ("train", "val"):
        split_dir = directory / split
        split_dir.mkdir(parents=True)
        np.asarray(tokens, dtype=np.int32).tofile(split_dir / "tokens.bin")
        (split_dir / "meta.json").write_text(
            json.dumps({"tokens": len(tokens), "block_size": 64}), encoding="utf-8"
        )
    return directory


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path, tokenizer: ConversationTokenizer, model_config: ModelConfig) -> Path:
    from conversation_llm.model import GPT
    from conversation_llm.training.checkpoint import save_checkpoint
    from conversation_llm.config import TrainingConfig

    torch.manual_seed(0)
    model = GPT(model_config)
    path = tmp_path / "checkpoints" / "best.pt"
    save_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        step=3,
        tokens_seen=1234,
        model_config=model_config,
        training_config=TrainingConfig(stage="sft", max_steps=10, warmup_steps=1),
        best_val_loss=4.5,
        tokenizer_fingerprint=tokenizer.fingerprint(),
    )
    return path
