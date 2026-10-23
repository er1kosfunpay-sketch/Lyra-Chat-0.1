from __future__ import annotations

import json
from pathlib import Path

from conversation_llm.config import DataConfig
from conversation_llm.data.dataset import (
    PretrainDataset,
    SFTDataset,
    tokenize_pretrain,
    tokenize_sft,
)


def _write_cleaned(tmp_path: Path) -> Path:
    cleaned = tmp_path / "artifacts" / "data" / "cleaned"
    cleaned.mkdir(parents=True)
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        rows = []
        for index in range(count):
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "turns": [
                        f"Привет, это диалог {split} {index}",
                        f"Привет! Отвечаю на диалог {split} {index}",
                        "Как дела?",
                        "Хорошо, спасибо!",
                    ],
                    "roles": ["user", "assistant", "user", "assistant"],
                    "roles_inferred": True,
                    "confidence": 0.6,
                }
            )
        (cleaned / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
    return cleaned


def test_tokenized_pretrain_and_sft_are_cached(tmp_path: Path, tokenizer) -> None:
    cleaned = _write_cleaned(tmp_path)
    config = DataConfig(
        source_url="x",
        cleaned_dir=str(cleaned),
        tokenized_dir=str(tmp_path / "tokenized"),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        reports_dir=str(tmp_path / "reports"),
        sft_max_len=64,
        context_turns=2,
    )
    pretrain = tokenize_pretrain(config, tmp_path, tokenizer, block_size=64)
    assert pretrain["splits"]["train"]["tokens"] > 64
    dataset = PretrainDataset(tmp_path / "tokenized" / "pretrain" / "train" / "tokens.bin", 64)
    assert len(dataset) > 0

    sft = tokenize_sft(config, tmp_path, tokenizer, block_size=64)
    meta = sft["splits"]["train"]
    assert meta["examples"] == 12  # two assistant turns in each of 6 dialogues
    sft_dataset = SFTDataset(tmp_path / "tokenized" / "sft" / "train")
    item = sft_dataset[0]
    supervised = item["labels"] != -100
    assert supervised.any()
    assert item["input_ids"][0].item() == tokenizer.bos_id
    assert item["input_ids"][supervised][-1].item() == tokenizer.turn_end_id
    assistant_positions = (item["input_ids"][:-1] == tokenizer.assistant_id).nonzero()
    first_target = int(supervised.nonzero()[0].item())
    assert any(int(position) < first_target for position in assistant_positions)
    for index in range(len(sft_dataset)):
        example = sft_dataset[index]
        assert example["input_ids"][0].item() == tokenizer.bos_id
        assert example["input_ids"][1].item() == tokenizer.user_id

    ids_mtime = (tmp_path / "tokenized" / "sft" / "train" / "ids.bin").stat().st_mtime_ns
    cached = tokenize_sft(config, tmp_path, tokenizer, block_size=64)
    assert cached["splits"]["train"]["examples"] == meta["examples"]
    assert (tmp_path / "tokenized" / "sft" / "train" / "ids.bin").stat().st_mtime_ns == ids_mtime
