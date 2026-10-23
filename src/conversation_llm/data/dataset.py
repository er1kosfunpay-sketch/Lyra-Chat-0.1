"""Tokenizer training plus tokenized pretraining/SFT datasets."""

from __future__ import annotations

import json
import os
from array import array
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from ..tokenizer import ConversationTokenizer, TURN_END_TOKEN, train_tokenizer
from ..utils.logging import setup_logger
from .manifest import (
    atomic_write_json,
    file_fingerprint,
    manifest_matches,
    read_json,
    stable_hash,
)
from .preprocessing import SPLITS, iter_clean_dialogues

LOGGER = setup_logger("conversation_llm.data.dataset")
TOKENIZATION_VERSION = 2


# --------------------------------------------------------------------- tokenizer
def train_tokenizer_from_corpus(
    config: DataConfig,
    root: str | Path,
    force: bool = False,
) -> ConversationTokenizer:
    root = Path(root)
    resolved = config.resolve(root)
    tokenizer_dir = resolved.tokenizer_dir
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    train_file = resolved.cleaned_dir / "train.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(
            f"Cleaned training corpus not found: {train_file}. Run scripts/prepare_data.py first."
        )
    data_report = read_json(resolved.cleaned_dir / "manifest.json", default={})
    token_count_chars = int(
        data_report.get("splits", {}).get("train", {}).get("chars", 0)
    )
    if token_count_chars <= 0:
        token_count_chars = train_file.stat().st_size
    sample_bytes = max(1, config.tokenizer_sample_mb) * 1_000_000
    stride = max(1, token_count_chars // sample_bytes)
    signature = stable_hash(
        {
            "version": TOKENIZATION_VERSION,
            "vocab_size": config.tokenizer_vocab_size,
            "sample_mb": config.tokenizer_sample_mb,
            "stride": stride,
            "seed": config.seed,
            "train": file_fingerprint(train_file),
        }
    )
    manifest_path = tokenizer_dir / "manifest.json"
    if (
        not force
        and manifest_matches(manifest_path, signature)
        and (tokenizer_dir / "tokenizer.json").exists()
    ):
        LOGGER.info("Using cached tokenizer: %s", tokenizer_dir)
        return ConversationTokenizer.load(tokenizer_dir)

    LOGGER.info(
        "Training byte-level BPE: vocab=%d, sample=%.0f MB, stride=%d",
        config.tokenizer_vocab_size,
        sample_bytes / 1e6,
        stride,
    )
    texts = _iter_tokenizer_texts(train_file, stride)
    tokenizer = train_tokenizer(texts, vocab_size=config.tokenizer_vocab_size)
    wrapper = ConversationTokenizer(
        tokenizer=tokenizer,
        vocab_size=tokenizer.get_vocab_size(),
        special_tokens={token: tokenizer.token_to_id(token) for token in _special_names()},
    )
    metadata = {
        "sample_mb": config.tokenizer_sample_mb,
        "stride": stride,
        "seed": config.seed,
        "trained_on": str(train_file),
        "data_signature": signature,
    }
    wrapper.save(tokenizer_dir, metadata=metadata)
    meta = {
        "signature": signature,
        "vocab_size": wrapper.vocab_size,
        "fingerprint": wrapper.fingerprint(),
        "sample_mb": config.tokenizer_sample_mb,
        "stride": stride,
        "seed": config.seed,
    }
    atomic_write_json(manifest_path, meta)
    atomic_write_json(
        resolved.reports_dir / "tokenizer_report.json",
        {**meta, "special_tokens": dict(wrapper.special_tokens)},
    )
    LOGGER.info("Tokenizer trained and saved to %s", tokenizer_dir)
    return wrapper


def _special_names() -> Sequence[str]:
    from ..tokenizer import SPECIAL_TOKENS

    return SPECIAL_TOKENS


def _iter_tokenizer_texts(path: Path, stride: int) -> Iterator[str]:
    for index, dialogue in enumerate(iter_clean_dialogues(path)):
        if index % stride:
            continue
        turns = dialogue.get("turns", [])
        for turn in turns:
            if isinstance(turn, str) and turn.strip():
                yield turn.strip()


# ------------------------------------------------------------------ tokenization
def _pretrain_meta_path(directory: Path) -> Path:
    return directory / "meta.json"


def tokenize_pretrain(
    config: DataConfig,
    root: str | Path,
    tokenizer: ConversationTokenizer,
    block_size: int,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    resolved = config.resolve(root)
    base = resolved.tokenized_dir / "pretrain"
    signature_base = {
        "version": TOKENIZATION_VERSION,
        "kind": "pretrain",
        "tokenizer": tokenizer.fingerprint(),
        "block_size": block_size,
        "data": read_json(resolved.cleaned_dir / "manifest.json", default={}).get("signature"),
        "config": config.to_dict(),
    }
    outputs: dict[str, Any] = {"splits": {}, "block_size": block_size}
    for split in SPLITS:
        directory = base / split
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = _pretrain_meta_path(directory)
        meta = read_json(meta_path, default={})
        source = resolved.cleaned_dir / f"{split}.jsonl"
        expected = stable_hash({**signature_base, "split": split, "source": file_fingerprint(source)})
        if not force and meta.get("signature") == expected and (directory / "tokens.bin").exists():
            outputs["splits"][split] = meta
            LOGGER.info("Pretrain %s cache is valid: %d tokens", split, meta["tokens"])
            continue

        tmp = directory / "tokens.bin.tmp"
        tokens = 0
        dialogues = 0
        iterator = iter_clean_dialogues(source)
        with tmp.open("wb") as fh:
            while True:
                chunk = list(islice(iterator, 1024))
                if not chunk:
                    break
                turn_texts = [text for dialogue in chunk for text in dialogue["turns"]]
                encoded = tokenizer.encode_texts(turn_texts)
                cursor = 0
                for dialogue in chunk:
                    ids = [tokenizer.bos_id]
                    for role in dialogue["roles"]:
                        ids.append(tokenizer.role_id(role))
                        ids.extend(encoded[cursor])
                        ids.append(tokenizer.turn_end_id)
                        cursor += 1
                    ids.append(tokenizer.eos_id)
                    np.asarray(ids, dtype=np.int32).tofile(fh)
                    tokens += len(ids)
                    dialogues += 1
                    if dialogues % 100_000 == 0:
                        LOGGER.info(
                            "Tokenized %s pretrain: %d dialogues, %d tokens",
                            split,
                            dialogues,
                            tokens,
                        )
        if tokens <= block_size + 1:
            raise RuntimeError(
                f"Split {split!r} has only {tokens} tokens; cannot create {block_size}-token blocks"
            )
        os.replace(tmp, directory / "tokens.bin")
        meta = {
            "signature": expected,
            "split": split,
            "tokens": tokens,
            "dialogues": dialogues,
            "block_size": block_size,
            "dtype": "int32",
            "tokenizer_fingerprint": tokenizer.fingerprint(),
        }
        atomic_write_json(meta_path, meta)
        outputs["splits"][split] = meta
        LOGGER.info("Pretrain %s: %d dialogues -> %d tokens", split, dialogues, tokens)

    outputs["signature"] = stable_hash({**signature_base, "splits": outputs["splits"]})
    atomic_write_json(base / "meta.json", outputs)
    report_path = resolved.reports_dir / "tokenization_report.json"
    report = read_json(report_path, default={})
    report["pretrain"] = outputs
    atomic_write_json(report_path, report)
    return outputs


def tokenize_sft(
    config: DataConfig,
    root: str | Path,
    tokenizer: ConversationTokenizer,
    block_size: int,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    resolved = config.resolve(root)
    base = resolved.tokenized_dir / "sft"
    base.mkdir(parents=True, exist_ok=True)
    max_len = min(config.sft_max_len, block_size)
    signature_base = {
        "version": TOKENIZATION_VERSION,
        "kind": "sft",
        "tokenizer": tokenizer.fingerprint(),
        "max_len": max_len,
        "context_turns": config.context_turns,
        "sft_format": 2,
        "data": read_json(resolved.cleaned_dir / "manifest.json", default={}).get("signature"),
        "config": config.to_dict(),
    }
    outputs: dict[str, Any] = {"splits": {}, "max_len": max_len}
    for split in SPLITS:
        directory = base / split
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "meta.json"
        meta = read_json(meta_path, default={})
        source = resolved.cleaned_dir / f"{split}.jsonl"
        expected = stable_hash({**signature_base, "split": split, "source": file_fingerprint(source)})
        if (
            not force
            and meta.get("signature") == expected
            and (directory / "ids.bin").exists()
            and (directory / "labels.bin").exists()
            and (directory / "offsets.npy").exists()
        ):
            outputs["splits"][split] = meta
            LOGGER.info("SFT %s cache is valid: %d examples", split, meta["examples"])
            continue

        tmp_ids = directory / "ids.bin.tmp"
        tmp_labels = directory / "labels.bin.tmp"
        # Compact append-only storage: a list of Python tuples would need
        # several GB for the full corpus (tens of millions of examples).
        offsets = array("q")
        token_total = 0
        response_tokens = 0
        examples = 0
        skipped = Counter()
        with tmp_ids.open("wb") as ids_file, tmp_labels.open("wb") as labels_file:
            iterator = iter_clean_dialogues(source)
            while True:
                chunk = list(islice(iterator, 512))
                if not chunk:
                    break
                turn_texts = [text for dialogue in chunk for text in dialogue["turns"]]
                encoded = tokenizer.encode_texts(turn_texts)
                cursor = 0
                for dialogue in chunk:
                    turns: list[str] = dialogue["turns"]
                    roles: list[str] = dialogue["roles"]
                    turn_ids: list[list[int]] = []
                    for role in roles:
                        turn_ids.append(
                            [tokenizer.role_id(role), *encoded[cursor], tokenizer.turn_end_id]
                        )
                        cursor += 1
                    for index, role in enumerate(roles):
                        if role != "assistant" or index == 0:
                            continue
                        if "user" not in roles[:index]:
                            continue
                        context_start = max(0, index - config.context_turns)
                        # Prompts must start with a user turn; otherwise a
                        # long dialogue would train the model on an
                        # assistant-first context that chat never produces.
                        while (
                            context_start < index
                            and roles[context_start] != "user"
                        ):
                            context_start += 1
                        target_ids = list(turn_ids[index][1:])
                        if len(target_ids) <= 1:
                            skipped["empty_target"] += 1
                            continue

                        prefix_len = 2 + sum(
                            len(turn_ids[position])
                            for position in range(context_start, index)
                        )
                        # Remove the oldest user/assistant pair until the
                        # example fits.  Keeping pairs preserves the
                        # user-first prompt structure.
                        while (
                            prefix_len + len(target_ids) > max_len
                            and context_start < index
                        ):
                            remove_end = min(index, context_start + 2)
                            prefix_len -= sum(
                                len(turn_ids[position])
                                for position in range(context_start, remove_end)
                            )
                            context_start = remove_end
                            skipped["context_truncated"] += 1
                        # Re-anchor after truncation: a prompt must still start
                        # with a user turn, never with an orphan assistant turn.
                        while context_start < index and roles[context_start] != "user":
                            context_start += 1
                        if context_start >= index:
                            skipped["no_user_context"] += 1
                            continue
                        prefix_len = 2 + sum(
                            len(turn_ids[position])
                            for position in range(context_start, index)
                        )
                        if prefix_len >= max_len - 2:
                            skipped["prompt_too_long"] += 1
                            continue
                        prompt_ids = [tokenizer.bos_id]
                        for position in range(context_start, index):
                            prompt_ids.extend(turn_ids[position])
                        prompt_ids.append(tokenizer.assistant_id)

                        budget = max_len - len(prompt_ids)
                        if len(target_ids) > budget:
                            target_ids = target_ids[: max(1, budget - 1)] + [
                                tokenizer.turn_end_id
                            ]
                            skipped["target_truncated"] += 1
                        if len(target_ids) <= 1:
                            skipped["target_too_short"] += 1
                            continue

                        ids = prompt_ids + target_ids
                        labels = [-100] * len(prompt_ids) + target_ids
                        start_offset = token_total
                        np.asarray(ids, dtype=np.int32).tofile(ids_file)
                        np.asarray(labels, dtype=np.int32).tofile(labels_file)
                        token_total += len(ids)
                        response_tokens += len(target_ids)
                        offsets.append(start_offset)
                        offsets.append(token_total)
                        examples += 1

        if examples == 0:
            raise RuntimeError(f"No SFT examples could be built for split {split!r}")
        os.replace(tmp_ids, directory / "ids.bin")
        os.replace(tmp_labels, directory / "labels.bin")
        np.save(
            directory / "offsets.npy",
            np.asarray(offsets, dtype=np.int64).reshape(-1, 2),
        )
        meta = {
            "signature": expected,
            "split": split,
            "examples": examples,
            "tokens": token_total,
            "response_tokens": response_tokens,
            "max_len": max_len,
            "skipped": dict(skipped),
            "tokenizer_fingerprint": tokenizer.fingerprint(),
        }
        atomic_write_json(meta_path, meta)
        outputs["splits"][split] = meta
        LOGGER.info(
            "SFT %s: %d examples, %d tokens (%d response), skipped=%s",
            split,
            examples,
            token_total,
            response_tokens,
            dict(skipped),
        )

    outputs["signature"] = stable_hash({**signature_base, "splits": outputs["splits"]})
    atomic_write_json(base / "meta.json", outputs)
    report_path = resolved.reports_dir / "tokenization_report.json"
    report = read_json(report_path, default={})
    report["sft"] = outputs
    atomic_write_json(report_path, report)
    return outputs


# ----------------------------------------------------------------------- datasets
class PretrainDataset(Dataset):
    """Contiguous next-token blocks over a packed token stream."""

    def __init__(self, tokens_path: str | Path, block_size: int) -> None:
        self.tokens_path = Path(tokens_path)
        self.block_size = int(block_size)
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Token stream not found: {self.tokens_path}")
        self.tokens = np.memmap(self.tokens_path, dtype=np.int32, mode="r")
        self.length = max(0, (len(self.tokens) - 1) // self.block_size)
        if self.length == 0:
            raise ValueError(
                f"Token stream {self.tokens_path} is shorter than block_size={block_size}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        start = index * self.block_size
        chunk = np.asarray(self.tokens[start : start + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": x, "labels": y}


class SFTDataset(Dataset):
    """Variable-length response-only supervision examples.

    `labels` are -100 for prompt tokens, so the cross-entropy is computed only
    on assistant response tokens and the turn-end marker.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        ids_path = self.directory / "ids.bin"
        labels_path = self.directory / "labels.bin"
        offsets_path = self.directory / "offsets.npy"
        if not (ids_path.exists() and labels_path.exists() and offsets_path.exists()):
            raise FileNotFoundError(f"Tokenized SFT dataset is incomplete: {self.directory}")
        self.ids = np.memmap(ids_path, dtype=np.int32, mode="r")
        self.labels = np.memmap(labels_path, dtype=np.int32, mode="r")
        self.offsets = np.load(offsets_path)
        if len(self.ids) != len(self.labels):
            raise RuntimeError("SFT ids and labels have different lengths")

    def __len__(self) -> int:
        return int(len(self.offsets))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start, end = (int(v) for v in self.offsets[index])
        return {
            "input_ids": torch.from_numpy(np.asarray(self.ids[start:end], dtype=np.int64)),
            "labels": torch.from_numpy(np.asarray(self.labels[start:end], dtype=np.int64)),
        }


def load_tokenized_meta(
    directory: str | Path, split: str = "train"
) -> dict[str, Any]:
    path = Path(directory) / split / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenized split metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
