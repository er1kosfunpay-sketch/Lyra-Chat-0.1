"""Byte-level BPE tokenizer with stable role markers.

The tokenizer is trained once, saved next to the model, and loaded by ID.  Its
special-token order is fixed and persisted, so resuming training can never
silently reshuffle role markers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|turn_end|>",
)
SPECIAL_TOKEN_IDS: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
PAD_TOKEN = "<|pad|>"
UNK_TOKEN = "<|unk|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
TURN_END_TOKEN = "<|turn_end|>"
ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
TOKENIZER_VERSION = 1


def _trainer(vocab_size: int) -> trainers.BpeTrainer:
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="",
        end_of_word_suffix="",
    )


def build_tokenizer(vocab_size: int = 24000) -> Tokenizer:
    if vocab_size < len(SPECIAL_TOKENS) + 260:
        raise ValueError("vocab_size is too small for byte-level BPE")
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 24000,
) -> Tokenizer:
    tokenizer = build_tokenizer(vocab_size)
    tokenizer.train_from_iterator(texts, trainer=_trainer(vocab_size), length=None)
    _validate_special_tokens(tokenizer)
    return tokenizer


def _validate_special_tokens(tokenizer: Tokenizer) -> None:
    for token, expected in SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected:
            raise RuntimeError(
                f"Special token {token!r} has id {actual}, expected {expected}. "
                "The tokenizer is not compatible with the fixed conversation format."
            )
    if tokenizer.get_vocab_size() > 65536:
        raise RuntimeError("Tokenizer vocabulary is unexpectedly large")


@dataclass
class ConversationTokenizer:
    tokenizer: Tokenizer
    vocab_size: int
    special_tokens: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.tokenizer.token_to_id(PAD_TOKEN) is None:
            raise ValueError("Tokenizer does not contain the required special tokens")

    # ------------------------------------------------------------------ encode
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode() expects a string")
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def encode_text(self, text: str) -> list[int]:
        """Encode user-provided text without letting it inject role markers."""
        return self.encode(sanitize_user_text(text))

    def encode_texts(self, texts: Sequence[str]) -> list[list[int]]:
        """Batch-encode turn texts (much faster than one Python call per turn)."""
        sanitized = [sanitize_user_text(text) for text in texts]
        encodings = self.tokenizer.encode_batch(sanitized, add_special_tokens=False)
        return [encoding.ids for encoding in encodings]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise KeyError(f"Unknown special token: {token}")
        return value

    @property
    def pad_id(self) -> int:
        return self.token_id(PAD_TOKEN)

    @property
    def bos_id(self) -> int:
        return self.token_id(BOS_TOKEN)

    @property
    def eos_id(self) -> int:
        return self.token_id(EOS_TOKEN)

    @property
    def turn_end_id(self) -> int:
        return self.token_id(TURN_END_TOKEN)

    @property
    def user_id(self) -> int:
        return self.token_id(ROLE_TOKENS["user"])

    @property
    def assistant_id(self) -> int:
        return self.token_id(ROLE_TOKENS["assistant"])

    def role_id(self, role: str) -> int:
        token = ROLE_TOKENS.get(role)
        if token is None:
            raise ValueError(f"Unsupported role: {role!r}")
        return self.token_id(token)

    # ---------------------------------------------------------------- dialogue
    def encode_turn(self, role: str, text: str) -> list[int]:
        return [self.role_id(role), *self.encode_text(text), self.turn_end_id]

    def encode_dialogue(
        self,
        turns: Sequence[str],
        roles: Sequence[str],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        if len(turns) != len(roles):
            raise ValueError("turns and roles must have equal length")
        ids: list[int] = [self.bos_id] if add_bos else []
        for role, text in zip(roles, turns):
            ids.extend(self.encode_turn(role, text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode_turn(self, ids: Sequence[int]) -> str:
        values = [token for token in ids if token not in {self.pad_id, self.bos_id}]
        return self.decode(values, skip_special_tokens=True).strip()

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path, metadata: dict[str, object] | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tmp_path = directory / "tokenizer.json.tmp"
        self.tokenizer.save(str(tmp_path))
        os.replace(tmp_path, directory / "tokenizer.json")
        specials = {token: self.token_id(token) for token in SPECIAL_TOKENS}
        (directory / "special_tokens.json").write_text(
            json.dumps(specials, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        config = {
            "version": TOKENIZER_VERSION,
            "vocab_size": self.vocab_size,
            "special_tokens": list(SPECIAL_TOKENS),
            "byte_level_bpe": True,
            **(metadata or {}),
        }
        (directory / "tokenizer_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "ConversationTokenizer":
        directory = Path(directory)
        tokenizer_file = directory / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_file}. Train it first with "
                "scripts/train_tokenizer.py or set the correct tokenizer_dir."
            )
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        _validate_special_tokens(tokenizer)
        specials_path = directory / "special_tokens.json"
        expected = dict(SPECIAL_TOKEN_IDS)
        if specials_path.exists():
            stored = json.loads(specials_path.read_text(encoding="utf-8"))
            if stored != expected:
                raise RuntimeError(
                    "special_tokens.json does not match the fixed token order; "
                    "refusing to load an incompatible tokenizer"
                )
        return cls(tokenizer=tokenizer, vocab_size=tokenizer.get_vocab_size(), special_tokens=expected)

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "vocab": self.tokenizer.get_vocab(),
                "specials": dict(self.special_tokens),
                "version": TOKENIZER_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, object]:
        return {
            "type": "byte-level BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": dict(self.special_tokens),
            "fingerprint": self.fingerprint(),
        }


def sanitize_user_text(text: str) -> str:
    """Neutralize strings that the tokenizer would treat as control markers."""
    safe = str(text)
    for token in SPECIAL_TOKENS:
        if token in safe:
            safe = safe.replace(token, token.replace("<|", "< |"))
    return safe


def render_prompt_text(
    messages: Sequence[tuple[str, str]],
    add_bos: bool = True,
) -> str:
    """Human-readable prompt representation (tokenization is done separately)."""
    parts: list[str] = []
    if add_bos:
        parts.append(BOS_TOKEN)
    for role, text in messages:
        parts.append(ROLE_TOKENS.get(role, ROLE_TOKENS["user"]))
        parts.append(str(text).strip())
        parts.append(TURN_END_TOKEN)
    parts.append(ROLE_TOKENS["assistant"])
    return "".join(parts)


def iter_texts_from_dialogues(
    dialogues: Iterable[dict[str, object]], stride: int = 1
) -> Iterator[str]:
    """Yield turn text for tokenizer training, sampling every `stride` dialogue."""
    for index, dialogue in enumerate(dialogues):
        if index % stride:
            continue
        turns = dialogue.get("turns", [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if isinstance(turn, str) and turn.strip():
                yield turn.strip()
