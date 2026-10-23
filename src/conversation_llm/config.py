"""Typed configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _filtered(cls: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unknown keys instead of silently ignoring a typo in a config."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return dict(payload)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


@dataclass
class ModelConfig:
    vocab_size: int = 24000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    tie_embeddings: bool = True
    init_std: float = 0.02
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < 256:
            raise ValueError("vocab_size is implausibly small")
        if self.n_embd % self.n_head:
            raise ValueError(
                f"n_embd={self.n_embd} must be divisible by n_head={self.n_head}"
            )
        if self.n_layer < 1 or self.n_head < 1 or self.block_size < 2:
            raise ValueError("n_layer, n_head and block_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.init_std < 1.0:
            raise ValueError("init_std must be in (0, 1)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig":
        return cls(**_filtered(cls, payload))

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        raw = load_yaml(path)
        return cls.from_dict(raw.get("model", raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump({"model": self.to_dict()}, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


@dataclass
class DataConfig:
    source_url: str
    source_name: str = "Den4ikAI/russian_dialogues_2"
    source_license: str = "MIT"
    raw_path: str = "data/raw/dataset.jsonl.gz"
    cleaned_dir: str = "artifacts/data/cleaned"
    tokenized_dir: str = "artifacts/data/tokenized"
    tokenizer_dir: str = "artifacts/tokenizer"
    reports_dir: str = "reports"
    seed: int = 42
    split_fractions: tuple[float, float, float] = (0.98, 0.01, 0.01)
    min_turns: int = 2
    max_turns: int = 24
    max_turn_chars: int = 600
    max_dialogue_chars: int = 4000
    min_letters_per_turn: int = 2
    near_duplicate_filter: bool = True
    tokenizer_vocab_size: int = 24000
    tokenizer_sample_mb: int = 160
    context_turns: int = 8
    sft_max_len: int = 1024
    max_dialogues: int | None = None

    def __post_init__(self) -> None:
        self.split_fractions = tuple(float(x) for x in self.split_fractions)  # type: ignore[assignment]
        if len(self.split_fractions) != 3 or abs(sum(self.split_fractions) - 1.0) > 1e-6:
            raise ValueError("split_fractions must contain three values summing to 1")
        if any(x <= 0 for x in self.split_fractions):
            raise ValueError("every split must be non-empty")
        if self.context_turns < 1:
            raise ValueError("context_turns must be >= 1")
        if self.sft_max_len < 32:
            raise ValueError("sft_max_len is too small")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataConfig":
        return cls(**_filtered(cls, payload))

    @classmethod
    def load(cls, path: str | Path) -> "DataConfig":
        raw = load_yaml(path)
        return cls.from_dict(raw.get("data", raw))

    def resolve(self, root: str | Path = PROJECT_ROOT) -> "ResolvedDataConfig":
        root = Path(root)
        return ResolvedDataConfig(
            raw_path=root / self.raw_path,
            cleaned_dir=root / self.cleaned_dir,
            tokenized_dir=root / self.tokenized_dir,
            tokenizer_dir=root / self.tokenizer_dir,
            reports_dir=root / self.reports_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["split_fractions"] = list(self.split_fractions)
        return out


@dataclass
class ResolvedDataConfig:
    raw_path: Path
    cleaned_dir: Path
    tokenized_dir: Path
    tokenizer_dir: Path
    reports_dir: Path


@dataclass
class TrainingConfig:
    seed: int = 42
    stage: str = "sft"
    batch_size: int = 6
    grad_accum_steps: int = 10
    max_steps: int = 20000
    max_minutes: float = 660.0
    eval_interval: int = 250
    eval_batches: int = 40
    checkpoint_interval: int = 250
    log_interval: int = 10
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    num_workers: int = 2
    prefetch_factor: int = 4
    amp: str = "auto"
    compile: bool = False
    run_dir: str = "artifacts/runs"
    checkpoint_dir: str = "checkpoints"
    keep_last_checkpoints: int = 3
    save_best: bool = True
    log_jsonl: bool = True
    pretrain_data_dir: str = "artifacts/data/tokenized/pretrain"
    sft_data_dir: str = "artifacts/data/tokenized/sft"
    stop_after_seconds: float | None = None
    allow_cpu: bool = True

    def __post_init__(self) -> None:
        if self.stage not in {"pretrain", "sft"}:
            raise ValueError("stage must be 'pretrain' or 'sft'")
        if self.amp not in {"auto", "bf16", "fp16", "off"}:
            raise ValueError("amp must be auto, bf16, fp16 or off")
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.eval_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("intervals must be >= 1")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingConfig":
        return cls(**_filtered(cls, payload))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        raw = load_yaml(path)
        return cls.from_dict(raw.get("training", raw))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> "TrainingConfig":
        payload = self.to_dict()
        for key, value in overrides.items():
            if value is not None:
                if key not in payload:
                    raise ValueError(f"Unknown training override: {key}")
                payload[key] = value
        # A short run (for example --max-steps 100 with the default 500-step
        # warmup) must stay runnable.  Train scripts report this adjustment;
        # direct TrainingConfig(...) construction still validates strictly.
        if payload["warmup_steps"] >= payload["max_steps"]:
            payload["warmup_steps"] = max(
                0, min(payload["warmup_steps"], payload["max_steps"] // 10)
            )
        return TrainingConfig(**payload)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 80
    min_new_tokens: int = 1
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    context_tokens: int = 900

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.context_tokens < 32:
            raise ValueError("context_tokens is too small")

    @classmethod
    def load(cls, path: str | Path) -> "GenerationConfig":
        raw = load_yaml(path)
        return cls.from_dict(raw.get("generation", raw))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationConfig":
        return cls(**_filtered(cls, payload))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(
                {"generation": self.to_dict()}, sort_keys=False, allow_unicode=True
            ),
            encoding="utf-8",
        )


def load_all_configs(
    root: str | Path = PROJECT_ROOT,
) -> tuple[ModelConfig, DataConfig, TrainingConfig]:
    root = Path(root)
    return (
        ModelConfig.load(root / "configs" / "model.yaml"),
        DataConfig.load(root / "configs" / "data.yaml"),
        TrainingConfig.load(root / "configs" / "training.yaml"),
    )
