"""Local text generation with KV cache, sampling controls and stop tokens."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ..config import GenerationConfig, ModelConfig
from ..model import GPT
from ..tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TURN_END_TOKEN,
    ConversationTokenizer,
    ROLE_TOKENS,
)
from ..utils.device import resolve_device


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    stopped: bool
    stop_reason: str
    prompt_tokens: int
    generated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
        }


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    tokenizer_dir: str | Path,
    device: str | torch.device = "auto",
) -> tuple[GPT, ConversationTokenizer, ModelConfig, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {checkpoint_path}. Train or export the model first."
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Could not read model checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"{checkpoint_path} does not contain a 'model' state dict")
    model_config = ModelConfig.from_dict(payload["model_config"])
    tokenizer = ConversationTokenizer.load(tokenizer_dir)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match "
            f"model vocab_size={model_config.vocab_size}"
        )
    expected_fingerprint = payload.get("tokenizer_fingerprint")
    if expected_fingerprint and expected_fingerprint != tokenizer.fingerprint():
        raise RuntimeError(
            "Tokenizer files do not match the tokenizer used for training. "
            "Use the tokenizer directory saved with this checkpoint."
        )
    model = GPT(model_config)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Weights are incompatible with the architecture config: {exc}"
        ) from exc
    model.eval()
    target = resolve_device(str(device))
    model.to(target)
    metadata = {
        "step": payload.get("step"),
        "tokens_seen": payload.get("tokens_seen"),
        "best_val_loss": payload.get("best_val_loss"),
        "model_config": model_config.to_dict(),
        "tokenizer_fingerprint": tokenizer.fingerprint(),
    }
    return model, tokenizer, model_config, metadata


def load_export_directory(
    directory: str | Path,
    device: str | torch.device = "auto",
    dtype: str = "float32",
) -> tuple[GPT, ConversationTokenizer, ModelConfig, dict[str, Any]]:
    """Load the self-contained export produced by scripts/export_model.py."""
    directory = Path(directory)
    config_path = directory / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found in export: {directory}")
    model_config = ModelConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    tokenizer = ConversationTokenizer.load(directory / "tokenizer")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise RuntimeError("Exported tokenizer and model config disagree on vocab_size")
    model = GPT(model_config)
    weights_path = directory / "model.safetensors"
    if weights_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("safetensors is required to load this export") from exc
        state = load_file(str(weights_path), device="cpu")
    else:
        fallback = directory / "weights.pt"
        if not fallback.exists():
            raise FileNotFoundError("Neither model.safetensors nor weights.pt found in export")
        payload = torch.load(fallback, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    cast = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype not in cast:
        raise ValueError(f"Unsupported dtype: {dtype}")
    model.to(cast[dtype])
    model.eval()
    target = resolve_device(str(device))
    model.to(target)
    info_path = directory / "export_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return model, tokenizer, model_config, info


class ChatGenerator:
    def __init__(
        self,
        model: GPT,
        tokenizer: ConversationTokenizer,
        config: GenerationConfig | None = None,
        device: str | torch.device = "auto",
        max_history_turns: int = 40,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = resolve_device(str(device))
        self.model.to(self.device)
        self.model.eval()
        self.history: list[tuple[str, str]] = []
        self.max_history_turns = max_history_turns
        self.system_prompt = system_prompt
        self._stop_tokens = {
            self.tokenizer.token_id(token): token
            for token in (
                TURN_END_TOKEN,
                EOS_TOKEN,
                BOS_TOKEN,
                PAD_TOKEN,
                ROLE_TOKENS["user"],
                ROLE_TOKENS["assistant"],
                ROLE_TOKENS["system"],
            )
        }

    # ------------------------------------------------------------------ prompt
    def _turn_tokens(self, role: str, text: str) -> list[int]:
        return self.tokenizer.encode_turn(role, text)

    def build_prompt_ids(
        self,
        history: Sequence[tuple[str, str]] | None = None,
        max_new_tokens: int | None = None,
    ) -> list[int]:
        history = list(history if history is not None else self.history)
        if not history:
            raise ValueError("Cannot build a prompt from empty history")
        if history[-1][0] != "user":
            raise ValueError("The final history turn must be from the user")

        # Системная инструкция идёт первой — она говорит модели быть полезным
        # ассистентом, а не участником случайного интернет-чата.
        turns: list[tuple[str, str]] = []
        if self.system_prompt:
            turns.append(("system", self.system_prompt))
        turns.extend(history)

        budget = min(
            self.config.context_tokens,
            self.model.config.block_size - (max_new_tokens or self.config.max_new_tokens),
        )
        if budget < 16:
            raise ValueError("Context budget is too small; reduce max_new_tokens")
        encoded = [(role, self._turn_tokens(role, text)) for role, text in turns]
        # Find the longest suffix beginning with a system/user turn that fits.
        chosen: list[int] | None = None
        for start in range(len(encoded)):
            if encoded[start][0] not in ("system", "user"):
                continue
            candidate = [self.tokenizer.bos_id]
            for _, ids in encoded[start:]:
                candidate.extend(ids)
            candidate.append(self.tokenizer.assistant_id)
            if len(candidate) <= budget:
                chosen = candidate
                break
        if chosen is None:
            last_role, last_ids = encoded[-1]
            room = budget - 4
            if room < 1:
                raise ValueError("Current message is too long for the model context")
            content = last_ids[1:-1]
            chosen = [
                self.tokenizer.bos_id,
                self.tokenizer.role_id(last_role),
                *content[-room:],
                self.tokenizer.turn_end_id,
                self.tokenizer.assistant_id,
            ]
        return chosen

    # ---------------------------------------------------------------- sampling
    def _apply_repetition_penalty(self, logits: torch.Tensor, token_ids: Sequence[int]) -> None:
        penalty = self.config.repetition_penalty
        if penalty == 1.0 or not token_ids:
            return
        values = torch.tensor(sorted(set(token_ids)), device=logits.device, dtype=torch.long)
        selected = logits[values]
        adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
        logits[values] = adjusted

    def _banned_tokens(self, prefix: Sequence[int], n: int) -> set[int]:
        if n < 2 or len(prefix) < n:
            return set()
        tail = tuple(prefix[-(n - 1) :])
        banned: set[int] = set()
        for index in range(len(prefix) - n + 1):
            if tuple(prefix[index : index + n - 1]) == tail:
                banned.add(prefix[index + n - 1])
        return banned

    def _filter_logits(
        self,
        logits: torch.Tensor,
        prefix: Sequence[int],
        step: int,
    ) -> torch.Tensor:
        if step < self.config.min_new_tokens:
            for token_id in self._stop_tokens:
                logits[token_id] = float("-inf")
        banned = self._banned_tokens(prefix, self.config.no_repeat_ngram_size)
        if banned:
            ids = torch.tensor(sorted(banned), device=logits.device, dtype=torch.long)
            logits[ids] = float("-inf")

        if self.config.top_k and self.config.top_k < logits.numel():
            values, indices = torch.topk(logits, self.config.top_k)
            filtered = torch.full_like(logits, float("-inf"))
            filtered[indices] = values
            logits = filtered

        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative - probs > self.config.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits[sorted_indices] = sorted_logits
        return logits

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: Sequence[int],
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        cfg = config or self.config
        prompt = [int(v) for v in prompt_ids]
        block = self.model.config.block_size
        if len(prompt) >= block:
            raise ValueError("Prompt is longer than the model context window")
        max_new = min(cfg.max_new_tokens, block - len(prompt))
        if max_new < 1:
            raise ValueError("No room left for generation in the context window")

        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        output = self.model(input_ids=input_ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :].float()
        generated: list[int] = []
        prefix = list(prompt)
        stopped = False
        stop_reason = "max_new_tokens"

        for step in range(max_new):
            scores = logits[0].clone()
            self._apply_repetition_penalty(scores, prefix)
            scores = self._filter_logits(scores, prefix, step)
            if not torch.isfinite(scores).any():
                stop_reason = "all_tokens_masked"
                break

            if cfg.do_sample and cfg.temperature > 0:
                scores = scores / cfg.temperature
                probs = torch.softmax(scores, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                token = int(torch.argmax(scores).item())

            if token in self._stop_tokens and step >= cfg.min_new_tokens:
                stopped = True
                stop_reason = self._stop_tokens[token]
                break
            generated.append(token)
            prefix.append(token)
            if len(generated) >= max_new:
                break

            next_input = torch.tensor([[token]], dtype=torch.long, device=self.device)
            output = self.model(
                input_ids=next_input,
                past_key_values=past,
                use_cache=True,
            )
            past = output.past_key_values
            logits = output.logits[:, -1, :].float()

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            token_ids=generated,
            stopped=stopped,
            stop_reason=stop_reason,
            prompt_tokens=len(prompt),
            generated_tokens=len(generated),
        )

    # -------------------------------------------------------------------- chat
    def chat(
        self,
        user_text: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        text = str(user_text).strip()
        if not text:
            raise ValueError("User message is empty")
        pending = [*self.history, ("user", text)]
        prompt = self.build_prompt_ids(
            pending, max_new_tokens=(config or self.config).max_new_tokens
        )
        result = self.generate(prompt, config=config)
        self.history = pending
        if result.text:
            self.history.append(("assistant", result.text))
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns :]
            if self.history and self.history[0][0] != "user":
                self.history = self.history[1:]
        return result

    def reset(self) -> None:
        self.history.clear()

    def set_history(self, history: Sequence[tuple[str, str]]) -> None:
        allowed = {"user", "assistant"}
        for role, _ in history:
            if role not in allowed:
                raise ValueError(f"History role must be user/assistant, got {role!r}")
        self.history = [(str(role), str(text)) for role, text in history]

    def history_as_text(self) -> str:
        if not self.history:
            return "(история пуста)"
        lines = []
        for role, text in self.history:
            label = "Вы" if role == "user" else "Модель"
            lines.append(f"{label}: {text}")
        return "\n".join(lines)
