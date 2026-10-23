from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from conversation_llm.config import GenerationConfig
from conversation_llm.inference.chat import main as chat_main
from conversation_llm.inference.generator import (
    ChatGenerator,
    load_model_from_checkpoint,
)
from conversation_llm.model import CausalLMOutput
from conversation_llm.training.checkpoint import save_checkpoint


class StubModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer,
        first_token: int,
        second_token: int,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(block_size=block_size)
        self.tokenizer = tokenizer
        self.first_token = first_token
        self.second_token = second_token
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
    ):
        batch, length = input_ids.shape
        past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
        logits = torch.full((batch, length, self.tokenizer.vocab_size), -20.0)
        last = int(input_ids[0, -1])
        chosen = self.first_token if last == self.tokenizer.assistant_id else self.second_token
        logits[:, -1, chosen] = 20.0
        cache = (
            (
                torch.zeros(batch, 1, past_length + length, 1),
                torch.zeros(batch, 1, past_length + length, 1),
            ),
        )
        return CausalLMOutput(logits=logits, past_key_values=cache if use_cache else None)


def make_generator(tokenizer, first: int, second: int, **config) -> ChatGenerator:
    model = StubModel(tokenizer, first, second)
    values = {"do_sample": False, "max_new_tokens": 8, "min_new_tokens": 1, "no_repeat_ngram_size": 0}
    values.update(config)
    return ChatGenerator(model, tokenizer, config=GenerationConfig(**values), device="cpu")


def test_generation_stops_on_turn_end(tokenizer) -> None:
    text_id = tokenizer.encode("Привет")[0]
    generator = make_generator(tokenizer, text_id, tokenizer.turn_end_id)
    prompt = [tokenizer.bos_id, tokenizer.user_id] + tokenizer.encode_text("Привет") + [
        tokenizer.turn_end_id,
        tokenizer.assistant_id,
    ]
    result = generator.generate(prompt)
    assert result.stopped
    assert result.stop_reason == "<|turn_end|>"
    assert result.generated_tokens == 1
    assert result.text


def test_generation_stops_on_user_marker(tokenizer) -> None:
    text_id = tokenizer.encode("ответ")[0]
    generator = make_generator(tokenizer, text_id, tokenizer.user_id)
    result = generator.chat("Привет")
    assert result.stopped
    assert result.stop_reason == "<|user|>"
    assert tokenizer.user_id not in result.token_ids


def test_generation_respects_max_new_tokens(tokenizer) -> None:
    text_id = tokenizer.encode("да")[0]
    generator = make_generator(
        tokenizer, text_id, text_id, max_new_tokens=3, min_new_tokens=0
    )
    result = generator.chat("Привет")
    assert not result.stopped
    assert result.stop_reason == "max_new_tokens"
    assert result.generated_tokens == 3


def test_empty_response_does_not_corrupt_history(tokenizer) -> None:
    generator = make_generator(
        tokenizer, tokenizer.turn_end_id, tokenizer.turn_end_id, min_new_tokens=0
    )
    result = generator.chat("Привет")
    assert result.text == ""
    assert generator.history == [("user", "Привет")]
    generator.reset()
    assert generator.history == []


def test_history_is_kept_and_prompt_is_trimmed(tokenizer) -> None:
    text_id = tokenizer.encode("ага")[0]
    generator = make_generator(tokenizer, text_id, tokenizer.turn_end_id, max_new_tokens=2)
    generator.chat("Привет")
    generator.chat("Как дела?")
    assert len(generator.history) >= 3
    assert generator.history[-1][0] == "assistant"
    long_history = [
        ("user" if index % 2 == 0 else "assistant", f"сообщение номер {index}")
        for index in range(29)
    ]
    generator.config.context_tokens = 40
    prompt = generator.build_prompt_ids(long_history, max_new_tokens=8)
    assert len(prompt) <= 40
    assert prompt[0] == tokenizer.bos_id
    assert prompt[-1] == tokenizer.assistant_id

    generator.config.context_tokens = 40
    huge = generator.build_prompt_ids([("user", "очень длинное сообщение " * 40)], max_new_tokens=8)
    assert len(huge) <= 40
    assert huge[0] == tokenizer.bos_id
    assert huge[-1] == tokenizer.assistant_id


def test_real_checkpoint_load_and_chat(
    tokenizer, model_config, tiny_checkpoint: Path, tmp_path: Path
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer.save(tokenizer_dir)
    model, loaded_tokenizer, config, info = load_model_from_checkpoint(
        tiny_checkpoint, tokenizer_dir, device="cpu"
    )
    assert info["step"] == 3
    assert config.vocab_size == model_config.vocab_size
    generator = ChatGenerator(
        model,
        loaded_tokenizer,
        config=GenerationConfig(do_sample=False, max_new_tokens=2),
        device="cpu",
    )
    result = generator.chat("Привет")
    assert result.generated_tokens <= 2
    assert len(generator.history) >= 1


def test_chat_cli_single_message(
    tokenizer, model_config, tiny_checkpoint: Path, tmp_path: Path, capsys
) -> None:
    tokenizer_dir = tmp_path / "tokenizer-cli"
    tokenizer.save(tokenizer_dir)
    code = chat_main(
        [
            "--checkpoint",
            str(tiny_checkpoint),
            "--tokenizer",
            str(tokenizer_dir),
            "--device",
            "cpu",
            "--message",
            "Привет",
            "--max-new-tokens",
            "2",
            "--greedy",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Модель" in captured.out or "пустой ответ" in captured.out or captured.out.strip()


def test_missing_weights_error(tokenizer, tmp_path: Path) -> None:
    tokenizer_dir = tmp_path / "tok"
    tokenizer.save(tokenizer_dir)
    try:
        load_model_from_checkpoint(tmp_path / "nope.pt", tokenizer_dir, device="cpu")
    except FileNotFoundError as exc:
        assert "weights" in str(exc).lower() or "not found" in str(exc).lower()
    else:
        raise AssertionError("expected FileNotFoundError")
