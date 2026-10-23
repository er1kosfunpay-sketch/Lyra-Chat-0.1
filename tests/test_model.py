from __future__ import annotations

import torch
import torch.nn.functional as F

import pytest

from conversation_llm.config import ModelConfig
from conversation_llm.model import GPT, causal_mask_reference
from conversation_llm.utils.reproducibility import set_seed


def test_logits_shapes_and_parameter_count(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    ids = torch.randint(0, model_config.vocab_size, (3, 17))
    output = model(ids)
    assert output.logits.shape == (3, 17, model_config.vocab_size)
    assert output.loss is None
    assert model.num_parameters() > 0
    assert model.num_parameters(non_embedding=True) < model.num_parameters()
    if model_config.tie_embeddings:
        assert model.transformer["wte"].weight is model.lm_head.weight


def test_causal_mask_blocks_future_tokens(model_config: ModelConfig) -> None:
    set_seed(1)
    model = GPT(model_config).eval()
    ids = torch.randint(0, model_config.vocab_size, (1, 12))
    changed = ids.clone()
    changed[:, 8:] = torch.randint(0, model_config.vocab_size, (1, 4))
    with torch.no_grad():
        first = model(ids).logits
        second = model(changed).logits
    assert torch.allclose(first[:, :8], second[:, :8], atol=1e-6)
    assert not torch.allclose(first[:, 8:], second[:, 8:], atol=1e-6)
    mask = causal_mask_reference(5)
    assert mask[0, 0] and mask[4, 4]
    assert not mask[0, 4] and not mask[2, 4]
    assert mask.sum().item() == 15


def test_attention_padding_does_not_change_real_tokens(model_config: ModelConfig) -> None:
    set_seed(2)
    model = GPT(model_config).eval()
    ids = torch.randint(1, model_config.vocab_size, (1, 9))
    padded = ids.clone()
    padded[:, 5:] = model_config.vocab_size - 1
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]])
    with torch.no_grad():
        full = model(padded, attention_mask=attention_mask).logits
        short = model(padded[:, :5]).logits
    assert torch.allclose(full[:, :5], short, atol=1e-6)


def test_loss_is_cross_entropy_and_backpropagates(model_config: ModelConfig) -> None:
    set_seed(3)
    model = GPT(model_config)
    ids = torch.randint(0, model_config.vocab_size, (2, 10))
    labels = torch.randint(0, model_config.vocab_size, (2, 10))
    labels[:, :3] = -100
    output = model(ids, labels=labels)
    assert output.loss is not None
    expected = F.cross_entropy(
        output.logits.reshape(-1, model_config.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(output.loss, expected)
    output.loss.backward()
    assert model.transformer["wte"].weight.grad is not None
    assert torch.isfinite(model.transformer["wte"].weight.grad).all()


def test_only_padding_labels_raises(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    ids = torch.randint(0, model_config.vocab_size, (1, 4))
    labels = torch.full((1, 4), -100)
    with pytest.raises(ValueError, match="no supervised"):
        model(ids, labels=labels)


def test_kv_cache_matches_full_forward(model_config: ModelConfig) -> None:
    set_seed(4)
    model = GPT(model_config).eval()
    ids = torch.randint(0, model_config.vocab_size, (1, 10))
    with torch.no_grad():
        full = model(ids).logits
        first = model(ids[:, :4], use_cache=True)
        second = model(
            ids[:, 4:], past_key_values=first.past_key_values, use_cache=True
        )
    assert first.past_key_values is not None
    assert len(first.past_key_values) == model_config.n_layer
    assert torch.allclose(full[:, 4:], second.logits, atol=1e-5)
    assert torch.allclose(full[:, :4], first.logits, atol=1e-5)


def test_invalid_shapes_and_context_are_rejected(model_config: ModelConfig) -> None:
    model = GPT(model_config)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.zeros((1, model_config.block_size + 1), dtype=torch.long))
    with pytest.raises(ValueError, match="batch"):
        model(torch.zeros((3,), dtype=torch.long))
    ids = torch.zeros((2, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="labels"):
        model(ids, labels=torch.zeros((2, 3), dtype=torch.long))
