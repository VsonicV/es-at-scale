"""Tests for model loading and batched greedy generation.

Built against a real (tiny, randomly initialised) Qwen model and the real Qwen
tokenizer rather than mocks, so the transformers generate path is genuinely
exercised. Nothing here downloads weights.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from mps_es.runner import ModelRunner, iter_minibatches, resolve_dtype, stop_token_ids

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# TODO(refactor): needs the Qwen tokenizer in the HF cache; a tiny tokenizer saved under tests/ would make this hermetic.
@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture
def runner(tiny_model, tokenizer):
    return ModelRunner(
        model=tiny_model, tokenizer=tokenizer, device="cpu", max_new_tokens=4
    )


def test_resolve_dtype_maps_precision_names_and_rejects_unknown_ones():
    assert resolve_dtype("fp32") is torch.float32
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16
    with pytest.raises(ValueError, match="unknown precision"):
        resolve_dtype("int4")


def test_iter_minibatches_covers_every_item_in_order_and_rejects_a_non_positive_size():
    assert list(iter_minibatches(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
    with pytest.raises(ValueError):
        list(iter_minibatches([1, 2], 0))


def test_generate_returns_one_completion_per_prompt_without_the_prompt(runner):
    prompts = ["hello", "a distinctive prompt string"]

    outputs = runner.generate(prompts, mini_batch_size=8)

    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)
    assert prompts[1] not in outputs[1]


def test_perturbing_trainable_params_changes_the_live_model(runner, tiny_model):
    """The ES loop mutates these tensors in place, so they must be the model's
    own storage rather than copies."""
    params = runner.trainable_params()
    assert len(params) == len(list(tiny_model.parameters()))

    before = next(iter(tiny_model.parameters())).clone()
    with torch.no_grad():
        params[0].add_(torch.ones_like(params[0]))

    after = next(iter(tiny_model.parameters()))
    assert not torch.equal(before, after)

    with torch.no_grad():  # leave the module-scoped fixture as we found it
        params[0].sub_(torch.ones_like(params[0]))


class _RecordingModel:
    """Stands in for a model: records generate() kwargs, echoes the prompt."""

    def __init__(self, eos_token_id):
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self.calls = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        return input_ids


def test_stop_token_ids_adds_end_of_turn_and_keeps_configured_stops_without_duplicates(tokenizer):
    """A base model's config stops only at <|endoftext|>, so the chat-format
    <|im_end|> is added; an instruct model that already lists it gets no duplicate."""
    end_of_turn = tokenizer.convert_tokens_to_ids("<|im_end|>")

    base = stop_token_ids(tokenizer, _RecordingModel(eos_token_id=tokenizer.eos_token_id))
    instruct = stop_token_ids(tokenizer, _RecordingModel(eos_token_id=[tokenizer.eos_token_id, end_of_turn]))

    assert tokenizer.eos_token_id in base and end_of_turn in base
    assert instruct == sorted({tokenizer.eos_token_id, end_of_turn})


def test_generate_passes_the_stop_tokens_to_the_model(tokenizer):
    model = _RecordingModel(eos_token_id=tokenizer.eos_token_id)
    runner = ModelRunner(model=model, tokenizer=tokenizer, device="cpu",
                         max_new_tokens=4, stop_token_ids=[1, 2])

    runner.generate(["hello"], mini_batch_size=1)

    assert model.calls[0]["eos_token_id"] == [1, 2]
