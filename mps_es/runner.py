"""Model loading and batched greedy generation.

Separated from the ES mechanics so that `es_loop` stays free of HuggingFace and
device concerns, and so this half can be exercised against a tiny local model.
"""

from typing import Iterable, Iterator, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = [
    "ModelRunner",
    "iter_minibatches",
    "load_runner",
    "resolve_device",
    "resolve_dtype",
    "stop_token_ids",
    "weighted_mean",
]

# Marks the end of an assistant turn in the Qwen chat format. Base models list
# only <|endoftext|> as their stop token, so without this a chat-formatted
# answer runs on past its end into junk until the token limit.
END_OF_TURN_TOKEN = "<|im_end|>"

_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_dtype(precision: str) -> torch.dtype:
    try:
        return _DTYPES[precision]
    except KeyError:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {sorted(_DTYPES)}"
        ) from None


def resolve_device(preferred: Optional[str] = None) -> str:
    """Pick a device, preferring Apple Silicon then CUDA then CPU."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def stop_token_ids(tokenizer, model) -> List[int]:
    """Token ids at which generation stops.

    The model's own stop tokens, plus the chat end-of-turn token when the
    tokenizer knows it. Instruct models already list both; base models list
    only end-of-text.
    """
    ids = set()
    configured = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(int(i) for i in configured)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    end_of_turn = tokenizer.convert_tokens_to_ids(END_OF_TURN_TOKEN)
    if end_of_turn is not None and end_of_turn != tokenizer.unk_token_id:
        ids.add(end_of_turn)
    return sorted(ids)


def iter_minibatches(items: Sequence, size: int) -> Iterator[List]:
    """Split a batch into sequential chunks.

    Purely a memory lever, exactly as upstream's `--mini-batch-size`: rewards are
    recombined with `weighted_mean`, so chunking does not change the iteration's
    result.
    """
    if size <= 0:
        raise ValueError(f"mini-batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# TODO(refactor): unused since the trainers grade a whole batch at once; delete it, its __all__ entry and the mention above.
def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """Recombine per-chunk means into the mean over the undivided batch."""
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


class ModelRunner:
    """Owns the model and turns prompts into completions.

    `trainable_params` hands out the model's own tensors, not copies, because the
    ES loop perturbs them in place.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        max_new_tokens: int,
        stop_token_ids: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids

    def trainable_params(self) -> List[torch.Tensor]:
        return [p.data for p in self.model.parameters()]

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], mini_batch_size: int) -> List[str]:
        """Greedily complete each prompt; returns completions in prompt order."""
        if not prompts:
            return []

        completions: List[str] = []
        for chunk in iter_minibatches(prompts, mini_batch_size):
            encoded = self.tokenizer(
                chunk, return_tensors="pt", padding=True
            ).to(self.device)
            stop = {}
            if self.stop_token_ids:
                stop["eos_token_id"] = self.stop_token_ids
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                **stop,
            )
            # strip the prompt: everything up to the (left-padded) input width
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            completions.extend(
                self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            )
        return completions


def load_runner(
    model_name: str,
    device: Optional[str],
    precision: str,
    max_new_tokens: int,
) -> ModelRunner:
    """Load a HuggingFace causal LM ready for ES perturbation."""
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # transformers renamed `torch_dtype` to `dtype` in 5.x. Accept both so this
    # module also works in an environment pinned to the upstream 4.57.6.
    dtype = resolve_dtype(precision)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)  # ES never differentiates; saves autograd bookkeeping

    return ModelRunner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids(tokenizer, model),
    )
