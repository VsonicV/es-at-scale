"""Seed-stream management and resumable checkpoints.

Resuming an ES run needs three things restored together: the weights, the
iteration counter, and the state of the generator that produces per-iteration
population seeds. Restoring only the weights would silently start a fresh seed
stream, so a restarted run would explore differently from an uninterrupted one.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

__all__ = ["make_rng", "next_seeds", "save_checkpoint", "load_checkpoint"]

SEED_UPPER_BOUND = 2**30  # matches upstream's np.random.randint(0, 2**30)


def make_rng(seed: int) -> np.random.Generator:
    """The generator that produces population seeds, one draw per iteration."""
    return np.random.default_rng(seed)


def next_seeds(rng: np.random.Generator, population_size: int) -> List[int]:
    """Draw one iteration's worth of population seeds."""
    return [
        int(s)
        for s in rng.integers(0, SEED_UPPER_BOUND, size=population_size, dtype=np.int64)
    ]


def save_checkpoint(
    path: Union[str, Path],
    *,
    iteration: int,
    model: torch.nn.Module,
    rng: np.random.Generator,
    config: Dict[str, Any],
) -> None:
    """Write a checkpoint, replacing any existing one only once fully written.

    Serialising to a temporary file and renaming means an interrupted or failing
    save leaves the previous checkpoint intact rather than truncating it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "iteration": iteration,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "rng_state": rng.bit_generator.state,
        "config": config,
    }

    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a checkpoint, returning a ready-to-use rng under the "rng" key.

    Loaded with weights_only=True: a checkpoint is an untrusted file if it came
    from anywhere but this machine, and everything stored here (tensors, ints,
    strings, plain dicts) is expressible under the restricted unpickler.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)

    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]
    payload["rng"] = rng
    return payload
