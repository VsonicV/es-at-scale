"""Device-agnostic Evolution Strategies mechanics (Algorithm 2 of the paper).

Deliberately free of model, HuggingFace and device assumptions: everything here
is in-place tensor arithmetic over a plain list of parameter tensors, so it can
be tested without loading an LLM and runs unchanged on mps, cuda or cpu.

The defining property is that perturbation noise is never stored -- only the
seed that generates it. Perturb, restore and update all re-derive the identical
noise from that seed, which is what keeps memory flat regardless of population
size.

This file repeats the perturb, restore and update maths from
`es_at_scale/utils/worker_extension.py`. The upstream versions are methods on a
class that vLLM mixes into its GPU worker, and they read the parameters from the
vLLM model runner, so they cannot be imported here. Moving that maths into one
shared module that both the vLLM worker and this file call would remove the
duplication. That is left for a separate change so this one touches nothing
outside `mps_es/`.
"""

from typing import Iterable, List, Sequence

import numpy as np
import torch

__all__ = ["perturb_", "restore_", "apply_update_", "zscore"]


# TODO(refactor): this module repeats the maths of es_at_scale/utils/worker_extension.py; move both onto one shared module.
def _layer_seed(seed: int, layer_index: int, decorrelate_layers: bool) -> int:
    """Seed for one layer's noise draw.

    Upstream (`es_at_scale/utils/worker_extension.py`) re-seeds the generator
    with the same value for every layer, so identically-shaped layers receive
    identical noise. That is replicated by default so results stay comparable;
    `decorrelate_layers=True` mixes the layer index in instead.
    """
    if not decorrelate_layers:
        return int(seed)
    return (int(seed) * 1_000_003 + layer_index) % (2**31 - 1)


def _noise_like(param: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=param.device)
    generator.manual_seed(seed)
    return torch.randn(
        param.shape, dtype=param.dtype, device=param.device, generator=generator
    )


def _add_scaled_noise_(
    params: Iterable[torch.Tensor],
    seed: int,
    scale: float,
    decorrelate_layers: bool,
) -> None:
    """theta_l += scale * noise(seed, l), one layer at a time.

    Peak extra memory is one layer's worth of noise, never the whole model.
    """
    for layer_index, param in enumerate(params):
        noise = _noise_like(param, _layer_seed(seed, layer_index, decorrelate_layers))
        param.add_(noise, alpha=scale)
        del noise


def perturb_(
    params: Iterable[torch.Tensor],
    seed: int,
    scale: float,
    decorrelate_layers: bool = False,
) -> None:
    """Add `scale * noise(seed)` to every parameter, in place."""
    _add_scaled_noise_(params, seed, scale, decorrelate_layers)


def restore_(
    params: Iterable[torch.Tensor],
    seed: int,
    scale: float,
    decorrelate_layers: bool = False,
) -> None:
    """Undo the matching `perturb_` by subtracting the same noise, in place.

    Not bit-exact -- floating point addition is not invertible -- but in fp32
    the residue is about one ulp (~2e-9 at typical weight scale, six orders of
    magnitude below sigma). In bf16 it is far larger and accumulates as a slow
    random walk: measured RMS drift reaches roughly 20% of sigma after 480
    perturb/restore cycles. That is a property of the upstream design too, which
    likewise perturbs in the model's own dtype.
    """
    _add_scaled_noise_(params, seed, -scale, decorrelate_layers)


def apply_update_(
    params: Iterable[torch.Tensor],
    seeds: Sequence[int],
    z_scores: Sequence[float],
    alpha: float,
    decorrelate_layers: bool = False,
) -> None:
    """theta += alpha * mean_n(z_n * noise(seed_n)), in place.

    One layer at a time: the contributions of all seeds are summed in a float32
    buffer and added to the layer once, as upstream does. Each seed's share of
    the update is far smaller than the resolution of a bf16 weight, so adding
    them one by one in the model's own dtype would round most of them away.
    Peak extra memory is one layer in float32 plus one layer of noise.
    """
    params = list(params)
    n = len(seeds)
    if n == 0:
        raise ValueError("apply_update_ requires at least one seed")
    if len(z_scores) != n:
        raise ValueError(
            f"seeds and z_scores must be the same length, got {n} and {len(z_scores)}"
        )

    scale = float(alpha) / n
    for layer_index, param in enumerate(params):
        total = torch.zeros_like(param, dtype=torch.float32)
        for seed, z in zip(seeds, z_scores):
            if float(z) == 0.0:
                continue
            noise = _noise_like(param, _layer_seed(seed, layer_index, decorrelate_layers))
            total.add_(noise.to(torch.float32), alpha=float(z))
            del noise
        total.mul_(scale)
        param.add_(total.to(param.dtype))
        del total


def zscore(rewards: Sequence[float]) -> np.ndarray:
    """Normalise a population's rewards to mean 0, std 1.

    Returns zeros when the population is degenerate (all rewards equal), which
    would otherwise divide by zero and push NaNs into the weights.
    """
    values = np.asarray(rewards, dtype=np.float64)
    std = values.std()
    if std == 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / std
