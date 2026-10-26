"""One ES iteration, and the loop that drives many of them.

Kept independent of the countdown task and of HuggingFace: `run_iteration` takes
a list of tensors and a zero-argument `evaluate` callable that scores whatever
the parameters currently are. That is what makes the update maths testable
against a reward with a known optimum.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np
import torch

from mps_es.es_loop import apply_update_, perturb_, restore_, zscore

__all__ = ["ESHyperparams", "IterationResult", "run_iteration"]


@dataclass(frozen=True)
class ESHyperparams:
    sigma: float
    alpha: float
    population_size: int
    decorrelate_layers: bool = False


@dataclass
class IterationResult:
    rewards: List[float]
    z_scores: np.ndarray = field(repr=False)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def max_reward(self) -> float:
        return float(np.max(self.rewards))


def run_iteration(
    params: Sequence[torch.Tensor],
    seeds: Sequence[int],
    evaluate: Callable[[], float],
    hp: ESHyperparams,
) -> IterationResult:
    """Evaluate one population, then apply the aggregated update in place.

    Each member is perturbed, scored, and restored before the next is sampled,
    so at most one perturbation is ever live and peak memory stays flat in the
    population size.
    """
    params = list(params)
    rewards: List[float] = []

    for seed in seeds:
        perturb_(params, seed, hp.sigma, hp.decorrelate_layers)
        try:
            rewards.append(float(evaluate()))
        finally:
            # restore even if scoring raises, or the weights keep the perturbation
            restore_(params, seed, hp.sigma, hp.decorrelate_layers)

    z = zscore(rewards)
    apply_update_(params, seeds, z, hp.alpha, hp.decorrelate_layers)

    return IterationResult(rewards=rewards, z_scores=z)
