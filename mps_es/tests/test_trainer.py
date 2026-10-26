"""Tests for one ES iteration and for convergence of the loop as a whole.

The convergence test uses a reward with a known optimum, so it verifies the
update maths end to end without needing an LLM or a grader.
"""

import numpy as np
import pytest
import torch

from mps_es.checkpoint import make_rng, next_seeds
from mps_es.trainer import ESHyperparams, run_iteration


@pytest.fixture
def hyperparams():
    return ESHyperparams(sigma=0.01, alpha=0.005, population_size=8)


# TODO(refactor): see the note on make_params in test_es_loop.
def make_params():
    torch.manual_seed(0)
    return [torch.randn(16, 8) * 0.02, torch.randn(8) * 0.02]


def test_run_iteration_scores_each_member_once_with_its_own_perturbed_weights(hyperparams):
    """If the weights were not perturbed at evaluation time, every member would
    see identical parameters and ES would be sampling nothing."""
    params = make_params()
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    supply = iter(rewards)
    seen = []

    result = run_iteration(params, seeds=list(range(8)),
                           evaluate=lambda: seen.append(params[0].clone()) or next(supply),
                           hp=hyperparams)

    assert len(seen) == 8
    assert not torch.equal(seen[0], seen[1])
    assert result.rewards == rewards
    assert result.mean_reward == pytest.approx(0.45)


# TODO(refactor): see the note on the naive-update reference in test_es_loop.
def test_run_iteration_leaves_only_the_update_applied_not_leftover_noise(hyperparams):
    """After the population sweep every perturbation must have been restored, so
    the net change equals exactly the ES update and nothing else."""
    params = make_params()
    before = [p.clone() for p in params]
    seeds = list(range(1, 9))
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    supply = iter(rewards)

    result = run_iteration(params, seeds=seeds, evaluate=lambda: next(supply), hp=hyperparams)

    expected = [b.clone() for b in before]
    for seed, z in zip(seeds, result.z_scores):
        for i, e in enumerate(expected):
            g = torch.Generator(device=e.device)
            g.manual_seed(seed)
            noise = torch.randn(e.shape, dtype=e.dtype, generator=g)
            e += (hyperparams.alpha / len(seeds)) * z * noise

    for p, e in zip(params, expected):
        assert torch.allclose(p, e, atol=1e-7)


def test_run_iteration_with_a_flat_reward_applies_no_update(hyperparams):
    """All-equal rewards give z-scores of zero, so there is no signal to follow.

    The weights still shift by the rounding residue of eight perturb/restore
    round trips; what must be absent is any *systematic* step, which at this
    alpha would be five orders of magnitude larger.
    """
    params = make_params()
    before = [p.clone() for p in params]

    result = run_iteration(params, seeds=list(range(8)), evaluate=lambda: 0.5, hp=hyperparams)

    assert np.all(result.z_scores == 0.0)
    for p, b in zip(params, before):
        drift = (p - b).abs().max().item()
        rounding_budget = 8 * 4 * torch.finfo(p.dtype).eps * b.abs().max().item()
        assert drift <= rounding_budget
        assert drift < 1e-4 * hyperparams.alpha


def test_es_climbs_a_reward_with_a_known_optimum():
    """End-to-end sanity on the update maths: reward = -distance to a target, so
    a working ES must reduce that distance."""
    torch.manual_seed(0)
    target = [torch.randn(16, 8), torch.randn(8)]
    params = [torch.zeros_like(t) for t in target]
    hp = ESHyperparams(sigma=0.05, alpha=0.4, population_size=30)

    def evaluate():
        return -sum(((p - t) ** 2).sum().item() for p, t in zip(params, target))

    rng = make_rng(0)
    first = evaluate()
    trajectory = []
    for _ in range(40):
        trajectory.append(run_iteration(
            params, seeds=next_seeds(rng, hp.population_size), evaluate=evaluate, hp=hp
        ).mean_reward)
    last = evaluate()

    assert last > first, f"reward did not improve: {first:.3f} -> {last:.3f}"
    assert np.mean(trajectory[-5:]) > np.mean(trajectory[:5])
