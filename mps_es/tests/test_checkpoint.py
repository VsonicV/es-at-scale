"""Tests for checkpoint save/resume.

A resumed run must continue the *same* seed stream, not merely start a new one
from the same weights -- otherwise a multi-day run silently changes character
every time it is restarted.
"""

import torch

from mps_es.checkpoint import load_checkpoint, make_rng, next_seeds, save_checkpoint


def test_resumed_rng_state_continues_the_identical_seed_stream(tmp_path):
    rng = make_rng(42)
    next_seeds(rng, 30)  # iteration 0
    next_seeds(rng, 30)  # iteration 1

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path, iteration=2, model=torch.nn.Linear(4, 4), rng=rng, config={"sigma": 0.001}
    )
    expected = next_seeds(rng, 30)  # iteration 2, from the live rng

    resumed = load_checkpoint(path)
    actual = next_seeds(resumed["rng"], 30)  # iteration 2, from the restored rng

    assert actual == expected


def test_checkpoint_round_trips_iteration_config_and_weights(tmp_path):
    model = torch.nn.Linear(4, 4)
    with torch.no_grad():
        model.weight.fill_(0.123)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path, iteration=7, model=model, rng=make_rng(1),
        config={"sigma": 0.001, "alpha": 0.0005},
    )

    loaded = load_checkpoint(path)
    restored = torch.nn.Linear(4, 4)
    restored.load_state_dict(loaded["model_state_dict"])

    assert loaded["iteration"] == 7
    assert loaded["config"] == {"sigma": 0.001, "alpha": 0.0005}
    assert torch.equal(restored.weight, model.weight)


def test_saving_is_atomic_enough_to_leave_no_partial_file_on_failure(tmp_path):
    """A crash mid-save must not destroy the previous good checkpoint."""
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, iteration=1, model=torch.nn.Linear(4, 4), rng=make_rng(1), config={})

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    try:
        save_checkpoint(
            path, iteration=2, model=torch.nn.Linear(4, 4), rng=make_rng(1),
            config={"bad": Unpicklable()},
        )
    except Exception:
        pass

    assert load_checkpoint(path)["iteration"] == 1
    assert not list(tmp_path.glob("*.tmp"))
