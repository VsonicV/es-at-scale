"""Tests for the countdown task wiring.

Scoring is the part with real logic: mean reward drives ES, while accuracy
(exact-answer rate) is the number comparable to the paper's Table 1, and the
format/answer split shows whether the model solves puzzles or only formats.
"""

import json

import pytest

from mps_es.train_countdown import (
    TimedGrader,
    eval_standard_error,
    evaluate_split,
    score_batch,
    score_responses,
)


CORRECT = "</think>\n<answer> (44 + 19) + 35 </answer>"
WRONG = "</think>\n<answer> 44 + 19 </answer>"
UNFORMATTED = "the answer is 98"


def target(numbers=(44, 19, 35), value=98):
    return {"numbers": list(numbers), "target": value}


def test_score_batch_gives_a_correct_answer_full_credit_and_counts_it_as_accuracy():
    result = score_batch([CORRECT, WRONG], [target(), target()])

    assert result.accuracy == pytest.approx(0.5)
    assert score_batch([CORRECT], [target()]).mean_reward == pytest.approx(1.1)  # 0.1 * format + 1.0 * answer


def test_score_batch_splits_reward_into_format_and_answer_components():
    """total = 0.1 * format + answer, the weighting countdown_reward_fn uses."""
    wrong = score_batch([WRONG], [target()])
    assert wrong.mean_answer_reward == pytest.approx(0.0)
    assert wrong.mean_format_reward > 0.0

    unformatted = score_batch([UNFORMATTED], [target()])
    assert unformatted.mean_reward == pytest.approx(0.0)
    assert unformatted.mean_format_reward == pytest.approx(0.0)

    mixed = score_batch([CORRECT, WRONG, UNFORMATTED], [target()] * 3)
    assert mixed.mean_reward == pytest.approx(0.1 * mixed.mean_format_reward + mixed.mean_answer_reward)


def test_score_batch_on_an_empty_batch_is_zero_rather_than_nan():
    result = score_batch([], [])

    assert result.mean_reward == 0.0
    assert result.accuracy == 0.0
    assert result.mean_format_reward == 0.0
    assert result.mean_answer_reward == 0.0


def test_score_responses_returns_one_graded_record_per_response():
    records = score_responses([CORRECT, WRONG], [target(), target()])

    assert [r["correct"] for r in records] == [True, False]
    assert records[0]["reward"] == pytest.approx(1.1)
    assert records[0]["response"] == CORRECT
    assert records[0]["target"] == target()


# TODO(refactor): duplicated in test_train_math; move to a conftest.py fixture.
class _ScriptedRunner:
    """Returns fixed responses so eval can be tested without a model."""

    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, prompts, mini_batch_size):
        out, self._responses = self._responses[: len(prompts)], self._responses[len(prompts):]
        return out


def test_evaluate_split_grades_every_prompt_and_writes_each_response(tmp_path):
    runner = _ScriptedRunner([CORRECT, WRONG, UNFORMATTED])
    prompts = ["p1", "p2", "p3"]
    targets = [target(), target(), target()]
    out = tmp_path / "eval_iteration1.json"

    score = evaluate_split(runner, prompts, targets, mini_batch_size=2, output_path=out)

    assert score.accuracy == pytest.approx(1 / 3)
    saved = json.loads(out.read_text())
    assert [r["prompt"] for r in saved] == prompts
    assert [r["correct"] for r in saved] == [True, False, False]


def test_evaluate_split_result_does_not_depend_on_mini_batch_size():
    responses = [CORRECT, WRONG, UNFORMATTED, CORRECT, WRONG]
    prompts = [f"p{i}" for i in range(5)]
    targets = [target() for _ in range(5)]

    whole = evaluate_split(_ScriptedRunner(responses), prompts, targets, mini_batch_size=5)
    chunked = evaluate_split(_ScriptedRunner(responses), prompts, targets, mini_batch_size=2)

    assert whole == chunked


def _grade_or_hang(response, target):
    """Stand-in grader: hangs on one specific response, grades the rest."""
    if response == "hang":
        import time
        time.sleep(30)
    from es_at_scale.reward_function.countdown_grader import countdown_reward_fn
    return countdown_reward_fn(response, target)


def test_timed_grader_scores_a_hung_call_as_zero_and_keeps_grading():
    grader = TimedGrader(_grade_or_hang, timeout=1)
    try:
        detail, reward = grader("hang", target())
        after_detail, after = grader(CORRECT, target())
    finally:
        grader.close()

    assert reward == 0.0
    assert detail["timed_out"] is True
    assert after == pytest.approx(1.1)
    assert after_detail["answer_reward"] == 1.0


def test_eval_standard_error_follows_the_binomial_formula():
    """A small eval set reads as a hard plateau, so the run prints its own
    resolution: sqrt(p(1-p)/n), and 0 rather than a division error for n=0."""
    assert eval_standard_error(100, accuracy=0.1) == pytest.approx(0.03, abs=1e-9)
    assert eval_standard_error(1600) == pytest.approx(eval_standard_error(100) / 4, rel=1e-9)
    assert eval_standard_error(0) == 0.0
