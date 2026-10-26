"""Tests for the status viewer."""

import json
import time

from rich.console import Console
from typer.testing import CliRunner

from mps_es.status import RunStatus
from mps_es.watch import app, build_report, load_metrics, render, sparkline


def write_metrics(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def as_text(renderable, width=140) -> str:
    console = Console(record=True, width=width, force_terminal=False, color_system=None)
    console.print(renderable)
    return console.export_text()


TRAIN_ROWS = [
    {"iteration": 0, "baseline": True, "eval_accuracy": 0.01, "eval_mean_reward": 0.05},
    {"iteration": 0, "mean_reward": 0.03, "max_reward": 0.05, "mean_format_reward": 0.3,
     "mean_answer_reward": 0.0, "best_member_accuracy": 0.0, "seconds": 100.0},
    {"iteration": 1, "mean_reward": 0.04, "max_reward": 0.1141, "mean_format_reward": 0.4,
     "mean_answer_reward": 0.004, "best_member_accuracy": 0.03, "seconds": 120.0,
     "eval_accuracy": 0.02, "eval_mean_reward": 0.09},
]


def test_sparkline_maps_the_range_onto_one_block_per_value():
    line = sparkline([0, 1, 2, 3, 4])
    assert len(line) == 5
    assert line[0] == "▁"
    assert line[-1] == "█"
    assert len(set(sparkline([3, 3, 3]))) == 1
    assert sparkline([]) == ""


def test_load_metrics_skips_a_partially_written_trailing_line(tmp_path):
    """The watcher may read while the trainer is mid-write; a torn final line
    must not crash it."""
    path = tmp_path / "m.jsonl"
    path.write_text('{"iteration": 0, "mean_reward": 0.1}\n{"iteration": 1, "mea')

    rows = load_metrics(path)

    assert len(rows) == 1
    assert rows[0]["iteration"] == 0


def test_build_report_separates_training_rows_from_eval_rows(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)

    report = build_report(tmp_path)

    assert len(report.iterations) == 2
    assert len(report.evals) == 2  # the baseline row counts as an eval, not an iteration
    assert report.completed == 2


def test_build_report_takes_the_planned_total_from_run_json_or_the_total_option(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)
    assert build_report(tmp_path, total=30).n_iterations == 30

    RunStatus(tmp_path).start(task="countdown", n_iterations=50)
    report = build_report(tmp_path)

    assert report.n_iterations == 50
    assert report.task == "countdown"


def test_eta_uses_the_recent_seconds_per_iteration(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)

    report = build_report(tmp_path, total=12)

    assert report.seconds_per_iteration() == 110.0
    assert report.eta_seconds() == 10 * 110.0


def test_render_shows_the_latest_training_reward_and_held_out_accuracy(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)

    text = as_text(render(build_report(tmp_path, total=50)))

    assert "0.0400" in text          # latest mean reward
    assert "0.020" in text           # latest eval accuracy
    assert "baseline 0.010" in text  # first eval was the baseline
    assert "2/50" in text            # iteration progress


def test_render_flags_answer_credit_for_countdown_only(tmp_path):
    """A member above the 0.1 format ceiling proves a solved puzzle in
    countdown; the math reward has no ceiling, so the note would mislead."""
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)
    RunStatus(tmp_path).start(task="countdown", n_iterations=50)
    countdown = as_text(render(build_report(tmp_path)))

    RunStatus(tmp_path).start(task="math", n_iterations=50)
    math = as_text(render(build_report(tmp_path)))

    assert "beat the format ceiling" in countdown
    assert "format ceiling" not in math
    assert "boxed" in math


def test_render_shows_the_current_phase_and_eval_progress_from_status_json(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)
    status = RunStatus(tmp_path)
    status.start(task="math", n_iterations=30, population_size=30)

    status.update("train", iteration=2, member=7, population=30)
    training = as_text(render(build_report(tmp_path)))

    status.update("eval", tag="iteration10", iteration=10, split="math500", done=48, total=100)
    evaluating = as_text(render(build_report(tmp_path)))

    assert "train" in training and "7/30" in training
    assert "math500" in evaluating and "48/100" in evaluating


def test_render_of_an_empty_directory_does_not_crash(tmp_path):
    text = as_text(render(build_report(tmp_path)))

    assert "none yet" in text


def test_verbose_adds_recent_iterations_evals_and_settings(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS + [
        {"iteration": 2, "mean_reward": 0.05, "max_reward": 0.12, "seconds": 90.0,
         "eval_accuracy": 0.03, "eval_mean_reward": 0.1, "eval_accuracy_math500": 0.03,
         "eval_accuracy_amc": 0.01},
    ])
    RunStatus(tmp_path).start(task="math", n_iterations=30, model_name="m")

    compact = as_text(render(build_report(tmp_path)))
    verbose = as_text(render(build_report(tmp_path), verbose=True))

    assert "last 3 iterations" not in compact
    assert "last 3 iterations" in verbose
    assert "amc" in verbose
    assert "run settings" in verbose


def test_cli_prints_one_report_without_follow(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)

    result = CliRunner().invoke(app, [str(tmp_path), "--total", "50"])

    assert result.exit_code == 0, result.output
    assert "2/50" in result.output


def test_cli_follow_exits_on_its_own_once_the_run_is_done(tmp_path):
    write_metrics(tmp_path / "metrics.jsonl", TRAIN_ROWS)
    status = RunStatus(tmp_path)
    status.start(task="math", n_iterations=2)
    status.update("done", iteration=2)

    started = time.time()
    result = CliRunner().invoke(app, [str(tmp_path), "--follow", "--interval", "0.1"])

    assert result.exit_code == 0, result.output
    assert time.time() - started < 5
