"""Tests for the run and status files the watcher reads."""

import json

from mps_es.status import RUN_FILE, STATUS_FILE, RunStatus, read_json, write_json


def test_write_json_is_atomic_and_read_json_tolerates_missing_or_broken_files(tmp_path):
    """The watcher may read while the trainer is mid-write, so writes go
    through a temporary file and reads never raise."""
    write_json(tmp_path / "a.json", {"x": 1})
    (tmp_path / "broken.json").write_text("{not json")

    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "broken.json"]
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 1}
    assert read_json(tmp_path / "absent.json") is None
    assert read_json(tmp_path / "broken.json") is None


def test_start_records_the_settings_and_a_start_time(tmp_path):
    RunStatus(tmp_path).start(task="math", n_iterations=30)

    run = read_json(tmp_path / RUN_FILE)
    assert run["task"] == "math"
    assert run["n_iterations"] == 30
    assert run["started"] > 0


def test_update_replaces_the_phase_and_progress_updates_within_it(tmp_path):
    status = RunStatus(tmp_path)

    status.update("train", iteration=3, member=7, population=30)
    status.update("eval", split="math500", done=0, total=100)
    status.progress(done=16)

    state = read_json(tmp_path / STATUS_FILE)
    assert state["phase"] == "eval"
    assert state["split"] == "math500"
    assert state["done"] == 16
    assert state["total"] == 100
    assert "member" not in state
