"""Small JSON files that tell the watcher what a run is doing right now.

`run.json` is written once at startup and holds the settings the watcher needs
to show progress: how many iterations are planned, the population size, the
model. `status.json` is rewritten as the run moves: which phase it is in and
how far through that phase it is. Both are tiny and written atomically, so the
watcher can read them at any moment without seeing a torn file.

The metrics file stays as it is: one row per completed iteration. These two
files only add the view between rows.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

__all__ = ["RunStatus", "read_json", "write_json"]

RUN_FILE = "run.json"
STATUS_FILE = "status.json"


def write_json(path: Union[str, Path], data: Dict[str, Any]) -> None:
    """Write `data` to `path` so a reader sees either the old or the new file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def read_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read a JSON file, or None if it is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


class RunStatus:
    """Writes `run.json` once and `status.json` on every phase change."""

    def __init__(self, out_dir: Union[str, Path]):
        self.out_dir = Path(out_dir)
        self._state: Dict[str, Any] = {}

    def start(self, **settings: Any) -> None:
        """Record the run's settings; called once when the run begins."""
        write_json(self.out_dir / RUN_FILE, {"started": time.time(), **settings})

    def update(self, phase: str, **fields: Any) -> None:
        """Record the current phase and whatever progress fields describe it.

        Fields from earlier calls are dropped, so a phase only shows its own
        numbers: a training phase shows member/population, an eval phase shows
        split/done/total.
        """
        self._state = {"phase": phase, "updated": time.time(), **fields}
        write_json(self.out_dir / STATUS_FILE, self._state)

    def progress(self, **fields: Any) -> None:
        """Update progress fields within the current phase."""
        self._state.update(fields)
        self._state["updated"] = time.time()
        write_json(self.out_dir / STATUS_FILE, self._state)
