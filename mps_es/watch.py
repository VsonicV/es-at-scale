#!/usr/bin/env python3
"""Status viewer for an mps_es run.

Reads only the small files a run writes (metrics.jsonl, run.json, status.json)
and touches no model and no GPU, so it is safe to run alongside training.

    python -m mps_es.watch experiments/<name>              # one report
    python -m mps_es.watch experiments/<name> --follow     # live, until Ctrl-C
    python -m mps_es.watch experiments/<name> -f -v        # live, with detail

The default view is one screen: where the run is, how far along, the latest
numbers and their trend. `--verbose` adds the recent iterations, every eval
split and the run's settings.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from mps_es.status import RUN_FILE, STATUS_FILE, read_json

__all__ = ["Report", "build_report", "load_metrics", "render", "sparkline", "main"]

BLOCKS = "▁▂▃▄▅▆▇█"

# Countdown only: format credit is weighted 0.1 and maxes at 1.0, so a member
# mean above this needs answer credit -- the earliest sign of real solving.
# TODO(refactor): restates the countdown grader's format weight; read it from the grader if upstream exposes it.
FORMAT_CEILING = 0.1

ETA_WINDOW = 10  # iterations averaged for the time estimate


def sparkline(values: Sequence[float]) -> str:
    """Render a series as one block character per point."""
    values = list(values)
    if not values:
        return ""
    low, high = min(values), max(values)
    if high == low:
        return BLOCKS[0] * len(values)
    span = high - low
    return "".join(
        BLOCKS[min(len(BLOCKS) - 1, int((v - low) / span * len(BLOCKS)))] for v in values
    )


def load_metrics(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a metrics.jsonl, tolerating a torn final line.

    The trainer may be mid-write when the watcher reads, so an incomplete last
    line is skipped rather than raising.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partial trailing write
    return rows


@dataclass
class Report:
    """Everything the view needs, pulled from the run directory once."""

    run_dir: Path
    run: Optional[Dict[str, Any]] = None       # run.json, settings
    status: Optional[Dict[str, Any]] = None    # status.json, current phase
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    evals: List[Dict[str, Any]] = field(default_factory=list)
    n_iterations: Optional[int] = None
    now: float = field(default_factory=time.time)

    @property
    def task(self) -> Optional[str]:
        return (self.run or {}).get("task")

    @property
    def completed(self) -> int:
        """Iterations finished so far, as a count (iteration numbers are 0-based)."""
        return self.iterations[-1]["iteration"] + 1 if self.iterations else (
            (self.run or {}).get("start_iteration", 0))

    @property
    def phase(self) -> str:
        if self.status:
            return self.status.get("phase", "unknown")
        if self.n_iterations and self.completed >= self.n_iterations:
            return "done"
        return "unknown"

    def seconds_per_iteration(self) -> Optional[float]:
        recent = [r["seconds"] for r in self.iterations[-ETA_WINDOW:] if "seconds" in r]
        return sum(recent) / len(recent) if recent else None

    def eta_seconds(self) -> Optional[float]:
        per = self.seconds_per_iteration()
        if per is None or not self.n_iterations:
            return None
        return max(0.0, (self.n_iterations - self.completed) * per)

    def elapsed_seconds(self) -> Optional[float]:
        if self.run and "started" in self.run:
            return self.now - float(self.run["started"])
        total = sum(r.get("seconds", 0.0) for r in self.iterations)
        return total or None

    def status_age(self) -> Optional[float]:
        if self.status and "updated" in self.status:
            return self.now - float(self.status["updated"])
        return None


def build_report(run_dir: Union[str, Path], total: Optional[int] = None) -> Report:
    """Read the run directory. `total` overrides the planned iteration count
    for runs started before run.json existed."""
    run_dir = Path(run_dir)
    if run_dir.is_file():  # a metrics.jsonl path was given
        run_dir = run_dir.parent
    rows = load_metrics(run_dir / "metrics.jsonl")
    run = read_json(run_dir / RUN_FILE)
    n_iterations = total if total is not None else (run or {}).get("n_iterations")
    return Report(
        run_dir=run_dir,
        run=run,
        status=read_json(run_dir / STATUS_FILE),
        iterations=[r for r in rows if "mean_reward" in r],
        evals=[r for r in rows if "eval_accuracy" in r],
        n_iterations=n_iterations,
    )


def _duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"


def _header(report: Report) -> Text:
    run = report.run or {}
    parts = [report.run_dir.name]
    if run.get("task"):
        parts.append(run["task"])
    if run.get("model_name"):
        parts.append(run["model_name"])
    if run.get("device"):
        parts.append(f"{run['device']}/{run.get('precision', '')}".rstrip("/"))
    text = Text("  ·  ".join(parts), style="bold")
    phase = report.phase
    style = {"done": "green", "unknown": "dim"}.get(phase, "cyan")
    text.append(f"\n{phase}", style=style)
    if phase not in ("done", "unknown") and report.status_age() is not None:
        text.append(f"  (updated {_duration(report.status_age())} ago)", style="dim")
    text.append(f"   elapsed {_duration(report.elapsed_seconds())}", style="dim")
    eta = report.eta_seconds()
    if eta is not None and phase != "done":
        text.append(f"   about {_duration(eta)} left", style="dim")
    return text


def _progress(report: Report) -> Progress:
    bars = Progress(
        TextColumn("{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed:>4.0f}/{task.total:<4.0f}"),
        TextColumn("[dim]{task.fields[note]}"),
    )
    per = report.seconds_per_iteration()
    note = f"{_duration(per)} per iteration" if per else ""
    bars.add_task("iterations ", total=report.n_iterations or max(report.completed, 1),
                  completed=report.completed, note=note if report.n_iterations else "planned total unknown; pass --total")

    status = report.status or {}
    phase = status.get("phase")
    if phase == "train":
        bars.add_task(f"iteration {status.get('iteration', '?')}  members",
                      total=status.get("population") or 1,
                      completed=status.get("member", 0), note="")
    elif phase == "eval":
        label = f"eval {status.get('tag', '')} {status.get('split', '')}".rstrip()
        bars.add_task(label, total=status.get("total") or 1,
                      completed=status.get("done", 0), note="")
    return bars


def _format_label(report: Report) -> str:
    return "boxed" if report.task == "math" else "format"


def _metrics(report: Report) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    if not report.iterations and not report.evals:
        table.add_row("metrics", "none yet")
        return table

    if report.iterations:
        latest = report.iterations[-1]
        means = [r["mean_reward"] for r in report.iterations]
        table.add_row(
            "training reward",
            f"mean {latest['mean_reward']:.4f}   best member {latest.get('max_reward', 0.0):.4f}"
            f"   {sparkline(means)}  {means[0]:.3f} → {means[-1]:.3f}",
        )
        if "mean_format_reward" in latest:
            table.add_row(
                _format_label(report),
                f"{latest['mean_format_reward']:.3f}   answer {latest.get('mean_answer_reward', 0.0):.4f}"
                f"   best member accuracy {latest.get('best_member_accuracy', 0.0):.3f}",
            )

    if report.evals:
        accs = [r["eval_accuracy"] for r in report.evals]
        first = "baseline" if report.evals[0].get("baseline") else f"iter {report.evals[0]['iteration']}"
        table.add_row(
            "held-out accuracy",
            f"{accs[-1]:.3f}   {sparkline(accs)}  {first} {accs[0]:.3f} → {accs[-1]:.3f}"
            f"   ({len(report.evals)} evals)",
        )

    if report.task == "countdown" and report.iterations:
        over = [r for r in report.iterations if r.get("max_reward", 0.0) > FORMAT_CEILING]
        if over:
            table.add_row("solving", f"a member beat the format ceiling in {len(over)} of "
                                     f"{len(report.iterations)} iterations, first at iter {over[0]['iteration']}")
        else:
            table.add_row("solving", "no member above the format ceiling yet, still learning the envelope")
    return table


def _recent_table(report: Report, n: int = 10) -> Table:
    table = Table(title=f"last {min(n, len(report.iterations))} iterations", title_justify="left")
    for col in ("iter", "mean", "best", _format_label(report), "answer", "best acc", "seconds", "eval"):
        table.add_column(col, justify="right")
    for r in report.iterations[-n:]:
        table.add_row(
            str(r["iteration"]), f"{r['mean_reward']:.4f}", f"{r.get('max_reward', 0.0):.4f}",
            f"{r.get('mean_format_reward', 0.0):.3f}", f"{r.get('mean_answer_reward', 0.0):.4f}",
            f"{r.get('best_member_accuracy', 0.0):.3f}", f"{r.get('seconds', 0.0):.0f}",
            f"{r['eval_accuracy']:.3f}" if "eval_accuracy" in r else "",
        )
    return table


def _evals_table(report: Report) -> Table:
    splits = sorted({k for r in report.evals for k in r if k.startswith("eval_accuracy_")})
    table = Table(title="evals", title_justify="left")
    table.add_column("at")
    table.add_column("accuracy", justify="right")
    for s in splits:
        table.add_column(s.replace("eval_accuracy_", ""), justify="right")
    for r in report.evals:
        at = "baseline" if r.get("baseline") else f"iter {r['iteration']}"
        table.add_row(at, f"{r['eval_accuracy']:.3f}",
                      *(f"{r[s]:.3f}" if s in r else "" for s in splits))
    return table


def _settings_lines(report: Report) -> Table:
    """Run settings and current status as two inline JSON lines."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    run = dict(report.run or {})
    if "started" in run:
        run["started"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(run["started"])))
    if run:
        table.add_row("run settings", json.dumps(run, default=str))
    if report.status:
        table.add_row("status", json.dumps({k: v for k, v in report.status.items() if k != "updated"}))
    return table


def render(report: Report, verbose: bool = False) -> Group:
    """The whole view as one renderable."""
    parts: List[Any] = [_header(report), Text(), _progress(report), Text(), _metrics(report)]
    if verbose:
        if report.iterations:
            parts += [Text(), _recent_table(report)]
        if report.evals:
            parts += [Text(), _evals_table(report)]
        if report.run or report.status:
            parts += [Text(), _settings_lines(report)]
    return Group(*parts)


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def watch(
    path: Annotated[Path, typer.Argument(help="experiment directory, or its metrics.jsonl")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="refresh until Ctrl-C or the run finishes")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="recent iterations, every eval split, settings")] = False,
    interval: Annotated[float, typer.Option(help="seconds between refreshes")] = 5.0,
    total: Annotated[Optional[int], typer.Option(help="planned iterations, for runs without run.json")] = None,
) -> None:
    """Show where a run is and how it is doing."""
    console = Console()
    report = build_report(path, total)
    if not follow:
        console.print(render(report, verbose))
        return
    with Live(render(report, verbose), console=console, refresh_per_second=4) as live:
        try:
            while True:
                if report.phase == "done":
                    break
                time.sleep(interval)
                report = build_report(path, total)
                live.update(render(report, verbose))
        except KeyboardInterrupt:
            pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    app(args=list(argv) if argv is not None else None, standalone_mode=False)
    return 0


if __name__ == "__main__":
    app()
