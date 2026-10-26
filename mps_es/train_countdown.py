#!/usr/bin/env python3
"""ES fine-tuning on the Countdown task, on Apple Silicon (or CPU, or CUDA).

Single-process counterpart to `es_at_scale/train.py`, which requires vLLM, Ray
and CUDA. This entry point wires the same countdown grader and the same on-disk
datasets to the device-agnostic ES loop in this package.

Run from the repository root so that `es_at_scale` resolves without installing
the package (installing it would pull vLLM, which has no macOS wheel):

    python -m mps_es.train_countdown --n-iterations 50
"""

import argparse
import json
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk  # noqa: E402

from es_at_scale.reward_function.countdown_grader import countdown_reward_fn  # noqa: E402
from mps_es.checkpoint import (  # noqa: E402
    load_checkpoint,
    make_rng,
    next_seeds,
    save_checkpoint,
)
from mps_es.runner import load_runner, resolve_device  # noqa: E402
from mps_es.status import RunStatus  # noqa: E402
from mps_es.trainer import ESHyperparams, run_iteration  # noqa: E402

__all__ = [
    "BatchScore",
    "TimedGrader",
    "score_batch",
    "score_responses",
    "write_eval_outputs",
    "to_prompts_and_targets",
    "main",
]


# TODO(refactor): BatchScore, TimedGrader, summarise_records, write_eval_outputs and eval_standard_error are task-agnostic; move them to a shared module instead of having train_math import them from this entry point.
@dataclass
class BatchScore:
    mean_reward: float
    accuracy: float
    mean_format_reward: float = 0.0
    mean_answer_reward: float = 0.0


def _worker_ready() -> bool:
    return True


class TimedGrader:
    """Runs a reward function in a worker process with a time limit.

    The countdown grader evaluates the model's arithmetic expression with
    Python, and a long enough expression can take unbounded time. Upstream runs
    the grader in a process pool and scores a timed-out response as zero. This
    does the same. A worker left busy by a timed-out call is replaced.
    """

    TIMED_OUT = {
        "formatted": False,
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "timed_out": True,
    }

    def __init__(self, fn, timeout: float):
        self._fn = fn
        self._timeout = timeout
        self._ctx = multiprocessing.get_context("spawn")
        self._pool = self._start_pool()

    def _start_pool(self):
        pool = self._ctx.Pool(1)
        pool.apply(_worker_ready)  # wait for the worker to start, so the limit covers grading only
        return pool

    def __call__(self, response: str, target: Dict[str, Any]):
        pending = self._pool.apply_async(self._fn, (response, target))
        try:
            return pending.get(self._timeout)
        except multiprocessing.TimeoutError:
            self._pool.terminate()
            self._pool = self._start_pool()
            return dict(self.TIMED_OUT), 0.0

    def close(self) -> None:
        self._pool.terminate()


# TODO(refactor): same loop as train_math.score_responses; only the format/answer extraction from the grader's detail differs, so pass that as a function.
def score_responses(
    responses: Sequence[str],
    targets: Sequence[Dict[str, Any]],
    grader=countdown_reward_fn,
) -> List[Dict[str, Any]]:
    """Grade each response and return one record per prompt.

    Each record carries the response, its target, the total reward and the two
    reward components. `score_batch` averages these; the eval step also writes
    them to disk so individual model outputs can be read later, which is what
    the upstream trainer does in its eval-output directory.
    """
    if len(responses) != len(targets):
        raise ValueError(
            f"got {len(responses)} responses for {len(targets)} targets"
        )
    records = []
    for response, target in zip(responses, targets):
        detail, reward = grader(response, target)
        records.append(
            {
                "target": target,
                "response": response,
                "reward": float(reward),
                "format_reward": float(detail["format_reward"]),
                "answer_reward": float(detail["answer_reward"]),
                "correct": detail["answer_reward"] == 1.0,
            }
        )
    return records


def score_batch(
    responses: Sequence[str],
    targets: Sequence[Dict[str, Any]],
    grader=countdown_reward_fn,
) -> BatchScore:
    """Score one batch of completions.

    `mean_reward` (0.1 * format + answer) is what ES optimises; `accuracy` is the
    exact-answer rate, i.e. the pass@1 number comparable to the paper.

    The two components are reported separately because the total alone cannot
    distinguish a model that has learned the `<think>`/`<answer>` envelope from
    one that is actually solving the puzzle -- format credit saturates at 0.1,
    so early progress is all envelope.
    """
    return summarise_records(score_responses(responses, targets, grader))


def summarise_records(records: Sequence[Dict[str, Any]]) -> BatchScore:
    """Average per-prompt records into one BatchScore."""
    if not records:
        return BatchScore(mean_reward=0.0, accuracy=0.0)
    n = len(records)
    return BatchScore(
        mean_reward=sum(r["reward"] for r in records) / n,
        accuracy=sum(int(r["correct"]) for r in records) / n,
        mean_format_reward=sum(r["format_reward"] for r in records) / n,
        mean_answer_reward=sum(r["answer_reward"] for r in records) / n,
    )


def write_eval_outputs(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Save every eval response and its grade as one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(list(records), f, indent=2)


def eval_standard_error(n_samples: int, accuracy: float = 0.1) -> float:
    """Roughly how much an eval accuracy reading can move by chance alone.

    Binomial standard error sqrt(p(1-p)/n). At n=100 and p=0.1 this is 3 points,
    which is large enough that a small eval set reads as a plateau when the
    underlying model is still improving.
    """
    if n_samples <= 0:
        return 0.0
    return (accuracy * (1.0 - accuracy) / n_samples) ** 0.5


def to_prompts_and_targets(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Dict]]:
    """Split dataset rows into prompts and grader targets (upstream's collate_fn)."""
    prompts = [row["context"] for row in rows]
    targets = [{"numbers": row["numbers"], "target": row["target"]} for row in rows]
    return prompts, targets


# TODO(refactor): the chunk, grade, progress loop is repeated in train_math.evaluate_splits; one generic evaluate over (prompts, targets) would serve both.
def evaluate_split(
    runner,
    prompts,
    targets,
    mini_batch_size,
    output_path: Path = None,
    grader=countdown_reward_fn,
    progress=None,
) -> BatchScore:
    """Greedily answer every prompt in the split and grade the answers.

    Generation runs in chunks of `mini_batch_size` to bound memory. The grade is
    computed over all prompts at once, so the chunk size does not change the
    result. When `output_path` is given, every response and its grade is
    written there as JSON. `progress(done, total)` is called after each chunk.
    """
    records: List[Dict[str, Any]] = []
    for start in range(0, len(prompts), mini_batch_size):
        chunk_prompts = prompts[start : start + mini_batch_size]
        chunk_targets = targets[start : start + mini_batch_size]
        responses = runner.generate(chunk_prompts, mini_batch_size=mini_batch_size)
        graded = score_responses(responses, chunk_targets, grader)
        for prompt, record in zip(chunk_prompts, graded):
            records.append({"prompt": prompt, **record})
        if progress is not None:
            progress(len(records), len(prompts))
    if output_path is not None:
        write_eval_outputs(output_path, records)
    return summarise_records(records)


# TODO(refactor): identical to train_math.build_parser apart from defaults and --eval-splits; build both from one parser with per-task defaults.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ES fine-tuning for Countdown on Apple Silicon / CPU / CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Flag names match es_at_scale/train.py.
    #
    # The training steps are the same as upstream. Only some default values are
    # smaller, to suit a single device:
    #
    #   flag              upstream default              default here
    #   --model-name      Qwen/Qwen2.5-1.5B-Instruct    Qwen/Qwen2.5-0.5B-Instruct
    #   --n-iterations    300                           50
    #   --batch-size      200 (the whole train split)   32
    #   --max-tokens      512                           256
    #   --eval-samples    all 2000 eval prompts         500
    #
    # Pass the upstream values on the command line to run the full configuration.
    p.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=-1, help="defaults to sigma/2")
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32,
                   help="prompts each population member is scored on")
    p.add_argument("--mini-batch-size", type=int, default=32,
                   help="memory lever only; does not change the result")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--eval-freq", type=int, default=10, help="0 disables evaluation")
    # Upstream evaluates on the whole eval split. A subset is used here to keep
    # eval time down. The run prints the sampling noise of the chosen size at
    # startup; too small a subset reads as a plateau.
    p.add_argument("--eval-samples", type=int, default=500,
                   help="prompts from the 2000-row eval split; -1 for all")
    p.add_argument("--train-dataset", default=str(REPO_ROOT / "datasets/train/countdown"))
    p.add_argument("--eval-dataset",
                   default=str(REPO_ROOT / "datasets/evaluation_suite/countdown"))
    p.add_argument("--output-directory", default=str(REPO_ROOT / "experiments"))
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reward-function-timeout", type=int, default=10,
                   help="seconds before a grader call is scored as zero; 0 grades inline")
    # additions over upstream
    p.add_argument("--device", default=None, help="mps / cuda / cpu (autodetected)")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--resume", default=None, help="path to a checkpoint to continue")
    p.add_argument("--decorrelate-layers", action="store_true",
                   help="give same-shaped layers independent noise (upstream does not)")
    p.add_argument("--checkpoint-freq", type=int, default=10)
    return p


# TODO(refactor): everything in main mirrors train_math.main (setup, resume, status, logging, loop, checkpoints); factor a task-agnostic run(args, task) that the two entry points parameterise.
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    alpha = args.sigma / 2 if args.alpha < 0 else args.alpha
    device = resolve_device(args.device)

    name = args.experiment_name or f"countdown-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_directory) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    checkpoint_path = out_dir / "checkpoint.pt"

    train_rows = list(load_from_disk(args.train_dataset)["train"])
    eval_split = load_from_disk(args.eval_dataset)
    eval_rows = list(eval_split[next(iter(eval_split.keys()))])
    if args.eval_samples >= 0 and args.eval_samples < len(eval_rows):
        print(f"[eval] using {args.eval_samples} of {len(eval_rows)} eval prompts")
        eval_rows = eval_rows[: args.eval_samples]
    if args.eval_freq:
        se = eval_standard_error(len(eval_rows)) * 100
        print(f"[eval] {len(eval_rows)} prompts, +/- {se:.1f} points of noise "
              f"at ~10% accuracy")
        if se > 2.0:
            print("[eval] that is coarse enough to look like a plateau; "
                  "raise --eval-samples to track progress")

    train_prompts, train_targets = to_prompts_and_targets(train_rows)
    eval_prompts, eval_targets = to_prompts_and_targets(eval_rows)
    batch_size = min(args.batch_size, len(train_prompts))
    if batch_size < args.batch_size:
        print(f"[train] batch size capped at {batch_size} (train split size)")

    print(f"[setup] device={device} precision={args.precision} model={args.model_name}")
    print(f"[setup] pop={args.population_size} sigma={args.sigma} alpha={alpha} "
          f"batch={batch_size} max_tokens={args.max_tokens}")
    # run.json and status.json in the run directory are what `mps_es.watch`
    # reads to show progress between metrics rows.
    status = RunStatus(out_dir)
    status.update("loading", model_name=args.model_name)
    runner = load_runner(args.model_name, device, args.precision, args.max_tokens)
    params = runner.trainable_params()

    hp = ESHyperparams(
        sigma=args.sigma,
        alpha=alpha,
        population_size=args.population_size,
        decorrelate_layers=args.decorrelate_layers,
    )

    start_iteration, rng = 0, make_rng(args.seed)
    if args.resume:
        state = load_checkpoint(args.resume)
        runner.model.load_state_dict(state["model_state_dict"])
        runner.model.to(device)
        params = runner.trainable_params()
        start_iteration, rng = state["iteration"], state["rng"]
        print(f"[resume] continuing from iteration {start_iteration} ({args.resume})")

    config = {k: v for k, v in vars(args).items()}
    config["alpha"] = alpha
    status.start(
        task="countdown", model_name=args.model_name, device=device,
        precision=args.precision, n_iterations=args.n_iterations,
        start_iteration=start_iteration, population_size=args.population_size,
        batch_size=batch_size, max_tokens=args.max_tokens, eval_freq=args.eval_freq,
        eval_samples=len(eval_prompts), resumed_from=args.resume,
    )

    grader = countdown_reward_fn
    if args.reward_function_timeout > 0:
        grader = TimedGrader(countdown_reward_fn, args.reward_function_timeout)

    def log(record: Dict[str, Any]) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Every eval writes each response and its grade to this directory, as the
    # upstream trainer does, so individual outputs can be inspected later.
    eval_dir = out_dir / "eval-output"

    # Upstream evaluates the untouched model before the first update. Do the
    # same, so the first eval after training has a baseline to compare against.
    # Skipped on resume because the baseline was recorded by the original run.
    def run_eval(tag: str, iteration: int) -> BatchScore:
        status.update("eval", tag=tag, iteration=iteration, done=0, total=len(eval_prompts))
        return evaluate_split(runner, eval_prompts, eval_targets, args.mini_batch_size,
                              output_path=eval_dir / f"eval_{tag}.json", grader=grader,
                              progress=lambda done, total: status.progress(done=done, total=total))

    if args.eval_freq and start_iteration == 0:
        score = run_eval("baseline", 0)
        log({"iteration": 0, "baseline": True,
             "eval_accuracy": score.accuracy, "eval_mean_reward": score.mean_reward})
        print(f"[eval base] accuracy={score.accuracy:.4f} "
              f"mean_reward={score.mean_reward:.4f}")

    # a fixed slice per iteration, as upstream does: every member sees the same prompts
    cursor = (start_iteration * batch_size) % max(1, len(train_prompts))

    for iteration in range(start_iteration, args.n_iterations):
        started = time.time()
        window = [
            (cursor + i) % len(train_prompts) for i in range(batch_size)
        ]
        prompts = [train_prompts[i] for i in window]
        targets = [train_targets[i] for i in window]
        cursor = (cursor + batch_size) % len(train_prompts)

        # TODO(refactor): per-member bookkeeping is rebuilt around run_iteration here and in train_math; let run_iteration report each member's score through a callback.
        member_scores: List[BatchScore] = []
        status.update("train", iteration=iteration, member=0, population=hp.population_size)

        def evaluate() -> float:
            status.progress(member=len(member_scores) + 1)
            responses = runner.generate(prompts, mini_batch_size=args.mini_batch_size)
            score = score_batch(responses, targets, grader)
            member_scores.append(score)
            return score.mean_reward

        seeds = next_seeds(rng, hp.population_size)
        result = run_iteration(params, seeds, evaluate, hp)
        elapsed = time.time() - started

        n_members = max(1, len(member_scores))
        mean_format = sum(s.mean_format_reward for s in member_scores) / n_members
        mean_answer = sum(s.mean_answer_reward for s in member_scores) / n_members
        best_accuracy = max((s.accuracy for s in member_scores), default=0.0)

        record = {
            "iteration": iteration,
            "mean_reward": result.mean_reward,
            "max_reward": result.max_reward,
            "mean_format_reward": mean_format,
            "mean_answer_reward": mean_answer,
            "best_member_accuracy": best_accuracy,
            "seconds": round(elapsed, 1),
        }
        print(f"[iter {iteration:4d}] mean_reward={result.mean_reward:.4f} "
              f"max={result.max_reward:.4f} | fmt={mean_format:.3f} "
              f"ans={mean_answer:.4f} best_acc={best_accuracy:.3f} ({elapsed:.0f}s)")

        if args.eval_freq and (iteration + 1) % args.eval_freq == 0:
            score = run_eval(f"iteration{iteration + 1}", iteration + 1)
            record["eval_accuracy"] = score.accuracy
            record["eval_mean_reward"] = score.mean_reward
            print(f"[eval {iteration:4d}] accuracy={score.accuracy:.4f} "
                  f"mean_reward={score.mean_reward:.4f}")

        log(record)

        if args.checkpoint_freq and (iteration + 1) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, iteration=iteration + 1,
                            model=runner.model, rng=rng, config=config)
            print(f"[ckpt] saved at iteration {iteration + 1} -> {checkpoint_path}")

    save_checkpoint(checkpoint_path, iteration=args.n_iterations,
                    model=runner.model, rng=rng, config=config)
    status.update("done", iteration=args.n_iterations)
    if isinstance(grader, TimedGrader):
        grader.close()
    print(f"[done] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
