#!/usr/bin/env python3
"""ES fine-tuning on the MATH task, on Apple Silicon (or CPU, or CUDA).

Single-process counterpart to `es_at_scale/train.py --task math`, which requires
vLLM, Ray and CUDA. This entry point wires the same boxed-answer grader, the
same Qwen math prompt template and the same on-disk datasets to the
device-agnostic ES loop in this package.

The reward is binary: 1.0 when the final `\\boxed{}` answer is correct, 0.0
otherwise. A boxed but wrong answer earns nothing, so there is no format credit
to climb first as there is in countdown. The starting model therefore has to
solve some of the training problems already, or every population member
scores zero and ES has no signal.

Run from the repository root so that `es_at_scale` resolves without installing
the package (installing it would pull vLLM, which has no macOS wheel):

    python -m mps_es.train_math --model-name Qwen/Qwen2.5-Math-7B --n-iterations 0
    python -m mps_es.train_math --model-name Qwen/Qwen2.5-Math-7B --n-iterations 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_from_disk  # noqa: E402

from es_at_scale.reward_function.math_grader import boxed_reward_fn  # noqa: E402
from es_at_scale.template_function.apply_template import qwen_math_template  # noqa: E402
from mps_es.checkpoint import (  # noqa: E402
    load_checkpoint,
    make_rng,
    next_seeds,
    save_checkpoint,
)
from mps_es.runner import load_runner, resolve_device  # noqa: E402
from mps_es.status import RunStatus  # noqa: E402
# TODO(refactor): shared helpers imported from the countdown entry point; see the note on BatchScore in train_countdown.
from mps_es.train_countdown import (  # noqa: E402
    BatchScore,
    TimedGrader,
    eval_standard_error,
    summarise_records,
    write_eval_outputs,
)
from mps_es.trainer import ESHyperparams, run_iteration  # noqa: E402

__all__ = [
    "score_responses",
    "score_batch",
    "to_prompts_and_targets",
    "evaluate_splits",
    "main",
]

ALL_EVAL_SPLITS = "math500,amc,aime,minerva,olympiad_bench"


# TODO(refactor): same loop as train_countdown.score_responses; see the note there.
def score_responses(
    responses: Sequence[str],
    targets: Sequence[Any],
    grader=boxed_reward_fn,
) -> List[Dict[str, Any]]:
    """Grade each response and return one record per prompt.

    The record layout matches countdown's so the watcher and the eval JSON
    files look the same for both tasks. `format_reward` here records whether a
    boxed answer was found at all. It is reported only, never rewarded, exactly
    as upstream's math grader does.
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
                "format_reward": 1.0 if detail.get("formatted") else 0.0,
                "answer_reward": float(reward),
                "correct": float(reward) == 1.0,
            }
        )
    return records


def score_batch(
    responses: Sequence[str],
    targets: Sequence[Any],
    grader=boxed_reward_fn,
) -> BatchScore:
    """Score one batch of completions. `mean_reward` and `accuracy` coincide
    for a binary reward; both are kept so the metrics file matches countdown's."""
    return summarise_records(score_responses(responses, targets, grader))


def to_prompts_and_targets(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Any]]:
    """Wrap each problem in the Qwen math chat template (upstream's collate_fn
    plus its template_function) and keep the reference answer as the target."""
    prompts = [qwen_math_template(row["problem"]) for row in rows]
    targets = [row["answer"] for row in rows]
    return prompts, targets


# TODO(refactor): the inner chunk, grade, progress loop duplicates train_countdown.evaluate_split; see the note there.
def evaluate_splits(
    runner,
    splits: Dict[str, Tuple[List[str], List[Any]]],
    mini_batch_size: int,
    eval_dir: Path = None,
    tag: str = "",
    grader=boxed_reward_fn,
    progress=None,
) -> Dict[str, BatchScore]:
    """Greedily answer every prompt in every split and grade the answers.

    Upstream evaluates each split of the math suite separately and reports it
    under its own name. This does the same. When `eval_dir` is given, every
    response and its grade is written to `<eval_dir>/eval_<tag>_<split>.json`.
    `progress(split, done, total)` is called after each chunk.
    """
    scores: Dict[str, BatchScore] = {}
    for name, (prompts, targets) in splits.items():
        records: List[Dict[str, Any]] = []
        for start in range(0, len(prompts), mini_batch_size):
            chunk_prompts = prompts[start : start + mini_batch_size]
            chunk_targets = targets[start : start + mini_batch_size]
            responses = runner.generate(chunk_prompts, mini_batch_size=mini_batch_size)
            graded = score_responses(responses, chunk_targets, grader)
            for prompt, record in zip(chunk_prompts, graded):
                records.append({"prompt": prompt, **record})
            if progress is not None:
                progress(name, len(records), len(prompts))
        if eval_dir is not None:
            write_eval_outputs(Path(eval_dir) / f"eval_{tag}_{name}.json", records)
        scores[name] = summarise_records(records)
    return scores


def eval_record(scores: Dict[str, BatchScore], headline: str) -> Dict[str, float]:
    """Metrics-file fields for one eval: a per-split accuracy each, and
    `eval_accuracy` for the headline split so the watcher has one line to follow."""
    record = {f"eval_accuracy_{name}": s.accuracy for name, s in scores.items()}
    if headline in scores:
        record["eval_accuracy"] = scores[headline].accuracy
        record["eval_mean_reward"] = scores[headline].mean_reward
    return record


# TODO(refactor): identical to train_countdown.build_parser apart from defaults and --eval-splits; see the note there.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ES fine-tuning for MATH on Apple Silicon / CPU / CUDA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Flag names match es_at_scale/train.py.
    #
    # The training steps are the same as upstream. Only some default values are
    # smaller, to suit a single device:
    #
    #   flag              upstream (README math example)   default here
    #   --model-name      Qwen/Qwen2.5-Math-7B             Qwen/Qwen2.5-Math-1.5B
    #   --n-iterations    500                              100
    #   --batch-size      1024                             128
    #   --max-tokens      3000                             1024
    #   --eval-samples    every problem in every split     200 per split
    #
    # Pass the upstream values on the command line to run the full configuration.
    # Batch size matters more here than in countdown: the reward is binary, so a
    # member's score moves in steps of 1/batch, and a small batch cannot tell
    # the population members apart.
    p.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B")
    p.add_argument("--sigma", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=-1, help="defaults to sigma/2")
    p.add_argument("--population-size", type=int, default=30)
    p.add_argument("--n-iterations", type=int, default=100,
                   help="0 runs the baseline eval only")
    p.add_argument("--batch-size", type=int, default=128,
                   help="prompts each population member is scored on")
    p.add_argument("--mini-batch-size", type=int, default=16,
                   help="memory lever only; does not change the result")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--eval-freq", type=int, default=10, help="0 disables evaluation")
    p.add_argument("--eval-splits", default=ALL_EVAL_SPLITS,
                   help="comma-separated splits of the eval suite; the first is the headline")
    p.add_argument("--eval-samples", type=int, default=200,
                   help="problems per eval split; -1 for all")
    p.add_argument("--train-dataset",
                   default=str(REPO_ROOT / "datasets/train/math_lvl3to5_8k"))
    p.add_argument("--eval-dataset",
                   default=str(REPO_ROOT / "datasets/evaluation_suite/math"))
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


# TODO(refactor): train_countdown caps its eval rows inline in main with the same rule; share this loader.
def load_eval_splits(path: str, names: Sequence[str], samples: int):
    """Load the requested splits of the eval suite, capped at `samples` each."""
    suite = load_from_disk(path)
    splits = {}
    for name in names:
        if name not in suite:
            raise SystemExit(f"eval split {name!r} not in {path}; have {sorted(suite.keys())}")
        rows = list(suite[name])
        if 0 <= samples < len(rows):
            rows = rows[:samples]
        splits[name] = to_prompts_and_targets(rows)
    return splits


# TODO(refactor): mirrors train_countdown.main; see the note there.
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    alpha = args.sigma / 2 if args.alpha < 0 else args.alpha
    device = resolve_device(args.device)

    name = args.experiment_name or f"math-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(args.output_directory) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    checkpoint_path = out_dir / "checkpoint.pt"

    train_rows = list(load_from_disk(args.train_dataset)["train"])
    train_prompts, train_targets = to_prompts_and_targets(train_rows)
    batch_size = min(args.batch_size, len(train_prompts))
    if batch_size < args.batch_size:
        print(f"[train] batch size capped at {batch_size} (train split size)")

    split_names = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
    headline = split_names[0] if split_names else ""
    eval_splits = {}
    if args.eval_freq:
        eval_splits = load_eval_splits(args.eval_dataset, split_names, args.eval_samples)
        for split, (prompts, _) in eval_splits.items():
            se = eval_standard_error(len(prompts)) * 100
            print(f"[eval] {split}: {len(prompts)} problems, +/- {se:.1f} points of noise "
                  f"at ~10% accuracy")
        print(f"[eval] headline split: {headline}")

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
        task="math", model_name=args.model_name, device=device,
        precision=args.precision, n_iterations=args.n_iterations,
        start_iteration=start_iteration, population_size=args.population_size,
        batch_size=batch_size, max_tokens=args.max_tokens, eval_freq=args.eval_freq,
        eval_splits={name: len(prompts) for name, (prompts, _) in eval_splits.items()},
        headline_split=headline, resumed_from=args.resume,
    )

    # The symbolic checker behind the grader can take unbounded time on odd
    # LaTeX, so it runs in a worker with a time limit, as upstream does.
    grader = boxed_reward_fn
    if args.reward_function_timeout > 0:
        grader = TimedGrader(boxed_reward_fn, args.reward_function_timeout)

    def log(record: Dict[str, Any]) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    # Every eval writes each response and its grade to this directory, as the
    # upstream trainer does, so individual outputs can be inspected later.
    eval_dir = out_dir / "eval-output"

    def run_eval(tag: str, iteration: int) -> Dict[str, float]:
        status.update("eval", tag=tag, iteration=iteration, split="", done=0, total=0)
        scores = evaluate_splits(
            runner, eval_splits, args.mini_batch_size, eval_dir=eval_dir, tag=tag,
            grader=grader,
            progress=lambda split, done, total: status.progress(split=split, done=done, total=total),
        )
        for split, s in scores.items():
            print(f"[eval {tag}] {split}: accuracy={s.accuracy:.4f} "
                  f"boxed={s.mean_format_reward:.3f}")
        return eval_record(scores, headline)

    # Upstream evaluates the untouched model before the first update. Do the
    # same, so the first eval after training has a baseline to compare against.
    # Skipped on resume because the baseline was recorded by the original run.
    if args.eval_freq and start_iteration == 0:
        log({"iteration": 0, "baseline": True, **run_eval("baseline", 0)})

    # a fixed slice per iteration, as upstream does: every member sees the same prompts
    cursor = (start_iteration * batch_size) % max(1, len(train_prompts))

    for iteration in range(start_iteration, args.n_iterations):
        started = time.time()
        window = [(cursor + i) % len(train_prompts) for i in range(batch_size)]
        prompts = [train_prompts[i] for i in window]
        targets = [train_targets[i] for i in window]
        cursor = (cursor + batch_size) % len(train_prompts)

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
        mean_boxed = sum(s.mean_format_reward for s in member_scores) / n_members
        best_accuracy = max((s.accuracy for s in member_scores), default=0.0)

        record = {
            "iteration": iteration,
            "mean_reward": result.mean_reward,
            "max_reward": result.max_reward,
            "mean_format_reward": mean_boxed,
            "mean_answer_reward": result.mean_reward,
            "best_member_accuracy": best_accuracy,
            "seconds": round(elapsed, 1),
        }
        print(f"[iter {iteration:4d}] mean_reward={result.mean_reward:.4f} "
              f"max={result.max_reward:.4f} | boxed={mean_boxed:.3f} "
              f"best_acc={best_accuracy:.3f} ({elapsed:.0f}s)")

        if args.eval_freq and (iteration + 1) % args.eval_freq == 0:
            record.update(run_eval(f"iteration{iteration + 1}", iteration + 1))

        log(record)

        if args.checkpoint_freq and (iteration + 1) % args.checkpoint_freq == 0:
            save_checkpoint(checkpoint_path, iteration=iteration + 1,
                            model=runner.model, rng=rng, config=config)
            print(f"[ckpt] saved at iteration {iteration + 1} -> {checkpoint_path}")

    # A baseline-only run (--n-iterations 0) changes no weights, so there is
    # nothing worth a checkpoint; for a large model that file is many gigabytes.
    if args.n_iterations > start_iteration:
        save_checkpoint(checkpoint_path, iteration=args.n_iterations,
                        model=runner.model, rng=rng, config=config)
    status.update("done", iteration=args.n_iterations)
    if isinstance(grader, TimedGrader):
        grader.close()
    print(f"[done] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
