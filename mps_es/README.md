# mps_es: ES fine-tuning on Apple Silicon

Single-process port of the Evolution Strategies loop for Apple Silicon (MPS),
CPU, or one CUDA GPU. Same algorithm as `es_at_scale/`, without vLLM and Ray.
Nothing outside this directory is changed.

vLLM has no macOS wheel, so the repo's own trainer cannot be installed on a Mac.
This module runs the ES loop directly with `transformers` instead.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r mps_es/requirements.txt
```

## Run

From the repository root:

```bash
python -m mps_es.train_countdown --n-iterations 50
python -m mps_es.train_countdown --n-iterations 200 --resume experiments/<name>/checkpoint.pt
python -m mps_es.train_math --model-name Qwen/Qwen2.5-Math-7B --n-iterations 0   # baseline eval only
python -m mps_es.train_math --model-name Qwen/Qwen2.5-Math-7B --n-iterations 100
python -m mps_es.watch experiments/<name> -f        # live status; -v for detail
python -m pytest mps_es/tests -q
```

Output goes to `experiments/<name>/`: `metrics.jsonl`, `checkpoint.pt`,
`eval-output/*.json` with every eval response and its grade, and two small
files the watcher reads: `run.json` (settings) and `status.json` (current
phase and progress). The untouched model
is evaluated once before training starts. Resume restores the weights, the
iteration counter and the seed stream.

Both upstream tasks are here: `train_countdown` and `train_math`. The math
task uses the repo's boxed-answer grader and Qwen math template, trains on the
MATH level 3 to 5 split and evaluates each split of the math suite separately,
with `math500` as the headline. Its reward is binary, so the starting model
must already solve some training problems; run with `--n-iterations 0` first
to see the baseline.

## Same as upstream

- Each iteration: perturb, greedy generate, grade with the repo's countdown
  grader, restore, z-score, update.
- Same sigma, alpha, population size and update rule. Noise is added in the
  model's own dtype and same-shaped layers share noise, exactly as upstream.
- Same datasets, read in place, never copied.

## Different from upstream

Only default values: a smaller model, fewer prompts per iteration, shorter
completions, fewer iterations and a subset of the eval split, to suit a single
device. The comment above the flag definitions in `train_countdown.py` lists
each default next to upstream's, and `--help` shows the current values. Pass
upstream's values on the command line to run the full configuration.

Extra flags: `--device`, `--precision` (bf16 default; fp32 removes perturb and
restore rounding drift at the cost of memory), `--resume`, `--checkpoint-freq`,
`--decorrelate-layers` (independent noise per layer; upstream does not do this).

## Notes

- Eval accuracy is a proportion, so a small eval set is noisy enough to look
  like a plateau. The run prints its own resolution at startup; raise
  `--eval-samples` if it warns.
- Format credit is capped far below answer credit, so any population member
  scoring above the format cap has solved at least one puzzle. The watcher flags
  this.
- For another task, copy `train_countdown.py` and replace the grader, the prompt
  extraction and the dataset paths. The ES core knows nothing about countdown.
- `es_loop.py` repeats the perturb, restore and update maths from upstream's
  vLLM worker, which cannot be imported outside vLLM. Moving that maths into a
  shared module is a follow-up change.
