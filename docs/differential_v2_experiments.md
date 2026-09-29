# Differential audit fixes and experiment plan

This protocol supersedes the old Qasper numbers. Do not mix v1 and v2 scores in
one comparison. All quality results below are **planned**, not measured by the
CPU integration run. Target submission: October 12, 2026.

## Questions and minimum comparisons

The first question is whether a nuisance objective improves a frozen-backbone
sidecar beyond the same parameter budget and data augmentation. A plus/minus
comparison alone cannot establish a new function class: a free output projection
can absorb a sign change.

| Arm | Forward | Training contexts | Objective | What it controls |
| --- | --- | --- | --- | --- |
| base | frozen Qwen | no training | none | unchanged backbone |
| add | Y + 0.1 C | original, repeated K times | QA | extra trainable capacity |
| sub | Y - 0.1 C | original, repeated K times | QA | sign under the same architecture |
| add_aug | Y + 0.1 C | original + shuffled chunks | QA | augmentation without nuisance supervision |
| sub_aug | Y - 0.1 C | original + shuffled chunks | QA | exact architecture/data/compute control for sub_nui |
| sub_nui | Y - 0.1 C | original + shuffled chunks | QA + nuisance + invariance | proposed objective |

All five sidecar arms use exactly the same modules, trainable parameter names,
initialization seed, data order, optimizer, learning rate, coefficient, and token
budget. Augmentation randomness is separate from data-order randomness. The
non-augmented arms still process K copies per group, so processed sequence counts
match. The primary contrast is **sub_nui minus sub_aug**; the practical contrast is
**sub_nui minus add_aug**. `attn_only` is an optional larger-capacity reference,
not a parameter-matched baseline.

The nuisance target is the group-centered Y **at the final query token, before
any answer token**. The target is matched to the actual correction, 0.1 C, rather
than to unscaled C. No context-position residuals or teacher-forced answer
representations enter the auxiliary losses. Raw MSE is averaged over features
and layers; beta=gamma=1 are fixed for the initial comparison. A pilot can expose
loss-scale problems, but tune any changed setting using validation only and
record a new experiment configuration before the main runs.

## Data protocol

`prepare_differential_data.py` creates a checksummed tokenized manifest shared by
every arm. It:

- keeps the natural prefix of each document, with no gold-based retrieval;
- defaults to whole paragraphs (`max_chunk_tok=0`) and a 4,500-token context cap;
- rejects questions whose complete annotated text evidence is no longer visible;
- keeps only full answers that fit within 24 tokens **including EOS**;
- retains the first annotation, matching the existing answer selection convention;
- requires at least two context chunks for every arm;
- reserves 32 eligible official-training papers for calibration, disjoint by
  paper ID from the 935-example training set;
- preserves the official validation and test split boundaries and audits all
  drops and selected IDs. It fails if the requested training count cannot be met.

This is an **evidence-visible short-answer Qasper subset**, not an official full
Qasper result. Filtering with annotations changes the population and must be
disclosed. The filtering algorithm and budgets must be locked before opening
test scores. Do not change data budgets selectively for one arm. If there are
fewer than 935 eligible examples, pick one smaller shared count (or larger shared
context budget), regenerate a new manifest, and rerun every arm.

## Execution

For Narval, use the [Slurm array setup](differential_narval.md), which preserves
Slurm-assigned GPU identifiers and runs one arm/seed per allocated GPU.

Use the project's existing GPU environment. CPU validation used torch 2.6.0 and
transformers 5.9.0; `requirements.txt` is the original environment freeze and
contains platform-specific CUDA/FlashAttention entries, so do not blindly install
it on an unrelated machine.

From the repository root, replace MODEL with the actual local Qwen directory:

```bash
export MODEL=/path/to/Qwen3-4B-Instruct-2507
export RUN_ROOT=runs/differential_v2

python scripts/prepare_differential_data.py \
  --model-path "$MODEL" --output "$RUN_ROOT/data.json"

python scripts/run_differential_v2.py plan \
  --model-path "$MODEL" --data-manifest "$RUN_ROOT/data.json" \
  --output-root "$RUN_ROOT" --phase pilot

python scripts/run_differential_v2.py run \
  --model-path "$MODEL" --data-manifest "$RUN_ROOT/data.json" \
  --output-root "$RUN_ROOT" --phase pilot --gpus 0,1

python scripts/run_differential_v2.py status \
  --model-path "$MODEL" --data-manifest "$RUN_ROOT/data.json" \
  --output-root "$RUN_ROOT" --phase pilot

python scripts/summarize_differential_v2.py \
  --directory "$RUN_ROOT/pilot" --output "$RUN_ROOT/pilot_summary.json"
```

Only list GPUs already allocated to this experiment. One process runs on each
listed GPU. The runner checks CUDA availability, logs each stage, strictly reloads
the checkpoint before robustness evaluation, and records COMPLETE only after all
expected artifacts exist and both subprocesses exit successfully. Rerun the same
command to resume completed stages. Changed source, data, or arguments require a
fresh output directory; old JSON files alone never count as a completed job.

**Pilot:** seed 0; 32 updates; K=2; eight groups/update (16 sequences/update);
constant sidecar LR=5e-4; 64 clean validation examples and 32 robustness examples.
This is a debugging/feasibility pass. Its prefix subset and single seed are not
publication evidence. Use recorded runtime and peak GPU memory before deciding
how many concurrent main runs fit. Do not select a winning seed from the pilot.

**Main:** run from fresh initialization, 156 updates, seeds 0/1/2, all retained
validation examples, same K and accumulation. The default matrix has 16 runs:
one base plus five sidecars for each of three seeds.

```bash
python scripts/run_differential_v2.py run \
  --model-path "$MODEL" --data-manifest "$RUN_ROOT/data.json" \
  --output-root "$RUN_ROOT" --phase main --gpus 0,1,2,3

python scripts/summarize_differential_v2.py \
  --directory "$RUN_ROOT/main" --output "$RUN_ROOT/main_summary.json"
```

There is no automatic promotion from pilot to main. Check finite losses,
nonzero sidecar gradients, exactly matched parameter counts, complete reload,
EOS stopping frequency, and the reported data drops first. If sub_nui fails to
improve on sub_aug/add_aug across seeds, do not attribute the ordinary sidecar's
gain to differential denoising.

## Readouts and interpretation

Report clean F1/EM, average F1 under three fixed paragraph permutations, worst
F1 across original+permutations, EOS stopping rate, generated length, trainable
parameters, runtime and peak GPU memory. Per-example sample IDs and permutations
are shared across arms. The summarizer rejects mismatched IDs, gold references,
permutations, seed sets or sidecar parameter counts. Report every seed and mean
± seed standard deviation, plus descriptive paired bootstrap intervals over
training seeds and paper clusters. Single-seed intervals exclude training-seed
uncertainty and cannot establish robust superiority.

The restored-model evaluator also measures pre/post-correction variance at the
aligned query endpoint. Lower variance is insufficient by itself: constant or
less evidence-sensitive representations can have lower variance. Require
preserved/improved task accuracy, and before claiming a denoising mechanism add
the repository's synthetic evidence-swap probe (changing the relevant fact must
still change the answer). Paragraph reordering is an order-sensitivity stress
test; it is not guaranteed semantically neutral for every natural document.

## Optional frozen-control mechanism check

Only after the matched comparison is useful, freeze each chosen sidecar and run
fixed_add/fixed_sub/learned_diff/variance_diff. This asks how to use that particular
learned control; a sign flip of an add-trained branch is not a fair test of every
subtractive method. Run both an add-trained and a nuisance-trained source if
making a claim about branch semantics.

```bash
python scripts/dex_stage1_fusion.py \
  --model-path "$MODEL" --data-manifest "$RUN_ROOT/data.json" \
  --steer-ckpt "$RUN_ROOT/main/sub_nui_s0_steer.pt" \
  --arm variance_diff --calibrate-batches 64 \
  --output-dir "$RUN_ROOT/mechanism" --tag sub_nui_s0_variance
```

Variance calibration sees only prompts from the held-out training-paper split.
It collects moments along a fixed additive path, then freezes means **and** lambda.
No inference batch or future suffix changes the coefficient. Calibration buffers
are saved; missing calibration is an error. Repeat the optional diagnostic across
the same seeds. Do not optimize a coefficient using validation/test labels.

After all choices are locked, call `eval_differential_checkpoint.py --split test`
for **each prespecified main arm and seed**, including the base. Keep test scores
out of model/configuration selection. The evaluator accepts both complete DEX
`*_model.pt` and sidecar `*_steer.pt` checkpoints.

## Validation commands

```bash
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_dex.py tests/test_dex_swa_steer.py tests/test_o_fusion_position.py \
  tests/test_fusion_differential.py tests/test_differential_audit.py \
  tests/test_eval_steer_checkpoint_loaders.py tests/test_prefix_steer_shared_main_v.py \
  tests/test_prefix_steer_write_only.py

OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  python scripts/smoke_differential_v2.py
```

The offline smoke run creates a random tiny Qwen and a synthetic manifest, trains
all six pilot arms for two updates, reloads them, performs paired perturbation
evaluation, then exercises learned/variance fusion. Its scores are only execution
checks, never Qwen3-4B or Qasper quality results.
