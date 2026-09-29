# Differential experiments on Narval

The GPU experiments have **not** been submitted by the assistant. The user's SSH
session is on Narval; the assistant's execution workspace is separate.

Use the fixed branch `fix/differential-audit-20260929`. Load the same Alliance
modules used to create a CUDA-enabled virtual environment before submitting.
Do not blindly install the original `requirements.txt`: it includes a
machine-specific FlashAttention wheel. The default attention implementation is
SDPA. Confirm the environment with the offline smoke command in
[the experiment protocol](differential_v2_experiments.md).

First identify your Slurm account, local **unquantized**
`Qwen3-4B-Instruct-2507` directory (or agree on one shared alternative before all
arms), and virtual environment. Quantization results from other projects are not
differential results and their model checkpoints should not silently replace this
backbone. Inspect `config.json` for the architecture and any quantization config.

```bash
sacctmgr -nP show assoc where user="$USER" format=Account,Partition,QOS
module -t list
ls -ld "$HOME"/projects/* "$HOME"/scratch
```

Set these to real, absolute paths. No GPU account is inferred automatically.
Keep the repository and runs on storage visible from compute nodes.

```bash
export DIFF_ACCOUNT=YOUR_SLURM_ACCOUNT
export DIFF_MODEL=/absolute/path/to/Qwen3-4B-Instruct-2507
export DIFF_VENV=/absolute/path/to/venv
export DIFF_RUN_ROOT=/absolute/path/to/differential_v2
```

Before GPU submission, download/cache the model and Qasper and create the shared
manifest with `prepare_differential_data.py`. Use an appropriate CPU allocation
for bulk tokenization; pre-stage downloads on a node where network access is
available. GPU jobs are offline and require the completed manifest. Its creation
uses only the tokenizer, never the 4B model weights. See the protocol for the
evidence and answer filtering, and the insufficient-example failure.

```bash
source "$DIFF_VENV/bin/activate"
python scripts/prepare_differential_data.py \
  --model-path "$DIFF_MODEL" --output "$DIFF_RUN_ROOT/data.json"
```

Submit **from the repository root**. The pilot has six array tasks: base, add,
sub, add_aug, sub_aug, sub_nui, each seed 0. Each requests one full A100, six CPU
cores, 64 GB host memory and a six-hour time limit. At most two run simultaneously.
These are initial resource requests, not a measured runtime or memory guarantee.
Use measured pilot time and memory to size main jobs.

```bash
sbatch --account="$DIFF_ACCOUNT" scripts/slurm/differential_narval.sbatch \
  pilot "$DIFF_MODEL" "$DIFF_RUN_ROOT/data.json" "$DIFF_RUN_ROOT" "$DIFF_VENV"

squeue -u "$USER"
python scripts/run_differential_v2.py status \
  --model-path "$DIFF_MODEL" --data-manifest "$DIFF_RUN_ROOT/data.json" \
  --output-root "$DIFF_RUN_ROOT" --phase pilot
```

The batch file preserves the GPU identifier assigned by Slurm, including a UUID
or nonzero index. Logs are `slurm-diff-JOB_TASK.log` plus per-arm stage logs under
`$DIFF_RUN_ROOT/pilot`. Repeating the same submission resumes completed stages;
an interrupted training stage restarts that arm from initialization. Do not run
overlapping submissions that write the same arm/seed. Do not edit Python source
or the environment while an array is queued or running.

After all six tasks complete, summarize and inspect the diagnostics described in
the protocol before approving the main configuration:

```bash
python scripts/summarize_differential_v2.py \
  --directory "$DIFF_RUN_ROOT/pilot" --output "$DIFF_RUN_ROOT/pilot_summary.json"
```

Main is an explicit new submission, not automatic promotion. Its 16 tasks are
one base and five sidecars times three seeds. Override the array range; adjust the
time request using pilot measurements. No task accesses test scores.

```bash
sbatch --account="$DIFF_ACCOUNT" --job-name=diff-main --array=0-15%2 \
  scripts/slurm/differential_narval.sbatch \
  main "$DIFF_MODEL" "$DIFF_RUN_ROOT/data.json" "$DIFF_RUN_ROOT" "$DIFF_VENV"
```

Cluster references: [Narval](https://docs.alliancecan.ca/wiki/Narval) and
[Using GPUs with Slurm](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm).
