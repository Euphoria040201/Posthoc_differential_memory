#!/bin/bash
# Submit from the repository root after loading the modules used by your venv.
# sbatch --account=ACCOUNT scripts/slurm/differential_narval.sbatch \
#   pilot /absolute/model /absolute/data.json /absolute/run_root /absolute/venv
# For main, also pass --array=0-15%2 to sbatch (before the script path).
#SBATCH --job-name=diff-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-5%2
#SBATCH --output=slurm-diff-%A_%a.log

set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: sbatch --account=ACCOUNT $0 pilot|main MODEL MANIFEST RUN_ROOT VENV" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit with sbatch, not bash on a login node}"
: "${SLURM_ARRAY_TASK_ID:?A job array is required}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible GPU}"

phase=$1
model_path=$2
manifest_path=$3
run_root=$4
venv_path=$5
task_id=$SLURM_ARRAY_TASK_ID

for path in "$model_path" "$manifest_path" "$run_root" "$venv_path"; do
    [[ "$path" = /* ]] || { echo "Use absolute paths: $path" >&2; exit 2; }
done
[[ -d "$model_path" && -f "$model_path/config.json" ]] || {
    echo "Local model/config.json missing: $model_path" >&2; exit 2;
}
[[ -f "$manifest_path" && -x "$venv_path/bin/python" ]] || {
    echo "Prepared manifest or virtual-environment Python is missing" >&2; exit 2;
}
[[ "$task_id" =~ ^[0-9]+$ ]] || { echo "Invalid task ID" >&2; exit 2; }
arms=(base add sub add_aug sub_aug sub_nui)
case "$phase" in
    pilot)
        (( task_id < 6 )) || { echo "Pilot requires --array=0-5%2" >&2; exit 2; }
        arm=${arms[$task_id]}
        seed=0
        ;;
    main)
        [[ ${SLURM_ARRAY_TASK_COUNT:-0} -eq 16 ]] || {
            echo "Main requires --array=0-15%2; do not use the pilot default" >&2; exit 2;
        }
        (( task_id < 16 )) || { echo "Invalid main task ID" >&2; exit 2; }
        if (( task_id == 0 )); then
            arm=base
            seed=0
        else
            arm=${arms[$(( (task_id - 1) / 3 + 1 ))]}
            seed=$(( (task_id - 1) % 3 ))
        fi
        ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac

cd "$SLURM_SUBMIT_DIR"
[[ -f scripts/run_differential_v2.py ]] || {
    echo "Submit from the Posthoc_differential_memory repository root" >&2; exit 2;
}
source "$venv_path/bin/activate"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python - <<'PY'
import os
import torch
import transformers
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expected exactly one allocated CUDA GPU")
print("torch:", torch.__version__, "transformers:", transformers.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
print("GPU:", torch.cuda.get_device_name(0))
PY

printf 'phase=%s arm=%s seed=%s job=%s\n' "$phase" "$arm" "$seed" "$SLURM_JOB_ID"
# Pass the actual Slurm device token through; never overwrite it with a guessed 0.
python scripts/run_differential_v2.py run \
    --model-path "$model_path" --data-manifest "$manifest_path" \
    --output-root "$run_root" --phase "$phase" --arms "$arm" --seeds "$seed" \
    --gpus "$CUDA_VISIBLE_DEVICES"
