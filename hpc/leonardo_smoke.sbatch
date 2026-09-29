#!/bin/bash
# Review-only template; do not submit until account, environment and output are approved.
# Synthetic development seed is fixed in the kernel as 19092026.
#SBATCH --job-name=blackswan-connected-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --account=REPLACE_WITH_PROJECT_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=leonardo-smoke-%j.out
#SBATCH --error=leonardo-smoke-%j.err
set -euo pipefail
: "${BLACKSWAN_PROJECT_ROOT:?set reviewed project root}"
: "${BLACKSWAN_VENV:?set reviewed Linux virtualenv path}"
: "${BLACKSWAN_OUTPUT_DIR:?set reviewed output path in WORK}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
mkdir -p "$BLACKSWAN_OUTPUT_DIR"
trap 'code=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\"}\n" "$code" "$SLURM_JOB_ID" > "$BLACKSWAN_OUTPUT_DIR/smoke-exit-$SLURM_JOB_ID.json"' EXIT
cd "$BLACKSWAN_PROJECT_ROOT"
source "$BLACKSWAN_VENV/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$BLACKSWAN_OUTPUT_DIR/smoke-start-$SLURM_JOB_ID.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$BLACKSWAN_OUTPUT_DIR/smoke-gpu-$SLURM_JOB_ID.csv"
python -m pip freeze > "$BLACKSWAN_OUTPUT_DIR/smoke-packages-$SLURM_JOB_ID.txt"
srun python -m benchmarks.connected_grid_calibration --scale S1 --steps 2 --device cuda --output "$BLACKSWAN_OUTPUT_DIR/smoke-result-$SLURM_JOB_ID.json"
