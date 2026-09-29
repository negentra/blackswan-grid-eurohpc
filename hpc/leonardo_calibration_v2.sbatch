#!/bin/bash
# REVIEW ONLY. Prospectively priced V2 matrix; never submit before host onboarding.
# Account/partition/QoS, one-GPU placement and charging require AI Factory confirmation.
#SBATCH --job-name=blackswan-connected-calibration-v2
#SBATCH --partition=boost_usr_prod
#SBATCH --account=REPLACE_WITH_PROJECT_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=leonardo-calibration-v2-%j.out
#SBATCH --error=leonardo-calibration-v2-%j.err
set -euo pipefail
: "${BLACKSWAN_PROJECT_ROOT:?set reviewed project root}"
: "${BLACKSWAN_VENV:?set reviewed Linux virtualenv path}"
: "${BLACKSWAN_OUTPUT_DIR:?set reviewed output path in WORK}"
: "${BLACKSWAN_ROW_INDEX:?set reviewed V2 row index 0..13}"
[[ "$BLACKSWAN_ROW_INDEX" =~ ^([0-9]|1[0-3])$ ]]
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
if (( BLACKSWAN_ROW_INDEX >= 12 )); then
  [[ "${BLACKSWAN_BF16_EQUIVALENCE_REVIEW:-}" == "PASS" ]] || { echo 'BF16 equivalence review required' >&2; exit 2; }
  bf16_flag=--allow-bf16
else
  bf16_flag=
fi
mkdir -p "$BLACKSWAN_OUTPUT_DIR"
trap 'code=$?; printf "{\"exit_code\":%s,\"job_id\":\"%s\",\"row_index\":%s}\n" "$code" "$SLURM_JOB_ID" "$BLACKSWAN_ROW_INDEX" > "$BLACKSWAN_OUTPUT_DIR/calibration-v2-exit-$SLURM_JOB_ID.json"' EXIT
cd "$BLACKSWAN_PROJECT_ROOT"
source "$BLACKSWAN_VENV/bin/activate"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader > "$BLACKSWAN_OUTPUT_DIR/calibration-v2-gpu-$SLURM_JOB_ID.csv"
python -m pip freeze > "$BLACKSWAN_OUTPUT_DIR/calibration-v2-packages-$SLURM_JOB_ID.txt"
srun python -m benchmarks.run_connected_grid_matrix --matrix results/playground_calibration_matrix_v2.csv --device cuda --row-index "$BLACKSWAN_ROW_INDEX" ${bf16_flag:+$bf16_flag} --output "$BLACKSWAN_OUTPUT_DIR/row-v2-$BLACKSWAN_ROW_INDEX-job-$SLURM_JOB_ID.json"
