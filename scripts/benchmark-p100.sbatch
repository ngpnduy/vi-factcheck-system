#!/bin/bash
#SBATCH --job-name=vifact-backbone
#SBATCH --ntasks=1
set -euo pipefail

if (( $# < 4 )); then
    echo "Usage: sbatch [Slurm flags] scripts/benchmark-p100.sbatch smoke|run mbert|xlmr OUTPUT CONDA_PREFIX [benchmark options]" >&2
    exit 2
fi
action=$1
backbone=$2
output=$3
environment=$4
shift 4

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$output"
export PATH="$environment/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src"

"$environment/bin/python" -u -m vifactcheck benchmark "$action" \
    --backbone "$backbone" --output "$output" "$@" \
    2>&1 | tee -a "$output/slurm-${SLURM_JOB_ID:-manual}.log"
