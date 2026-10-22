#!/bin/bash
#SBATCH --account=naiss2026-4-1390-cpu
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=posterior_kl_%j.out

# Corrected KLD KF of Table 1 from the saved checkpoints, and the true-model
# reference (CPU only).
# Usage (from the repository root): sbatch scripts/posterior_kl.sh [run_dir]

RUN_DIR=${1:-runs/linear}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
CONTAINER=/nobackup/proj/disk/naiss2025-22-438/apps/container.sif
apptainer exec --bind /nobackup "${CONTAINER}" python3 analysis/posterior_kl.py \
    "${RUN_DIR}"/hard_10db "${RUN_DIR}"/soft_10db "${RUN_DIR}"/erot_10db \
    "${RUN_DIR}"/dsr_10db --num_runs 3
apptainer exec --bind /nobackup "${CONTAINER}" python3 analysis/posterior_kl.py \
    --oracle configs/linear/hard_10db.gin configs/linear/soft_10db.gin \
    configs/linear/erot_10db.gin configs/linear/dsr_10db.gin
