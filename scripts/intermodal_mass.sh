#!/bin/bash
#SBATCH --account=naiss2026-4-1390-cpu
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=intermodal_mass_%j.out

# Paper appendix "Multimodal Posteriors and Inter-modal Mass" (CPU only), for the
# three separation directions (first = the published table).
# Usage (from the repository root): sbatch scripts/intermodal_mass.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
for AXIS in first last random; do
    apptainer exec --bind /nobackup /nobackup/proj/disk/naiss2025-22-438/apps/container.sif \
        python3 -m appendix.intermodal_mass --axis "${AXIS}"
done
