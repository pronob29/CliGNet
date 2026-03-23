#!/bin/bash
#SBATCH --job-name=clignet_baselines_B1B2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=general
#SBATCH --account=pi_jfoulds
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-user=pbarman1@umbc.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --cluster=chip-cpu

#-----------------------------
# Environment Setup
#-----------------------------
cd /umbc/rs/pi_jfoulds/pbarman1/pronob/CLiGNet
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv

#-----------------------------
# Logging Info
#-----------------------------
echo "Job started on: $(hostname) at $(date)"
echo "Python version: $(python --version)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Working dir: $(pwd)"

#-----------------------------
# Step 3: Train classical baselines B1 (TF-IDF + LR) and B2 (TF-IDF + SVM)
# CPU only — no GPU needed
#-----------------------------
python -u scripts/train_baselines.py --models B1 B2

echo "Job completed at: $(date)"
