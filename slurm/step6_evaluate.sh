#!/bin/bash
#SBATCH --job-name=clignet_evaluate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_jfoulds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-user=pbarman1@umbc.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --cluster=chip-gpu

#-----------------------------
# Environment Setup
#-----------------------------
cd /umbc/rs/pi_jfoulds/pbarman1/pronob/CLiGNet
module purge
module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate /umbc/rs/pi_jfoulds/users/pbarman1/conda_envs/testenv
export TORCH_HOME=/umbc/rs/pi_jfoulds/pbarman1/cache/torch
export HF_HOME=/umbc/rs/pi_jfoulds/pbarman1/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-----------------------------
# Logging Info
#-----------------------------
echo "Job started on: $(hostname) at $(date)"
echo "Python version: $(python --version)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Working dir: $(pwd)"

#-----------------------------
# Step 6: Generate results table + McNemar significance tests
# (CPU only — loads saved model outputs, no GPU needed)
#-----------------------------
python -u scripts/evaluate.py

echo "Job completed at: $(date)"
