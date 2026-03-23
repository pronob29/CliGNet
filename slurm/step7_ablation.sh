#!/bin/bash
#SBATCH --job-name=clignet_ablation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_jfoulds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
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
nvidia-smi

#-----------------------------
# Step 7: Ablation studies A1–A5 (compare variants vs B8)
# A1 = no label graph (GCN removed)
# A2 = no sliding window (single chunk)
# A4 = standard BCE instead of focal loss
# A5 = no frozen BERT layers (full fine-tune)
#-----------------------------
python -u scripts/ablation.py --ablation A1 A2 A4 A5 \
  --batch-size 4 --grad-accum 8

echo "Job completed at: $(date)"
