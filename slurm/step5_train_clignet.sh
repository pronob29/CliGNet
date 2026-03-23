#!/bin/bash
#SBATCH --job-name=clignet_main
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_jfoulds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
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
# Step 5: Train CLiGNet variants (~4h each on RTX 3090)
# B6 = CLiGNet without Platt calibration
# B8 = CLiGNet full (focal BCE + calibration) — primary model
#-----------------------------
echo "=== Training B6 (CLiGNet, no calibration) ==="
python -u scripts/train_clignet.py --mode clignet --no-calibration \
  --batch-size 4 --grad-accum 8

echo "=== Training B8 (CLiGNet full — focal BCE + Platt calibration) ==="
python -u scripts/train_clignet.py --mode clignet \
  --batch-size 4 --grad-accum 8

echo "Job completed at: $(date)"
