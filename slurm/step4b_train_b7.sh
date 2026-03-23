#!/bin/bash
#SBATCH --job-name=clignet_b7_longformer
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_jfoulds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

# Reduce GPU memory fragmentation (PyTorch allocator hint)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-----------------------------
# Logging Info
#-----------------------------
echo "Job started on: $(hostname) at $(date)"
echo "Python version: $(python --version)"
echo "Environment: $CONDA_DEFAULT_ENV"
echo "Working dir: $(pwd)"
echo "GPU info:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

#-----------------------------
# Step 4b: Train B7 (Longformer) — memory-reduced settings
#   max_length 2048 (vs 4096 default) — still 4x longer than BERT
#   batch_size 2  + grad_accum 8  => effective batch = 16 (same as others)
#-----------------------------
python -u scripts/train_clignet.py \
    --processed data/processed/ \
    --out results/clignet/ \
    --mode baseline \
    --baseline-id B7 \
    --max-length 2048 \
    --batch-size 2 \
    --grad-accum 8 \
    --max-epochs 30 \
    --patience 5

echo "Job completed at: $(date)"
