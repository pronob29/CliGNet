#!/bin/bash
#SBATCH --job-name=clignet_bert_baselines
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=pi_jfoulds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
# Step 4: Train BERT baselines B3 and B4 sequentially on GPU
# B3 = Bio_ClinicalBERT + Linear Head (flat classifier)
# B4 = BioBERT + Linear Head (flat classifier)
# NOTE: B7 (Longformer) runs separately in step4b_train_b7.sh
#       with reduced max_length=2048 to avoid GPU OOM on 10-12 GB GPUs.
#-----------------------------
echo "=== Training B3 (Bio_ClinicalBERT) ==="
python -u scripts/train_clignet.py \
  --mode baseline --baseline-id B3 \
  --batch-size 4 --grad-accum 8

echo "=== Training B4 (BioBERT) ==="
python -u scripts/train_clignet.py \
  --mode baseline --baseline-id B4 \
  --batch-size 4 --grad-accum 8

echo "Job completed at: $(date)"
