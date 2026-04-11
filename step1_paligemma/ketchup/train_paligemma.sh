#!/bin/bash
#SBATCH --job-name=paligemma2_ketchup
#SBATCH --gpus=a100-40
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "======================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "Start time  : $(date)"
echo "======================================"

# ── Activate environment ──────────────────────────────────────────────────
source ~/paligemma_env/bin/activate

# ── Verify GPU ───────────────────────────────────────────────────────────
nvidia-smi

# ── Set paths ────────────────────────────────────────────────────────────
export WORKING_DIR="$HOME/paligemma_output"
export DATA_DIR="$HOME/data/paligemma-training/data/ketchup"
export HF_TOKEN="your_hf_token_here"   # ← paste your HuggingFace token here
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p $WORKING_DIR

# ── Run training ─────────────────────────────────────────────────────────
python ~/paligemma_train.py

echo "======================================"
echo "End time : $(date)"
echo "======================================"
