#!/bin/bash
#SBATCH --job-name=paligemma2_meme
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

nvidia-smi

source ~/paligemma_env/bin/activate

export WORKING_DIR="$HOME/meme_output"
export DATA_JSON="$HOME/data/meme/dataset.json"    # ← your JSON filename
export IMG_DIR="$HOME/data/meme/meme_images"       # ← where images are stored
export HF_TOKEN="your_hf_token_here"               # ← paste your HF token
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p $WORKING_DIR

python ~/train_meme.py

echo "======================================"
echo "End time : $(date)"
echo "======================================"
