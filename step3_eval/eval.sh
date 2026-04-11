#!/bin/bash
#SBATCH --job-name=meme-eval
#SBATCH --gpus=a100-40
#SBATCH --partition=gpu-long
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/eval-%j.out

mkdir -p logs

cd ~/4248-groupProj

ulimit -s unlimited
source venv/bin/activate

export HF_TOKEN="your_hf_token_here"   # ← paste your HuggingFace token here
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python eval2.py
