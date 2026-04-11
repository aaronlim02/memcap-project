#!/bin/bash
#SBATCH --job-name=judge_llama
#SBATCH --output=logs/judge_llama_%j.out
#SBATCH --error=logs/judge_llama_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail

mkdir -p logs

cd ~/4248-groupProj

source venv/bin/activate

export HF_TOKEN="your_hf_token_here"   # ← paste your HuggingFace token here
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python judge_llama.py \
  --input  ketchup_enriched_test.json \
  --output judge_results.csv \
  --model  meta-llama/Llama-3.1-8B-Instruct \
  --save_every 10 \
  --max_new_tokens 120 \
  --num_votes 3 \
  --base_seed 42
