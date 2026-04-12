#!/bin/bash
#SBATCH --job-name=judge_llama_qwen_rand
#SBATCH --output=outputs/logs/judge_llama_qwen_rand_%j.out
#SBATCH --error=outputs/logs/judge_llama_qwen_rand_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-96:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=09:00:00

set -euo pipefail

cd ~/4248-groupProj
mkdir -p outputs/logs outputs/judge_results

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python step3_eval/judge_llama.py \
  --input outputs/judge_inputs/llama_generated_candidates_randomized.csv \
  --output outputs/judge_results/qwen_judges_llama_random.csv \
  --model Qwen/Qwen2.5-14B-Instruct \
  --num_votes 3 \
  --base_seed 42 \
  --save_every 10 \
  --max_new_tokens 120
