#!/bin/bash
#SBATCH --job-name=meme-finetune
#SBATCH --gpus=a100-40
#SBATCH --partition=gpu-long
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@u.nus.edu   # ← replace with your email
#SBATCH --output=logs/slurm-%j.out

mkdir -p logs

cd ~/4248-groupProj

if [ ! -d "venv" ]; then
    bash setup.sh
fi

source venv/bin/activate

export HUGGING_FACE_HUB_TOKEN="your_hf_token_here"   # ← paste your HuggingFace token here

python train.py
