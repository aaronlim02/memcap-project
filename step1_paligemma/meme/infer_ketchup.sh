#!/bin/bash
#SBATCH --job-name=ketchup_infer
#SBATCH --gpus=a100-40
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "======================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "======================================"

nvidia-smi

ulimit -s unlimited
source ~/paligemma_env/bin/activate

export HF_TOKEN="your_hf_token_here"   # ← paste your HuggingFace token here
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DATA_DIR="$HOME/data/paligemma-training/memecap/dataset"
SCRIPT_DIR="$HOME/data/paligemma-training/memecap"
OUT_DIR="$HOME/meme_output"

mkdir -p $OUT_DIR

echo ""
echo "======================================"
echo "Inference — Test set"
echo "======================================"
python $SCRIPT_DIR/infer_ketchup_enriched.py \
    --json    $DATA_DIR/memes-test.json \
    --img_dir $DATA_DIR/test_image \
    --output  $OUT_DIR/ketchup_enriched_test.json

echo ""
echo "======================================"
echo "Inference — Trainval set"
echo "======================================"
python $SCRIPT_DIR/infer_ketchup_enriched.py \
    --json    $DATA_DIR/memes-trainval.json \
    --img_dir $DATA_DIR/trainval_image \
    --output  $OUT_DIR/ketchup_enriched_trainval.json

echo ""
echo "======================================"
echo "ALL DONE"
echo "End time : $(date)"
echo "======================================"

echo ""
echo "Output files:"
ls -lh $OUT_DIR/*.json
