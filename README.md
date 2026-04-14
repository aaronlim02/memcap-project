# Meme Metaphor Understanding Pipeline

A pipeline for metaphor detection and meme caption generation using PaliGemma2 and Llama 3.2.

## Project Structure

```
meme-metaphor-project/
├── dataset_prep/           # Download and prepare the MemeCap dataset
├── step1_paligemma/
│   ├── ketchup/            # Fine-tune PaliGemma2 on KETCHUP for metaphor detection
│   └── meme/               # Fine-tune PaliGemma2 on MemeCap + run inference
├── step2_llama_finetune/   # Fine-tune Llama 3.2 on enriched meme captions
└── step3_eval/             # Evaluate with BERTScore + LLM-as-Judge
```

---

## Requirements

- NUS SoC Compute Cluster (A100 GPU, `gpu-long` partition)
- HuggingFace account with access to:
  - `google/paligemma2-3b-pt-224`
  - `meta-llama/Llama-3.2-3B-Instruct`
- Python 3.12, virtualenv

---

## Datasets

The `dataset/` folder contains:
- `memes-test.json` — original MemeCap test set (559 memes)
- `memes-trainval.json` — original MemeCap trainval set (5,823 memes)

The `step1_paligemma/dataset/` folder contains:
- `ketchup_enriched_test.json` — test set enriched with PaliGemma2 metaphor annotations
- `ketchup_enriched_trainval.json` — trainval set enriched with PaliGemma2 metaphor annotations

The enriched JSONs are the output of Step 1 and are used as input for Step 2 and Step 3.

The `dataset/ketchup/` folder contains:
- `EN_all.csv` — English advertisement image-text pairs with metaphor labels
- `CN_all.csv` — Chinese advertisement image-text pairs with metaphor labels

Ketchup advertisement images are not included. Clone the MultiMM repository to get them:

```bash
git clone https://github.com/DUTIR-YSQ/MultiMM.git
cp -r MultiMM/imgs_EN dataset/ketchup/imgs_EN
cp -r MultiMM/imgs_CN dataset/ketchup/imgs_CN
rm -rf MultiMM
```

---

## Dataset Preparation

The MemeCap dataset JSON files (`memes-trainval.json`, `memes-test.json`) are required. Download the images using:

```bash
cd dataset_prep/
python download_images.py
```

This will download meme images into:
- `memecap/dataset/trainval_image/`
- `memecap/dataset/test_image/`

The JSON files are already included in `dataset/`. Download the meme images using:

```bash
cd dataset_prep/

# Download trainval images
python download_images.py \
    --data   ../dataset/memes-trainval.json \
    --output ../dataset/trainval_image

# Download test images
python download_images.py \
    --data   ../dataset/memes-test.json \
    --output ../dataset/test_image
```

Images will be saved to `dataset/trainval_image/` and `dataset/test_image/`.

---

## Step 1 — PaliGemma2 Fine-tuning

### 1a. Ketchup — Metaphor Detection (to be completed by teammate)

Fine-tunes PaliGemma2 on the KETCHUP/MultiMM advertisement dataset for multi-task metaphor detection (relationship, sentiment, target, source).

```bash
cd step1_paligemma/ketchup/
sbatch train_paligemma.sh
```

**Output:** `~/paligemma_output/paligemma2_multitask_lora/`

**Results:** Metaphor F1=0.77, Sentiment F1=0.46, Target exact=30.1%, Source exact=17.5%

### 1b. Meme — Metaphor Extraction

Fine-tunes PaliGemma2 on MemeCap to generate METAPHOR/MEANING pairs. Then runs the ketchup model on MemeCap to enrich memes with Target/Source metaphor annotations.

**Train meme model:**
```bash
cd step1_paligemma/meme/
sbatch train_meme.sh
```

**Output:** `~/meme_output/paligemma2_meme_lora/`

**Run ketchup inference on MemeCap (generates enriched JSONs):**
```bash
sbatch infer_ketchup.sh
```

**Output:**
- `~/meme_output/ketchup_enriched_trainval.json`
- `~/meme_output/ketchup_enriched_test.json`

---

## Step 1.5 — Qwen2.5 Fine-tuning (Kaggle)

This step runs on Kaggle and fine-tunes `Qwen/Qwen2.5-VL-7B-Instruct` on the KETCHUP/MultiMM advertisement dataset for metaphor detection and extraction. It then uses the fine-tuned model to enrich MemeCap with `Vehicle` / `Target` metaphor annotations.

### Inputs
- `multimm_path`: `/kaggle/input/datasets/{username}/multimm/EN.xlsx`
- `multimm_images_path`: `/kaggle/input/datasets/{username}/multimm-images/imgs_EN/`
- `memecap_path`: `/kaggle/input/datasets/{username}/memecap-images/trainval_image/`
- `trainval_file`: original `dataset/memes-trainval.json`

### Outputs
- Fine-tuned LoRA model folder: `./qwen2.5-vl-metaphor-lora-save`
- Inference mapping: `updated_memecap.json`
- Final enriched MemeCap dataset: `updated_memes-trainval.json`

### Workflow
1. Train the model
   - Run the notebook section through model training and save the trained LoRA model.
   - The model checkpoint is saved at `./qwen2.5-vl-metaphor-lora-save`.

2. Final Evaluation
  - Load the saved model and processor from `./qwen2.5-vl-metaphor-lora-save`.
  - Run the final evaluation section in the notebook:
    - load the trained LoRA model
    - run inference on the held-out test split
    - compute SBERT-based metaphor similarity using `compute_single_score(...)`

  - Outputs:
    - average SBERT metaphor score for the fine-tuned model
    - per-sample predictions vs ground truth

  - Baseline comparison
    - To measure the baseline, run the same evaluation code using the base Qwen2.5-VL model without LoRA fine-tuning.
    - Keep the prompt, dataset, and SBERT scoring the same.
    - This gives a direct comparison between:
      - baseline score (untrained base model)
      - trained score (fine-tuned Qwen2.5-VL + LoRA)
> Note: the baseline helps verify that the Qwen2.5 fine-tuning is actually improving metaphor extraction, not just matching dataset formatting.

3. Inference
   - Run the “Using the finetuned Qwen2.5-VL model for inference” section.
   - This writes `updated_memecap.json` with predicted metaphors for MemeCap images.

4. Update MemeCap dataset
   - Run the helper function to convert `updated_memecap.json` + `memes-trainval.json` into:
     - `updated_memes-trainval.json`

### Notes
- This step is Kaggle-specific, while the rest of the pipeline runs on the cluster.
- The output `updated_memes-trainval.json` is the enriched dataset used by Step 2 and Step 3.
- The Qwen model expects output in the format: `Vehicle: <vehicle> | Target: <target>`

## Step 2 — Llama 3.2 Fine-tuning

Fine-tunes `meta-llama/Llama-3.2-3B-Instruct` on the ketchup-enriched MemeCap dataset to generate meme captions. The prompt includes title, image description, OCR text, and metaphors extracted in Step 1 or Step 1.5.

**Setup (first time only — handled automatically by job.sh):**
```bash
cd step2_llama_finetune/
bash setup.sh
```

**Train:**
```bash
sbatch job.sh
```

**Output:** `~/4248-groupProj/llama3_meme_model_final/`

---

## Step 3 — Evaluation

### Qwen Judge on Llama Outputs (`job_judge_qwen_on_llama.sh`)
Evaluates Llama-generated captions using Qwen2.5-14B-Instruct as judge with multi-run majority voting to reduce position bias. Runs 3 shuffled evaluations per meme and takes the majority vote.

```bash
cd step3_eval/
sbatch job_judge_qwen_on_llama.sh
```

**Input:** `outputs/judge_inputs/llama_generated_candidates_randomized.csv` (Llama-generated candidates)
**Output:** `outputs/judge_results/qwen_judges_llama_random.csv` with columns: `judge_best_option`, `judge_best_source`, `judge_explanation`, `judge_vote_counts`

---

## Cluster Setup Notes

- Login: `<username>@xlogin.comp.nus.edu.sg`
- GPU: A100-40GB (`#SBATCH --gpus=a100-40`)
- Partition: `gpu-long`
- Monitor jobs: `squeue -u <username>`
- View logs: `tail -f ~/logs/slurm-<jobid>.out`
