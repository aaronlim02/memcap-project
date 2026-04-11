"""
PaliGemma2-3b Fine-tuning for Meme Metaphor Generation
SoC Compute Cluster version — run via sbatch train_meme.sh

Task: Given image + captions, generate metaphor mappings
Input:  image + title + img_captions + meme_captions
Output: METAPHOR: <phrase> | MEANING: <what it represents>
        (one line per metaphor)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
HOME        = os.path.expanduser("~")
WORKING_DIR = os.environ.get("WORKING_DIR", f"{HOME}/meme_output")
DATA_JSON   = os.environ.get("DATA_JSON",   f"{HOME}/data/meme/dataset.json")
IMG_DIR     = os.environ.get("IMG_DIR",     f"{HOME}/data/meme/meme_images")
MODEL_ID    = "google/paligemma2-3b-pt-224"
HF_TOKEN    = os.environ.get("HF_TOKEN", None)

os.makedirs(WORKING_DIR, exist_ok=True)

print("=" * 60)
print("PaliGemma2 Meme Metaphor Generation — SoC Cluster")
print("=" * 60)
print(f"Working dir : {WORKING_DIR}")
print(f"Data JSON   : {DATA_JSON} (exists: {os.path.exists(DATA_JSON)})")
print(f"Image dir   : {IMG_DIR} (exists: {os.path.exists(IMG_DIR)})")
print(f"HF token    : {'set' if HF_TOKEN else 'NOT SET'}")
print()

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")
print()


# ===========================================================================
# 2. DATA LOADING
# ===========================================================================
def build_prefix(title, img_captions, meme_captions):
    img_desc  = " | ".join(img_captions)  if img_captions  else "none"
    meme_desc = " | ".join(meme_captions) if meme_captions else "none"
    return (
        f"<image>\nAnalyze this meme.\n"
        f"Title: {title}\n"
        f"Image description: {img_desc}\n"
        f"Meme meaning: {meme_desc}\n\n"
        f"List all metaphors in this exact format:\n"
        f"METAPHOR: <literal phrase> | MEANING: <what it represents>\n"
        f"Repeat for each metaphor found."
    )


def build_suffix(metaphors):
    if not metaphors:
        return "METAPHOR: none | MEANING: none"
    lines = []
    for m in metaphors:
        metaphor = str(m.get("metaphor", "")).strip()
        meaning  = str(m.get("meaning",  "")).strip()
        if metaphor and meaning:
            lines.append(f"METAPHOR: {metaphor} | MEANING: {meaning}")
    return "\n".join(lines) if lines else "METAPHOR: none | MEANING: none"


def load_meme_data(json_path, img_dir):
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle both list and dict formats
    if isinstance(raw, dict):
        items = list(raw.values())
    else:
        items = raw

    print(f"Total items in JSON: {len(items)}")

    skipped_img      = 0
    skipped_metaphor = 0
    loaded           = []

    for item in items:
        try:
            fname    = item.get("img_fname", "")
            img_path = os.path.join(img_dir, fname)

            if not os.path.exists(img_path):
                skipped_img += 1
                continue

            metaphors = item.get("metaphors", [])
            if not metaphors:
                skipped_metaphor += 1
                continue

            title         = item.get("title", "")
            img_captions  = item.get("img_captions",  [])
            meme_captions = item.get("meme_captions", [])
            post_id       = item.get("post_id", "")

            loaded.append({
                "image":          img_path,
                "prefix":         build_prefix(title, img_captions, meme_captions),
                "suffix":         build_suffix(metaphors),
                "gt_metaphors":   metaphors,
                "post_id":        post_id,
                "category":       item.get("category", ""),
            })

        except Exception as e:
            print(f"  [WARN] Skipping item: {e}")
            continue

    print(f"  Loaded      : {len(loaded)}")
    print(f"  Skipped img : {skipped_img} (image not found — run download_images.py first)")
    print(f"  Skipped meta: {skipped_metaphor} (no metaphors)")

    # Category distribution
    cat_counts = {}
    for d in loaded:
        c = d["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print(f"  Categories  : {cat_counts}")

    # Metaphors per sample
    metaphor_counts = [len(d["gt_metaphors"]) for d in loaded]
    print(f"  Avg metaphors/sample: {np.mean(metaphor_counts):.1f}")

    return loaded


def filter_bad_images(data, split_name=""):
    good, bad = [], []
    for item in data:
        try:
            with Image.open(item["image"]) as img:
                img.verify()
            good.append(item)
        except Exception:
            bad.append(item["image"])
    print(f"  {split_name} — Good: {len(good)} | Bad: {len(bad)}")
    return good


# ===========================================================================
# 3. LOAD & SPLIT DATA
# ===========================================================================
print("--- Loading Data ---")
all_data = load_meme_data(DATA_JSON, IMG_DIR)

if len(all_data) == 0:
    raise ValueError("No data loaded! Check your DATA_JSON and IMG_DIR paths, "
                     "and make sure images are downloaded.")

# 70 / 20 / 10 split
train_data, temp      = train_test_split(all_data, test_size=0.3,   random_state=42)
val_data,   test_data = train_test_split(temp,     test_size=0.333, random_state=42)
print(f"\nTrain: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

print("Filtering bad images...")
train_data = filter_bad_images(train_data, "train")
val_data   = filter_bad_images(val_data,   "val")
test_data  = filter_bad_images(test_data,  "test")

# Save JSONL
TRAIN_FIELDS = ["image", "prefix", "suffix"]
for split, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
    path = f"{WORKING_DIR}/{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in split:
            f.write(json.dumps({k: e[k] for k in TRAIN_FIELDS}, ensure_ascii=False) + "\n")
    print(f"Saved {name}.jsonl → {path}")

# Preview
sample = train_data[0]
print("\n=== Example sample ===")
print("PREFIX:\n", sample["prefix"])
print("\nSUFFIX:\n", sample["suffix"])
print()


# ===========================================================================
# 4. LOAD MODEL & PROCESSOR
# ===========================================================================
print(f"--- Loading Model: {MODEL_ID} ---")
torch.cuda.empty_cache()

processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
)

print(f"Model loaded | Device: {next(model.parameters()).device}")

# Freeze vision encoder
frozen = 0
total  = 0
for name, param in model.named_parameters():
    total += param.numel()
    if "language_model" not in name:
        param.requires_grad = False
        frozen += param.numel()
print(f"Frozen: {frozen:,} | Trainable: {total - frozen:,}")

# Apply LoRA
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
))
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
print()


# ===========================================================================
# 5. DATA COLLATOR
# ===========================================================================
def collate_fn(examples):
    valid = []
    for e in examples:
        try:
            img = Image.open(e["image"]).convert("RGB")
            img.verify()
            img = Image.open(e["image"]).convert("RGB")
            valid.append((img, e))
        except Exception as ex:
            print(f"  [WARN] Skipping: {e['image']} — {ex}")

    if not valid:
        return None

    tokens = processor(
        text=[x[1]["prefix"] for x in valid],
        images=[x[0] for x in valid],
        suffix=[x[1]["suffix"] for x in valid],
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )

    device = next(model.parameters()).device
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()}


# ===========================================================================
# 6. TRAINING
# ===========================================================================
from datasets import load_dataset as hf_load_dataset

dataset = hf_load_dataset("json", data_files={
    "train": f"{WORKING_DIR}/train.jsonl",
    "val":   f"{WORKING_DIR}/val.jsonl",
})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['val'])}")

n_gpus       = torch.cuda.device_count()
train_batch  = 4
grad_accum   = max(1, 16 // (train_batch * n_gpus))
total_steps  = (len(dataset["train"]) // (train_batch * n_gpus * grad_accum)) * 3
warmup_steps = max(10, int(total_steps * 0.03))

print(f"GPUs: {n_gpus} | Batch: {train_batch} | Grad accum: {grad_accum} | "
      f"Effective batch: {train_batch * n_gpus * grad_accum}")
print(f"Total steps: {total_steps} | Warmup: {warmup_steps}")

training_args = TrainingArguments(
    output_dir=f"{WORKING_DIR}/paligemma2_meme_checkpoints",

    per_device_train_batch_size=train_batch,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=3,

    learning_rate=2e-4,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    bf16=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
    eval_accumulation_steps=16,

    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    report_to="none",
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    data_collator=collate_fn,
)

torch.cuda.empty_cache()
print("--- Training Started ---")
trainer.train()
print("--- Training Complete ---")

SAVE_PATH = f"{WORKING_DIR}/paligemma2_meme_lora"
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
print()


# ===========================================================================
# 7. EVALUATION
# ===========================================================================
def parse_meme_output(text):
    """Parse generated metaphor lines into list of dicts."""
    metaphors = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if "METAPHOR:" in line and "MEANING:" in line:
            try:
                metaphor = line.split("METAPHOR:")[1].split("|")[0].strip()
                meaning  = line.split("MEANING:")[1].strip()
                if metaphor.lower() != "none":
                    metaphors.append({"metaphor": metaphor, "meaning": meaning})
            except:
                continue
    return metaphors


def metaphor_exact_match(pred_list, gt_list):
    """Check if predicted metaphor phrases exactly match ground truth."""
    pred_phrases = {m["metaphor"].lower().strip() for m in pred_list}
    gt_phrases   = {m["metaphor"].lower().strip() for m in gt_list}
    if not gt_phrases:
        return 1.0 if not pred_phrases else 0.0
    return len(pred_phrases & gt_phrases) / len(gt_phrases)


def metaphor_token_f1(pred_list, gt_list):
    """Token-level F1 over all metaphor phrases."""
    pred_tokens = Counter(" ".join(m["metaphor"] for m in pred_list).lower().split())
    gt_tokens   = Counter(" ".join(m["metaphor"] for m in gt_list).lower().split())
    common   = pred_tokens & gt_tokens
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / sum(pred_tokens.values())
    recall    = n_common / sum(gt_tokens.values())
    return 2 * precision * recall / (precision + recall)


def run_evaluation(model, processor, samples, max_new_tokens=100):
    model.eval()
    device  = next(model.parameters()).device
    results = []

    for i, sample in enumerate(samples):
        if i % 50 == 0:
            print(f"  Evaluating {i}/{len(samples)}...")

        image  = Image.open(sample["image"]).convert("RGB")
        inputs = processor(
            text=sample["prefix"],
            images=image,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = processor.decode(output_ids[0], skip_special_tokens=True)
        if sample["prefix"] in generated:
            generated = generated[len(sample["prefix"]):].strip()

        pred_metaphors = parse_meme_output(generated)
        gt_metaphors   = sample["gt_metaphors"]

        results.append({
            "post_id":        sample["post_id"],
            "pred_metaphors": pred_metaphors,
            "gt_metaphors":   gt_metaphors,
            "raw_output":     generated,
            "exact_match":    metaphor_exact_match(pred_metaphors, gt_metaphors),
            "token_f1":       metaphor_token_f1(pred_metaphors, gt_metaphors),
        })

    return results


def compute_metrics(results):
    exact_matches = [r["exact_match"] for r in results]
    token_f1s     = [r["token_f1"]    for r in results]

    print("=" * 55)
    print("Meme Metaphor Generation Results")
    print("=" * 55)
    print(f"  Samples evaluated     : {len(results)}")
    print(f"  Avg exact match       : {np.mean(exact_matches):.3f}")
    print(f"  Avg token F1          : {np.mean(token_f1s):.3f}")
    print(f"  Perfect predictions   : {sum(1 for e in exact_matches if e == 1.0)}")
    print(f"  Zero predictions      : {sum(1 for e in exact_matches if e == 0.0)}")
    print("=" * 55)

    return {
        "avg_exact_match": float(np.mean(exact_matches)),
        "avg_token_f1":    float(np.mean(token_f1s)),
    }


print("--- Running Evaluation on Test Set ---")
results = run_evaluation(model, processor, test_data)
metrics = compute_metrics(results)

# Save results
results_out = []
for r in results:
    results_out.append({
        "post_id":        r["post_id"],
        "pred_metaphors": r["pred_metaphors"],
        "gt_metaphors":   r["gt_metaphors"],
        "raw_output":     r["raw_output"],
        "exact_match":    r["exact_match"],
        "token_f1":       r["token_f1"],
    })

with open(f"{WORKING_DIR}/test_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results_out, f, indent=2, ensure_ascii=False)
print(f"Predictions saved to {WORKING_DIR}/test_predictions.json")

with open(f"{WORKING_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {WORKING_DIR}/metrics.json")

# Show a few examples
print("\n=== Sample Predictions ===")
for r in results[:3]:
    print(f"\nPost: {r['post_id']}")
    print(f"GT  : {r['gt_metaphors']}")
    print(f"Pred: {r['pred_metaphors']}")
    print(f"F1  : {r['token_f1']:.3f}")

print("\nDone!")
