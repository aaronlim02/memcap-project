"""
PaliGemma2-3b Multi-task Fine-tuning on MultiMM (KETCHUP)
SoC Compute Cluster version — run via sbatch train_paligemma.sh

Tasks:
  1. Metaphor Detection   : metaphorical / literal
  2. Sentiment            : negative / neutral / positive
  3. Target Extraction    : target concept word (or none if literal)
  4. Source Extraction    : source concept word (or none if literal)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import load_dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# Memory fragmentation fix
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ===========================================================================
# 1. CONFIGURATION
# ===========================================================================
HOME        = os.path.expanduser("~")
WORKING_DIR = os.environ.get("WORKING_DIR", f"{HOME}/paligemma_output")
DATA_DIR    = os.environ.get("DATA_DIR",    f"{HOME}/data/paligemma-training/data/ketchup")
MODEL_ID    = "google/paligemma2-3b-pt-224"
HF_TOKEN    = os.environ.get("HF_TOKEN", None)

EN_CSV  = f"{DATA_DIR}/EN_all.csv"
CN_CSV  = f"{DATA_DIR}/CN_all.csv"
EN_IMGS = f"{DATA_DIR}/imgs_EN"
CN_IMGS = f"{DATA_DIR}/imgs_CN"

os.makedirs(WORKING_DIR, exist_ok=True)

print("=" * 60)
print("PaliGemma2 Multi-task Fine-tuning — SoC Cluster")
print("=" * 60)
print(f"Working dir : {WORKING_DIR}")
print(f"Data dir    : {DATA_DIR}")
print(f"Model       : {MODEL_ID}")
print(f"HF token    : {'set' if HF_TOKEN else 'NOT SET'}")
print(f"EN CSV      : {EN_CSV} (exists: {os.path.exists(EN_CSV)})")
print(f"CN CSV      : {CN_CSV} (exists: {os.path.exists(CN_CSV)})")
print()

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available  : {n_gpus}")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({vram:.1f} GB)")
print()


# ===========================================================================
# 2. DATA LOADING
# ===========================================================================
COLS = {
    "en": {
        "metaphor":  "Unnamed: 2",
        "target":    "Target",
        "source":    "Source",
        "sentiment": "SentimentCategory",
    },
    "zh": {
        "metaphor":  "MetaphorOccurrence",
        "target":    "Target",
        "source":    "Source",
        "sentiment": "SentimentCategory",
    },
}

# SentimentCategory: 0=neutral, -1=negative, 1=positive
SENTIMENT_MAP = {
    "0":    "neutral",
    "-1":   "negative",
    "1":    "positive",
    "0.0":  "neutral",
    "-1.0": "negative",
    "1.0":  "positive",
}


def build_prefix(text):
    return (
        f"<image>\nAnalyze this advertisement.\n"
        f"Text: {text}\n\n"
        f"Answer in this exact format:\n"
        f"Relationship: metaphorical or literal\n"
        f"Sentiment: negative, neutral, or positive\n"
        f"Target: target concept or none\n"
        f"Source: source concept or none"
    )


def build_suffix(occurrence, sentiment, target, source):
    return (
        f"Relationship: {occurrence}\n"
        f"Sentiment: {sentiment}\n"
        f"Target: {target}\n"
        f"Source: {source}"
    )


def parse_sentiment(raw):
    if pd.isna(raw):
        return "neutral"
    s = str(raw).strip().lower()
    if s in ("negative", "neutral", "positive"):
        return s
    mapped = SENTIMENT_MAP.get(s)
    if mapped:
        return mapped
    print(f"  [WARN] Unknown sentiment value: '{raw}' — defaulting to neutral")
    return "neutral"


def load_data(csv_path, img_dir, language="en"):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    cols = COLS[language]

    skipped_img   = 0
    skipped_label = 0
    items         = []

    for _, row in df.iterrows():
        try:
            img_id = str(row["Pic_id"])
            if not img_id.endswith(".jpg"):
                img_id += ".jpg"
            img_path = os.path.join(img_dir, img_id)
            if not os.path.exists(img_path):
                skipped_img += 1
                continue

            label      = int(row[cols["metaphor"]])
            occurrence = "metaphorical" if label == 1 else "literal"
            sentiment  = parse_sentiment(row[cols["sentiment"]])

            if label == 1:
                raw_target = row[cols["target"]]
                raw_source = row[cols["source"]]
                target = str(raw_target).strip() if pd.notna(raw_target) else "none"
                source = str(raw_source).strip() if pd.notna(raw_source) else "none"
                if target.lower() in ("", "nan", "none"): target = "none"
                if source.lower() in ("", "nan", "none"): source = "none"
            else:
                target = "none"
                source = "none"

            text = str(row["Text"]).replace("\n", " ").strip()

            items.append({
                "image":           img_path,
                "prefix":          build_prefix(text),
                "suffix":          build_suffix(occurrence, sentiment, target, source),
                "gt_relationship": occurrence,
                "gt_sentiment":    sentiment,
                "gt_target":       target,
                "gt_source":       source,
                "language":        language,
            })

        except (ValueError, TypeError):
            skipped_label += 1
            continue

    print(f"  Loaded      : {len(items)}")
    print(f"  Skipped img : {skipped_img}")
    print(f"  Skipped lbl : {skipped_label}")

    metaphor_n = sum(1 for d in items if d["gt_relationship"] == "metaphorical")
    literal_n  = sum(1 for d in items if d["gt_relationship"] == "literal")
    sent_counts = {}
    for d in items:
        s = d["gt_sentiment"]
        sent_counts[s] = sent_counts.get(s, 0) + 1
    print(f"  Metaphorical: {metaphor_n} | Literal: {literal_n}")
    print(f"  Sentiment   : {sent_counts}")

    return items


def filter_bad_images(data, split_name=""):
    good, bad = [], []
    for item in data:
        try:
            with Image.open(item["image"]) as img:
                img.verify()
            good.append(item)
        except Exception:
            bad.append(item["image"])
    print(f"  {split_name} — Good: {len(good)} | Bad/truncated: {len(bad)}")
    if bad:
        for f in bad:
            print(f"    {f}")
    return good


# ===========================================================================
# 3. LOAD & SPLIT DATA
# ===========================================================================
print("--- Loading Data ---")
en_data  = load_data(EN_CSV, EN_IMGS, "en")
cn_data  = load_data(CN_CSV, CN_IMGS, "zh")
all_data = en_data + cn_data

print(f"\nTotal: {len(all_data)} | EN: {len(en_data)} | CN: {len(cn_data)}")

metaphor_count = sum(1 for d in all_data if d["gt_relationship"] == "metaphorical")
literal_count  = sum(1 for d in all_data if d["gt_relationship"] == "literal")
print(f"Metaphorical: {metaphor_count} ({metaphor_count/len(all_data)*100:.1f}%)")
print(f"Literal     : {literal_count}  ({literal_count/len(all_data)*100:.1f}%)")
print(f"Sentiment   : {dict(Counter(d['gt_sentiment'] for d in all_data))}")
print()

# 70 / 20 / 10 split
train_data, temp      = train_test_split(all_data, test_size=0.3,   random_state=42)
val_data,   test_data = train_test_split(temp,     test_size=0.333, random_state=42)
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

print("Filtering bad images...")
train_data = filter_bad_images(train_data, "train")
val_data   = filter_bad_images(val_data,   "val")
test_data  = filter_bad_images(test_data,  "test")

TRAIN_FIELDS = ["image", "prefix", "suffix"]
for split, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
    path = f"{WORKING_DIR}/{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for e in split:
            f.write(json.dumps({k: e[k] for k in TRAIN_FIELDS}, ensure_ascii=False) + "\n")
    print(f"Saved {name}.jsonl → {path}")
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

print("Model loaded successfully")
print(f"Device: {next(model.parameters()).device}")
print()

# Freeze vision encoder
frozen = 0
total  = 0
for name, param in model.named_parameters():
    total += param.numel()
    if "language_model" not in name:
        param.requires_grad = False
        frozen += param.numel()
print(f"Frozen parameters  : {frozen:,}")
print(f"Remaining trainable: {total - frozen:,}")

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
print()


# ===========================================================================
# 5. DATA COLLATOR
# ===========================================================================
def collate_fn(examples):
    valid_examples = []
    for e in examples:
        try:
            img = Image.open(e["image"]).convert("RGB")
            img.verify()
            img = Image.open(e["image"]).convert("RGB")
            valid_examples.append((img, e))
        except Exception as ex:
            print(f"  [WARN] Skipping bad image: {e['image']} — {ex}")
            continue

    if not valid_examples:
        return None

    images = [x[0] for x in valid_examples]
    texts  = [x[1]["prefix"] for x in valid_examples]
    labels = [x[1]["suffix"] for x in valid_examples]

    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )

    device = next(model.parameters()).device
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tokens.items()
    }


# ===========================================================================
# 6. TRAINING
# ===========================================================================
dataset = load_dataset("json", data_files={
    "train": f"{WORKING_DIR}/train.jsonl",
    "val":   f"{WORKING_DIR}/val.jsonl",
})
print(f"Train examples: {len(dataset['train'])}")
print(f"Val examples  : {len(dataset['val'])}")
print()

n_gpus      = torch.cuda.device_count()
train_batch = 8
eval_batch  = 8
grad_accum  = max(1, 32 // (train_batch * n_gpus))
total_steps = (len(dataset["train"]) // (train_batch * n_gpus * grad_accum)) * 3
warmup_steps = max(10, int(total_steps * 0.03))

print(f"Training on      : {n_gpus} GPU(s)")
print(f"Per-device batch : {train_batch}")
print(f"Grad accumulation: {grad_accum}")
print(f"Effective batch  : {train_batch * n_gpus * grad_accum}")
print(f"Total steps      : {total_steps}")
print(f"Warmup steps     : {warmup_steps}")
print()

training_args = TrainingArguments(
    output_dir=f"{WORKING_DIR}/paligemma2_multitask",

    per_device_train_batch_size=train_batch,
    per_device_eval_batch_size=eval_batch,
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
print()

# Save LoRA adapter
SAVE_PATH = f"{WORKING_DIR}/paligemma2_multitask_lora"
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
print()


# ===========================================================================
# 7. EVALUATION
# ===========================================================================
def parse_output(text):
    result = {
        "relationship": None,
        "sentiment":    None,
        "target":       None,
        "source":       None,
    }
    for line in text.strip().split("\n"):
        line = line.strip().lower()
        if line.startswith("relationship:"):
            val = line.split(":", 1)[1].strip()
            result["relationship"] = "metaphorical" if "metaphor" in val else "literal"
        elif line.startswith("sentiment:"):
            val = line.split(":", 1)[1].strip()
            if "neg"  in val: result["sentiment"] = "negative"
            elif "pos" in val: result["sentiment"] = "positive"
            else:              result["sentiment"] = "neutral"
        elif line.startswith("target:"):
            val = line.split(":", 1)[1].strip()
            result["target"] = None if val == "none" else val
        elif line.startswith("source:"):
            val = line.split(":", 1)[1].strip()
            result["source"] = None if val == "none" else val
    return result


def token_f1(pred, gold):
    if pred is None or gold is None:
        return int(pred == gold)
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common   = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_evaluation(model, processor, samples, max_new_tokens=50):
    model.eval()
    device  = next(model.parameters()).device
    results = []

    for i, sample in enumerate(samples):
        if i % 100 == 0:
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

        parsed = parse_output(generated)
        results.append({
            **parsed,
            "gt_relationship": sample["gt_relationship"],
            "gt_sentiment":    sample["gt_sentiment"],
            "gt_target":       sample["gt_target"],
            "gt_source":       sample["gt_source"],
            "raw_output":      generated,
        })

    return results


def compute_all_metrics(results):
    gt_rel   = [r["gt_relationship"] for r in results]
    pred_rel = [r["relationship"] or "literal"  for r in results]
    gt_sent   = [r["gt_sentiment"] for r in results]
    pred_sent = [r["sentiment"]    or "neutral" for r in results]

    # Target / Source — only on metaphorical samples
    # Convert None to "none" string to avoid sklearn comparison errors
    metaphor_idx = [i for i, r in enumerate(results) if r["gt_relationship"] == "metaphorical"]
    gt_target   = [results[i]["gt_target"] or "none" for i in metaphor_idx]
    pred_target = [results[i]["target"]    or "none" for i in metaphor_idx]
    gt_source   = [results[i]["gt_source"] or "none" for i in metaphor_idx]
    pred_source = [results[i]["source"]    or "none" for i in metaphor_idx]

    target_exact = accuracy_score(gt_target, pred_target)
    source_exact = accuracy_score(gt_source, pred_source)
    target_f1    = float(np.mean([token_f1(p, g) for p, g in zip(pred_target, gt_target)]))
    source_f1    = float(np.mean([token_f1(p, g) for p, g in zip(pred_source, gt_source)]))

    print("=" * 55)
    print("TASK 1 — Metaphor Detection")
    print("-" * 55)
    print(classification_report(gt_rel, pred_rel, target_names=["literal", "metaphorical"]))

    print("=" * 55)
    print("TASK 2 — Sentiment Classification")
    print("-" * 55)
    print(classification_report(gt_sent, pred_sent, target_names=["negative", "neutral", "positive"]))

    print("=" * 55)
    print(f"TASK 3 — Target Extraction  (on {len(metaphor_idx)} metaphorical samples)")
    print("-" * 55)
    print(f"  Exact match accuracy : {target_exact:.3f}")
    print(f"  Token-level F1       : {target_f1:.3f}")

    print("=" * 55)
    print(f"TASK 4 — Source Extraction  (on {len(metaphor_idx)} metaphorical samples)")
    print("-" * 55)
    print(f"  Exact match accuracy : {source_exact:.3f}")
    print(f"  Token-level F1       : {source_f1:.3f}")
    print("=" * 55)

    return {
        "metaphor_f1":  f1_score(gt_rel, pred_rel, pos_label="metaphorical"),
        "sentiment_f1": f1_score(gt_sent, pred_sent, average="macro"),
        "target_exact": target_exact,
        "target_f1":    target_f1,
        "source_exact": source_exact,
        "source_f1":    source_f1,
    }


print("--- Running Evaluation on Test Set ---")
results = run_evaluation(model, processor, test_data)
metrics = compute_all_metrics(results)

# Save predictions
results_df = pd.DataFrame(results)
pred_path  = f"{WORKING_DIR}/test_predictions.csv"
results_df.to_csv(pred_path, index=False)
print(f"\nPredictions saved to {pred_path}")

# Save metrics
metrics_path = f"{WORKING_DIR}/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

# Error analysis
wrong_metaphor  = results_df[results_df["gt_relationship"] != results_df["relationship"]]
wrong_sentiment = results_df[results_df["gt_sentiment"]    != results_df["sentiment"]]
print(f"\nMetaphor errors   : {len(wrong_metaphor)} / {len(results_df)}")
print(f"Sentiment errors  : {len(wrong_sentiment)} / {len(results_df)}")
print("\nDone!")
