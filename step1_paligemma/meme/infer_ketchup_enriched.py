"""
Meme Metaphor Inference — Forced Extraction Mode
Takes the OCR-enriched JSON (output of extract_ocr.py) as input.
Forces model to find multiple metaphors per meme — all memes are metaphorical.

Run extract_ocr.py first, then this script.

Usage:
    python infer_ketchup_enriched.py \
        --json ~/meme_output/memes-test-ocr.json \
        --img_dir ~/data/paligemma-training/memecap/dataset/test_image \
        --output ~/meme_output/ketchup_enriched_test.json
"""

import os
import json
import argparse
import torch
from PIL import Image, ImageFile
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

HOME         = os.path.expanduser("~")
MODEL_ID     = "google/paligemma2-3b-pt-224"
KETCHUP_LORA = f"{HOME}/paligemma_output/paligemma2_multitask_lora"
HF_TOKEN     = os.environ.get("HF_TOKEN", None)


# ===========================================================================
# PROMPT — forces multiple metaphors, no literal option
# ===========================================================================
def build_metaphor_prompt(title, img_captions, meme_captions):
    img_desc  = " | ".join(str(c) for c in img_captions  if c) if img_captions  else "none"
    meme_desc = " | ".join(str(c) for c in meme_captions if c) if meme_captions else "none"

    return (
        f"<image>\nAnalyze this meme.\n"
        f"Title: {title}\n"
        f"Image description: {img_desc}\n"
        f"Meme meaning: {meme_desc}\n\n"
        f"Answer in this exact format:\n"
        f"Relationship: metaphorical\n"
        f"Sentiment: negative, neutral, or positive\n"
        f"Target: target concept or none\n"
        f"Source: source concept or none"
    )


# ===========================================================================
# PARSER
# ===========================================================================
def parse_metaphors(text):
    target  = None
    source  = None
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("target:"):
            val = line.split(":", 1)[1].strip()
            if val.lower() not in ("none", ""):
                target = val
        elif line.lower().startswith("source:"):
            val = line.split(":", 1)[1].strip()
            if val.lower() not in ("none", ""):
                source = val
    if target:
        return [{"metaphor": target, "meaning": source or "unknown"}]
    return []


# ===========================================================================
# LOAD MODEL
# ===========================================================================
def load_model():
    print(f"Loading processor from {KETCHUP_LORA}...")
    processor = PaliGemmaProcessor.from_pretrained(KETCHUP_LORA, token=HF_TOKEN)

    print(f"Loading base model {MODEL_ID}...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from {KETCHUP_LORA}...")
    model = PeftModel.from_pretrained(base_model, KETCHUP_LORA)
    model.eval()

    print(f"Model ready | Device: {next(model.parameters()).device}")
    return model, processor


# ===========================================================================
# PREDICT
# ===========================================================================
def predict(model, processor, image_path, title, img_captions, meme_captions):
    device = next(model.parameters()).device
    image  = Image.open(image_path).convert("RGB")
    prefix = build_metaphor_prompt(title, img_captions, meme_captions)

    inputs = processor(text=prefix, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=150)

    generated = processor.decode(output_ids[0], skip_special_tokens=True)
    prompt_tail = "Source: source concept or none"
    if prompt_tail in generated:
        generated = generated.split(prompt_tail, 1)[-1].strip()

    metaphors = parse_metaphors(generated)

    # Fallback if nothing found
    if not metaphors:
        metaphors = [{"metaphor": title, "meaning": "unknown"}]

    return metaphors, generated


# ===========================================================================
# DATASET INFERENCE
# ===========================================================================
def run_on_dataset(model, processor, json_path, img_dir, output_path):
    print(f"Loading OCR-enriched dataset from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items   = list(raw.values())
        is_dict = True
        keys    = list(raw.keys())
    else:
        items   = raw
        is_dict = False

    print(f"Total items: {len(items)}")

    results    = []
    skipped    = 0
    total_meta = 0

    for i, item in enumerate(items):
        if i % 50 == 0:
            avg = total_meta / max(len(results), 1)
            print(f"  Processing {i}/{len(items)} | Avg metaphors: {avg:.1f}")

        fname    = item.get("img_fname", "")
        img_path = os.path.join(img_dir, fname)

        if not os.path.exists(img_path):
            skipped += 1
            out_item = dict(item)
            out_item["metaphors"] = []
            results.append(out_item)
            continue

        try:
            # img_captions already enriched with OCR text from extract_ocr.py
            metaphors, raw_output = predict(
                model, processor,
                image_path    = img_path,
                title         = item.get("title", ""),
                img_captions  = item.get("img_captions", []),   # ← already has OCR
                meme_captions = item.get("meme_captions", []),
            )

            total_meta += len(metaphors)

            out_item = {
                "category":     item.get("category", ""),
                "img_captions": item.get("img_captions", []),   # ← keep enriched captions
                "meme_captions":item.get("meme_captions", []),
                "title":        item.get("title", ""),
                "url":          item.get("url", ""),
                "img_fname":    fname,
                "metaphors":    metaphors,                       # ← predicted metaphors
                "post_id":      item.get("post_id", ""),
            }
            results.append(out_item)

        except Exception as e:
            print(f"  [WARN] Failed on {fname}: {e}")
            skipped += 1
            out_item = dict(item)
            out_item["metaphors"] = []
            results.append(out_item)
            continue

    print(f"\nDone!")
    print(f"  Processed         : {len(results)}")
    print(f"  Skipped           : {skipped}")
    print(f"  Avg metaphors/meme: {total_meta / max(len(results), 1):.1f}")

    if is_dict:
        output_data = {keys[i]: results[i] for i in range(len(results))}
    else:
        output_data = results

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_path}")

    print("\n=== Sample Output ===")
    for r in results[:2]:
        print(json.dumps(r, indent=2))
        print()


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    required=True,              help="OCR-enriched dataset JSON path")
    parser.add_argument("--img_dir", required=True,              help="Image directory")
    parser.add_argument("--output",  default="predictions.json", help="Output JSON path")
    args = parser.parse_args()

    model, processor = load_model()
    run_on_dataset(model, processor, args.json, args.img_dir, args.output)
