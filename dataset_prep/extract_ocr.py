"""
OCR Extraction Script
Extracts text visible in meme images using PaliGemma2's built-in OCR capability.
Creates a new JSON file with OCR text appended to img_captions.

Usage:
    python extract_ocr.py \
        --json ~/data/paligemma-training/memecap/dataset/memes-test.json \
        --img_dir ~/data/paligemma-training/memecap/dataset/test_image \
        --output ~/meme_output/memes-test-ocr.json
"""

import os
import json
import argparse
import torch
from PIL import Image, ImageFile
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

HOME     = os.path.expanduser("~")
MODEL_ID = "google/paligemma2-3b-pt-224"
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ===========================================================================
# LOAD BASE MODEL ONLY — no LoRA needed for OCR
# ===========================================================================
def load_model():
    print("Loading processor...")
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

    print(f"Loading base model {MODEL_ID}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model ready | Device: {next(model.parameters()).device}")
    return model, processor


# ===========================================================================
# OCR
# ===========================================================================
def extract_text_from_image(model, processor, image_path):
    """Use PaliGemma2 to read any text visible in the meme image."""
    device = next(model.parameters()).device
    image  = Image.open(image_path).convert("RGB")
    prefix = "<image>\nRead and transcribe all text visible in this image. If there is no text, say 'no text'."

    inputs = processor(text=prefix, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=100)

    generated = processor.decode(output_ids[0], skip_special_tokens=True)
    # The prompt text (minus the <image> token) appears verbatim in the decoded output.
    # Split on the last line of the prompt to reliably strip it.
    prompt_tail = "If there is no text, say 'no text'."
    if prompt_tail in generated:
        generated = generated.split(prompt_tail, 1)[-1].strip()

    if generated.lower() in ("no text", "no text.", "", "none"):
        return None
    return generated


# ===========================================================================
# MAIN
# ===========================================================================
def run(json_path, img_dir, output_path):
    print(f"Loading dataset from {json_path}...")
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

    model, processor = load_model()

    results       = []
    skipped       = 0
    ocr_found     = 0
    ocr_not_found = 0

    for i, item in enumerate(items):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(items)} | OCR found: {ocr_found} | No text: {ocr_not_found}")

        fname    = item.get("img_fname", "")
        img_path = os.path.join(img_dir, fname)

        if not os.path.exists(img_path):
            skipped += 1
            results.append(dict(item))
            continue

        try:
            ocr_text = extract_text_from_image(model, processor, img_path)

            # Append OCR text to img_captions if found
            img_captions = list(item.get("img_captions", []))
            if ocr_text:
                img_captions.append(f"Text in image: {ocr_text}")
                ocr_found += 1
            else:
                ocr_not_found += 1

            # Same structure as original, updated img_captions
            out_item = {
                "category":     item.get("category", ""),
                "img_captions": img_captions,            # ← updated with OCR
                "meme_captions":item.get("meme_captions", []),
                "title":        item.get("title", ""),
                "url":          item.get("url", ""),
                "img_fname":    fname,
                "metaphors":    item.get("metaphors", []),  # ← original unchanged
                "post_id":      item.get("post_id", ""),
            }
            results.append(out_item)

        except Exception as e:
            print(f"  [WARN] Failed on {fname}: {e}")
            skipped += 1
            results.append(dict(item))
            continue

    print(f"\nDone!")
    print(f"  Processed  : {len(results)}")
    print(f"  Skipped    : {skipped}")
    print(f"  OCR found  : {ocr_found}")
    print(f"  No text    : {ocr_not_found}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    required=True,              help="Input dataset JSON path")
    parser.add_argument("--img_dir", required=True,              help="Image directory")
    parser.add_argument("--output",  default="dataset-ocr.json", help="Output JSON path")
    args = parser.parse_args()

    run(args.json, args.img_dir, args.output)
