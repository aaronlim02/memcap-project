# eval.py
import os
import json
import torch
from tqdm import tqdm
from bert_score import score
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def generate_caption(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def format_metaphors(metaphors):
    if not metaphors:
        return "none"
    return "; ".join(f"{m['metaphor']} = {m['meaning']}" for m in metaphors)

def build_inference_prompt(row, ocr_text):
    return f"""You are an assistant that explains the intended meaning of internet memes.
Meme title: "{row['title']}"
Image description: {row['img_captions'][0] if isinstance(row['img_captions'], list) else row['img_captions']}
Text in meme: {ocr_text}
Metaphors: {format_metaphors(row.get('metaphors', []))}
Answer:"""

def build_judge_prompt(row, prediction, reference):
    return f"""Score this meme caption prediction on a scale of 1-3.

Meme title: {row['title']}
Predicted caption: {prediction}
Reference caption: {reference}

Rules:
1 = misses the point entirely
2 = partially correct
3 = correctly captures the meaning and humour

You MUST respond with ONLY a single digit: 1, 2, or 3. Nothing else."""

def load_judge_model():
    print("Loading LLM judge (Qwen2.5-7B-Instruct)...")
    judge_model_id = "Qwen/Qwen2.5-7B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    judge_tokenizer.pad_token = judge_tokenizer.eos_token
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    judge_model.eval()
    return judge_model, judge_tokenizer

def parse_judge_score(output: str) -> int:
    import re
    match = re.search(r'\b[123]\b', output.strip())
    if match:
        return int(match.group())
    return -1

def run_llm_judge(judge_model, judge_tokenizer, row, prediction, reference) -> tuple[int, str]:
    prompt = build_judge_prompt(row, prediction, reference)
    raw_output = generate_caption(judge_model, judge_tokenizer, prompt, max_new_tokens=60)
    score = parse_judge_score(raw_output)
    return score, raw_output

def evaluate():
    import easyocr
    import re

    reader = easyocr.Reader(['en'])

    def clean_ocr_text(text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- / ]+", " ", text)
        return text.strip()

    def get_ocr(image_path):
        try:
            results = reader.readtext(image_path)
            return clean_ocr_text(" ".join([r[1] for r in results]))
        except:
            return ""

    # Load test data
    with open("./ketchup_enriched_test.json", "r") as f:
        raw = json.load(f)
    test_data = raw if isinstance(raw, list) else list(raw.values())

    # --- Load fine-tuned model ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, "./llama3_meme_model_final")
    model.eval()

    # --- Generate predictions ---
    predictions = []
    references = []
    test_items = []

    print("Generating captions...")
    for item in tqdm(test_data):
        img_path = os.path.join(os.path.expanduser("~/data/paligemma-training/memecap/dataset/test_image"), item['img_fname'])
        ocr_text = get_ocr(img_path)
        prompt = build_inference_prompt(item, ocr_text)

        pred = generate_caption(model, tokenizer, prompt)
        ref = item['meme_captions'][0] if isinstance(item['meme_captions'], list) else item['meme_captions']

        predictions.append(pred)
        references.append(ref)
        test_items.append(item)

    # --- BERTScore ---
    print("\nComputing BERTScore...")
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    print(f"\nBERTScore Results:")
    print(f"  Precision: {P.mean():.4f}")
    print(f"  Recall:    {R.mean():.4f}")
    print(f"  F1:        {F1.mean():.4f}")

    # --- LLM-as-Judge ---
    # Save predictions to disk before unloading, in case judge run fails
    with open("predictions_intermediate.json", "w") as f:
        json.dump([{"pred": p, "ref": r} for p, r in zip(predictions, references)], f, indent=2)
    print("Saved intermediate predictions to predictions_intermediate.json")

    # Unload fine-tuned model from VRAM (weights stay on disk at ./llama3_meme_model_final)
    del model, base_model
    torch.cuda.empty_cache()

    judge_model, judge_tokenizer = load_judge_model()

    judge_scores = []
    judge_outputs = []

    print("\nRunning LLM judge evaluation...")
    for item, pred, ref in tqdm(zip(test_items, predictions, references)):
        js, raw = run_llm_judge(judge_model, judge_tokenizer, item, pred, ref)
        judge_scores.append(js)
        judge_outputs.append(raw)

    valid_scores = [s for s in judge_scores if s != -1]

    if valid_scores:
        print(f"  Mean score:       {sum(valid_scores) / len(valid_scores):.4f} (out of 3)")
        print(f"  Score dist:       1={valid_scores.count(1)}  2={valid_scores.count(2)}  3={valid_scores.count(3)}")
    else:
        print("  No parseable scores — check judge_outputs in predictions.json")
        print(f"  Unparseable:      {unparseable}")
    unparseable = judge_scores.count(-1)
    print(f"\nLLM Judge Results:")
    print(f"  Mean score:       {sum(valid_scores) / len(valid_scores):.4f} (out of 3)")
    print(f"  Score dist:       1={valid_scores.count(1)}  2={valid_scores.count(2)}  3={valid_scores.count(3)}")
    print(f"  Unparseable:      {unparseable}")

    # --- Save all results ---
    results = [
        {
            "pred": p,
            "ref": r,
            "bert_f1": float(f1),
            "judge_score": js,
            "judge_reasoning": jo,
        }
        for p, r, f1, js, jo in zip(predictions, references, F1.tolist(), judge_scores, judge_outputs)
    ]

    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to predictions.json")

if __name__ == "__main__":
    evaluate()
