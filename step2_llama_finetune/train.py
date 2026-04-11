import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import easyocr
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

# --- Preprocessing Helpers from your notebook ---
reader = easyocr.Reader(['en'])

def clean_ocr_text(text):
    import re
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- / ]+", " ", text)
    return text.strip()

def get_ocr(image_path):
    try:
        results = reader.readtext(image_path)
        return clean_ocr_text(" ".join([r[1] for r in results]))
    except: return ""

def format_metaphors(metaphors):
    if not metaphors:
        return "none"
    return "; ".join(f"{m['metaphor']} = {m['meaning']}" for m in metaphors)

def build_instruction_prompt(row, ocr_text):
    # Aligned with your 'build_prompt' structure
    prompt = f"""You are an assistant that explains the intended meaning of internet memes.
Meme title: "{row['title']}"
Image description: {row['img_captions'][0] if isinstance(row['img_captions'], list) else row['img_captions']}
Text in meme: {ocr_text}
Metaphors: {format_metaphors(row.get('metaphors', []))}
Answer: {row['meme_captions'][0]}"""
    return prompt

# --- Main Training Flow ---
def train():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Aborting — resubmit the job to get a non-MIG GPU node.")
    # Load dataset
    with open("./ketchup_enriched_trainval.json", "r") as f:
        raw = json.load(f)
    data = raw if isinstance(raw, list) else list(raw.values())

    # Pre-process OCR (Recommended to do before training to save time)
    print("Running OCR on training images...")
    formatted_data = []
    for item in tqdm(data):
        img_path = os.path.join(os.path.expanduser("~/data/paligemma-training/memecap/dataset/trainval_image"), item['img_fname'])
        ocr_text = get_ocr(img_path)
        full_text = build_instruction_prompt(item, ocr_text)
        formatted_data.append({"text": full_text})

    dataset = Dataset.from_list(formatted_data)

    # Model Configuration (4-bit for efficiency as in your notebook setup)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False
    # LoRA config for Llama 3
    peft_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    training_args = SFTConfig(
        output_dir="./llama3_meme_model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        dataloader_num_workers=0,

        # 2. Move these specific fields here
        max_length=512,
        #dataset_text_field="text",  # Uncomment if your dataset has a 'text' column
    )

    # 3. Clean up the Trainer call
    trainer = SFTTrainer(
        model=model,        # Use your loaded model object, not model_id string
        train_dataset=dataset,
        args=training_args, # This now contains your max_seq_length
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("./llama3_meme_model_final")

if __name__ == "__main__":
    train()
