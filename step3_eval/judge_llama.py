import argparse
import os
import re
import random
import pandas as pd
import torch
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_missing(x):
    return pd.isna(x) or (isinstance(x, str) and x.strip() == "")


def get_options(row):
    """Return a dict of {original_key (1-4): caption_text}."""
    return {
        str(k): safe_text(row.get(f"option_{k}", ""))
        for k in range(1, 5)
    }


def validate_options(options: dict):
    return [f"option_{k}_empty" for k, v in options.items() if v == ""]


# ──────────────────────────────────────────────
# Prompt builder — shuffles display order each call
# ──────────────────────────────────────────────

def build_judge_prompt(row, options: dict, seed: int | None = None):
    """
    Shuffle the four options into a random display order.

    Returns
    -------
    prompt : str
    display_to_original : dict
        Maps the displayed position ("1"–"4") back to the original
        option key ("1"–"4") so we can undo the shuffle after parsing.
    """
    rng = random.Random(seed)

    original_keys = list(options.keys())           # ["1", "2", "3", "4"]
    rng.shuffle(original_keys)                     # randomise presentation order

    display_to_original = {}                       # display pos → original key
    option_lines = []
    for display_pos, orig_key in enumerate(original_keys, start=1):
        display_to_original[str(display_pos)] = orig_key
        option_lines.append(f"Option {display_pos}: {options[orig_key]}")

    title         = safe_text(row.get("title", ""))
    image_caption = safe_text(row.get("image_caption", ""))
    ocr_text      = safe_text(row.get("ocr_text", ""))
    options_block = "\n".join(option_lines)

    prompt = f"""You are judging meme explanation captions.

Your task is to decide which caption best explains the intended meaning of the meme.

Meme title: {title}
Image description: {image_caption}
Text in meme: {ocr_text}

Here are four candidate captions presented in randomized order:

{options_block}

Evaluation criteria:
- Accuracy: does the caption correctly capture what the meme is trying to say?
- Humour / pragmatics: does it capture the joke, tone, or implied meaning?
- Non-literal understanding: does it go beyond surface description when needed?
- Faithfulness: is it grounded in the meme rather than hallucinated?
- Do not prefer a caption just because it is longer or appeared first.
- Read and evaluate ALL four options independently before deciding.
- Do not favour an option simply because it appeared first or last.

Choose the single best caption.

IMPORTANT OUTPUT RULES:
- Output exactly 2 lines only.
- First line must be exactly one of:
BEST_OPTION=1
BEST_OPTION=2
BEST_OPTION=3
BEST_OPTION=4
- Second line must start with:
REASON=
- Do not output JSON.
- Do not output markdown.
- Do not output anything else.
"""
    return prompt, display_to_original


# ──────────────────────────────────────────────
# Model inference
# ──────────────────────────────────────────────

@torch.inference_mode()
def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Slice off the prompt tokens so we only decode the new completion
    new_tokens = outputs[0][prompt_length:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return decoded.strip()


# ──────────────────────────────────────────────
# Output parsing
# ──────────────────────────────────────────────

def parse_output(text):
    """
    Returns {"best_option": "1"|"2"|"3"|"4", "reason": str}
    or None if parsing fails.
    """
    if not isinstance(text, str):
        return None

    best_match   = re.search(r"BEST_OPTION\s*=\s*([1234])", text, re.IGNORECASE)
    reason_match = re.search(r"REASON\s*=\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if not best_match:
        return None

    return {
        "best_option": best_match.group(1),
        "reason": reason_match.group(1).strip() if reason_match else "",
    }


# ──────────────────────────────────────────────
# Multi-run voting (core bias mitigation)
# ──────────────────────────────────────────────

def judge_row_with_voting(model, tokenizer, row, options, num_votes=3, max_new_tokens=200, base_seed=42):
    """
    Run the judge `num_votes` times with different shuffle seeds.
    Each run remaps the displayed winner back to its *original* key.
    Returns the majority-vote original key, plus per-run details.
    """
    vote_originals = []   # original keys that won each run
    run_details    = []   # for diagnostics

    for run_idx in range(num_votes):
        seed   = base_seed + run_idx
        prompt, display_to_original = build_judge_prompt(row, options, seed=seed)
        raw    = generate_response(model, tokenizer, prompt, max_new_tokens)
        parsed = parse_output(raw)

        detail = {
            "run": run_idx,
            "seed": seed,
            "raw": raw,
            "display_to_original": display_to_original,
            "parsed": parsed,
        }

        if parsed is not None:
            original_key = display_to_original[parsed["best_option"]]
            vote_originals.append(original_key)
            detail["original_winner"] = original_key
        else:
            detail["original_winner"] = None

        run_details.append(detail)

    if not vote_originals:
        return None, run_details

    winner = Counter(vote_originals).most_common(1)[0][0]
    return winner, run_details


def map_option_to_source(row, original_key):
    return safe_text(row.get(f"option_{original_key}_source", ""))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Judge randomized meme captions with a Llama model (bias-corrected)."
    )
    parser.add_argument("--input",          required=True,  help="Randomized judge input CSV")
    parser.add_argument("--output",         required=True,  help="Judge output CSV")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model name or local path for the Llama judge",
    )
    parser.add_argument("--save_every",     type=int, default=10,  help="Save progress every N rows")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--num_votes",
        type=int,
        default=3,
        help="Number of shuffle+judge runs per row for majority voting (odd number recommended)",
    )
    parser.add_argument("--base_seed",  type=int, default=42,  help="Base RNG seed for shuffle")
    parser.add_argument("--hf_token",   default=None,          help="HF token or set HF_TOKEN env var")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # ── Load or resume ──
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        print(f"Loaded existing judge progress from {args.output}", flush=True)
    else:
        df = pd.read_csv(args.input)
        print(f"Starting fresh from {args.input}", flush=True)

    # Ensure all output columns exist
    for col, default in [
        ("judge_best_option",   ""),
        ("judge_best_source",   ""),
        ("judge_explanation",   ""),
        ("judge_vote_counts",   ""),   # e.g. "1:2,3:1" — majority detail
        ("judge_raw_responses", ""),   # all runs joined
        ("judge_parse_ok",      False),
        ("judge_error",         ""),
    ]:
        if col not in df.columns:
            df[col] = default

        # Force safe dtypes for resumed CSVs
    text_cols = [
        "judge_best_option",
        "judge_best_source",
        "judge_explanation",
        "judge_raw_responses",
        "judge_error",
    ]
    for col in text_cols:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(pd.notna(df[col]), "")

    df["judge_parse_ok"] = df["judge_parse_ok"].fillna(False).astype(bool)

    missing_indices = [
        i for i in range(len(df))
        if is_missing(df.at[i, "judge_best_option"])
    ]
    print(f"Rows needing judgment: {len(missing_indices)}", flush=True)

    if not missing_indices:
        print("All rows already judged. Nothing to do.", flush=True)
        return

    # ── Load model ──
    print(f"Loading judge model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Judge loop ──
    for count, i in enumerate(tqdm(missing_indices, desc="Judging captions")):
        row     = df.iloc[i]
        options = get_options(row)

        # Validate all four options are non-empty
        option_errors = validate_options(options)
        if option_errors:
            df.at[i, "judge_best_option"]   = "INPUT_ERROR"
            df.at[i, "judge_best_source"]   = ""
            df.at[i, "judge_explanation"]   = ""
            df.at[i, "judge_vote_counts"]   = ""
            df.at[i, "judge_raw_responses"] = ""
            df.at[i, "judge_parse_ok"]      = False
            df.at[i, "judge_error"]         = ";".join(option_errors)
            continue

        winner, run_details = judge_row_with_voting(
            model=model,
            tokenizer=tokenizer,
            row=row,
            options=options,
            num_votes=args.num_votes,
            max_new_tokens=args.max_new_tokens,
            base_seed=args.base_seed,
        )

        # Aggregate raw responses across runs
        all_raws = " ||| ".join(d["raw"] for d in run_details)

        # Vote count string e.g. "1:2,3:1"
        originals_that_won = [d["original_winner"] for d in run_details if d["original_winner"]]
        vote_counts_str = ",".join(
            f"{k}:{v}" for k, v in Counter(originals_that_won).most_common()
        )

        # Best explanation: from the first run that voted for the winner
        best_reason = ""
        for d in run_details:
            if d["original_winner"] == winner and d["parsed"]:
                best_reason = d["parsed"]["reason"]
                break

        df.at[i, "judge_raw_responses"] = all_raws
        df.at[i, "judge_vote_counts"]   = vote_counts_str
        df.at[i, "judge_error"]         = ""

        if winner is not None:
            best_source = map_option_to_source(row, winner)
            df.at[i, "judge_best_option"]  = winner
            df.at[i, "judge_best_source"]  = best_source
            df.at[i, "judge_explanation"]  = best_reason
            df.at[i, "judge_parse_ok"]     = True

            if best_source == "":
                df.at[i, "judge_error"] = "best_source_missing"
        else:
            df.at[i, "judge_best_option"]  = "PARSE_ERROR"
            df.at[i, "judge_best_source"]  = ""
            df.at[i, "judge_explanation"]  = ""
            df.at[i, "judge_parse_ok"]     = False
            df.at[i, "judge_error"]        = "all_runs_failed_to_parse"

        if (count + 1) % args.save_every == 0:
            df.to_csv(args.output, index=False)
            done = (~df["judge_best_option"].apply(is_missing)).sum()
            print(f"Saved at count {count + 1} | judged rows: {done}", flush=True)

    df.to_csv(args.output, index=False)
    print(f"Final judge output saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
