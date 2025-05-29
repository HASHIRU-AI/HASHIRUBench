#!/usr/bin/env python3
"""
Evaluate an entire .jsonl of context+question+answerA/B/C+label
using Google Gemini + MiniLM embeddings, without sampling or offsets.

Usage:
    python3 eval_all_with_gemini.py --input all.jsonl --output results.jsonl

Requires:
    pip install google-generativeai python-dotenv sentence-transformers tqdm
    export GEMINI_KEY=<your-api-key> or put it in a .env file
"""
import argparse
import json
import os
import re
import time
from time import sleep
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "all-MiniLM-L6-v2"
GENIE_MODEL   = "gemini-2.0-flash"
MAX_RETRIES   = 2
DELAY_SEC     = 5     # seconds between API calls
# ────────────────────────────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> str:
    """Return the substring after 'FINAL ANSWER:' (case-insensitive)."""
    m = re.search(r"final\s+answer\s*:(.*)$", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text.strip()

def call_gemini(model, prompt: str) -> str:
    """
    Send `prompt` to Gemini, retrying if no FINAL ANSWER line is found.
    Returns only the answer text.
    """
    for attempt in range(MAX_RETRIES + 1):
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        text = response.text.strip()
        if "FINAL ANSWER" in text.upper():
            return extract_final_answer(text)
        if attempt < MAX_RETRIES:
            # ask it to finish properly
            prompt = (
                "Please finish your response and output exactly one line in the format:\n"
                "FINAL ANSWER: <answer>"
            )
            sleep(1)
        else:
            return extract_final_answer(text)
    return ""  # fallback

def main(input_path, output_path):
    # Load Gemini API key
    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY not set in environment or .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GENIE_MODEL)

    # Prepare embedder
    embedder = SentenceTransformer(EMBED_MODEL)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Process every sample
    with open(input_path) as fin, open(output_path, "w") as fout:
        for raw in tqdm(fin, desc="Processing samples"):
            sample = json.loads(raw)

            # Build prompt
            prompt = (
                f"{sample['context'].rstrip()}\n\n"
                f"Q: {sample['question'].strip()}\n\n"
                "Respond with a single line: FINAL ANSWER: <your answer>."
            )

            # Call Gemini
            start = time.time()
            try:
                gem_ans = call_gemini(model, prompt)
            except Exception as e:
                gem_ans = ""
                print(f"[ERROR] id={sample.get('id','?')} → {e}")
            elapsed = time.time() - start

            # Embed options + Gemini answer
            opts = [sample.get(k, "") for k in ("answerA", "answerB", "answerC")]
            emb = embedder.encode(opts + [gem_ans], convert_to_tensor=True)
            sims = util.cos_sim(emb[3], emb[:3])[0]
            choice = int(sims.argmax().item())  # 0, 1, or 2

            # Enrich and write out
            sample.update({
                "gemini_answer": gem_ans,
                "predicted_label": str(choice + 1),
                "is_correct": (str(choice + 1) == sample.get("label", "")),
                "response_time": elapsed,
            })
            fout.write(json.dumps(sample) + "\n")

            sleep(DELAY_SEC)

    print(f"✅ All done! Results written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate entire JSONL via Gemini + MiniLM"
    )
    parser.add_argument("--input",  required=True, help="path to input .jsonl")
    parser.add_argument("--output", required=True, help="path to output .jsonl")
    args = parser.parse_args()
    main(args.input, args.output)
