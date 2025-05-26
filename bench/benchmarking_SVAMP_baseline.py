#!/usr/bin/env python3
"""
Benchmark the GSM8K math‑word‑problem dataset with Google Gemini.

This replaces the original Gradio‑agent loop with direct Gemini API calls.

Example usage:
    python3 benchmark_gsm8k_gemini.py \
        --num_samples 100 \
        --offset 0

Environment:
    * Put a `GEMINI_KEY=<your‑api‑key>` entry in a .env file or export it
      as an environment variable before running.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from time import sleep

import google.generativeai as genai
from datasets import load_dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_numeric_answer(ans_field: str) -> str:
    """Return the canonical numeric/string answer appearing after the #### marker."""
    parts = ans_field.split("####")
    if len(parts) > 1:
        return parts[-1].strip()
    m = re.findall(r"-?\d+\.?\d*", ans_field)
    return m[-1] if m else ans_field.strip()


def extract_final_answer(text: str) -> str:
    """Grab the answer after `FINAL ANSWER:` (case‑insensitive) or return whole line."""
    match = re.search(r"final\s+answer\s*:(.*)$", text, flags=re.I | re.S)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_gsm8k(
    num_samples: int = 100,
    offset: int = 0,
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    max_retries: int = 3,
):
    """Run the benchmark and return a list of result dictionaries."""

    # ---------- dataset ----------
    random.seed(42)
    print("Loading GSM8K dataset…")
    ds = load_dataset("ChilleD/SVAMP", split="train+test")

    if offset >= len(ds):
        print(f"Offset {offset} ≥ dataset size {len(ds)} – nothing to do.")
        return []

    indices = list(range(offset, len(ds)))
    if len(indices) > num_samples:
        indices = random.sample(indices, num_samples)
    else:
        print(f"Dataset smaller than requested; using {len(indices)} samples.")

    # ---------- Gemini setup ----------
    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY missing – set it in .env or env var")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    gen_cfg = {"temperature": temperature}

    # ---------- output ----------
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = Path("results") / f"svamp_gemini_{ts}.jsonl"

    # ---------- run ----------
    results: list[dict] = []
    for rank, idx in enumerate(indices, 1):
        sample = ds[idx]
        question = sample['Body']  + sample['Question']
        answer = sample['Answer'] 
        gold_only = sample['Answer'].strip()
        

        prompt = (
            f"{question}\n\nSolve the problem. "
            "Provide the result in the form FINAL ANSWER: <answer>. "
            "Give *only* the numeric answer (or units if present) and no explanation."
        )

        print(f"\n[{rank}/{len(indices)}] id={idx}")
        retries = 0
        t0 = time.time()
        current_prompt = prompt
        while True:
            try:
                resp = model.generate_content(current_prompt, generation_config=gen_cfg)
            except Exception as exc:
                if retries < max_retries:
                    retries += 1
                    print(f"  …API error, retrying ({retries}/{max_retries}) – {exc}")
                    sleep(2)
                    continue
                raise

            agent_raw = resp.text.strip()
            if "FINAL ANSWER" not in agent_raw.upper() and retries < max_retries:
                retries += 1
                print("  …missing FINAL ANSWER, prompting again")
                current_prompt = (
                    prompt
                    + "\n\nRemember: respond with ONE line in the exact format "
                    "FINAL ANSWER: <answer>."
                )
                sleep(1)
                continue
            break

        elapsed = time.time() - t0
        final_only = extract_final_answer(agent_raw)
        is_correct = gold_only.lower() == final_only.lower()

        result = {
            "dataset_index": idx,
            "question": question,
            "gold_answer": gold_only,
            "agent_raw": agent_raw,
            "agent_final": final_only,
            "response_time": elapsed,
            "is_correct": is_correct,
        }
        results.append(result)
        with open(outfile, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        print(f"  ✔ {('✓' if is_correct else '✗')} in {elapsed:.1f}s – expected {gold_only}, got {final_only}")
        sleep(3)

    # ---------- summary ----------
    correct = sum(r["is_correct"] for r in results)
    avg_time = sum(r["response_time"] for r in results) / len(results)
    print("\n===== GSM8K BENCHMARK SUMMARY =====")
    print(f"Samples:  {len(results)}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {100 * correct / len(results):.2f}%")
    print(f"Avg time: {avg_time:.2f}s")
    print(f"Saved:    {outfile}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100, help="How many to run")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N")
    parser.add_argument("--model_name", default="gemini-2.0-flash", help="Gemini model variant")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    args = parser.parse_args()

    benchmark_gsm8k(
        num_samples=args.num_samples,
        offset=args.offset,
        model_name=args.model_name,
        temperature=args.temperature,
    )
