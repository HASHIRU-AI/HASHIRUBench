#!/usr/bin/env python3
"""
Benchmark a local Q&A dataset (JEE‑style) with Google Gemini instead of a
local Gradio agent.

Example usage:
    python3 benchmark_custom_gemini.py \
        --data_file jee_questions.json \
        --num_samples 100 \
        --offset 0

Environment:
    * Create a .env file (or export) with GEMINI_KEY=<your‑api‑key>
"""
import argparse
import json
import os
import random
import time
from datetime import datetime
from time import sleep
import re

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_dataset_file(path: str):
    """Load .json or .jsonl file into a list of dicts."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f]
    with open(path) as f:
        return json.load(f)


def extract_final_answer(text: str) -> str:
    """Return the substring after 'FINAL ANSWER:' (case‑insensitive)."""
    m = re.search(r"final\s+answer\s*:(.*)$", text, flags=re.I | re.S)
    return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def benchmark_local(
    data_file: str,
    num_samples: int = 100,
    offset: int = 0,
    model_name: str = "gemini-2.0-flash",
    max_retries: int = 3,
):
    # ---------- load data ----------
    all_samples = load_dataset_file(data_file)
    if offset >= len(all_samples):
        print(f"Offset {offset} ≥ dataset size {len(all_samples)} – nothing to do.")
        return []
    all_samples = all_samples[offset:]

    # ---------- sample ----------
    if len(all_samples) > num_samples:
        random.seed(42)
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
        print(f"Dataset smaller than requested; using {len(samples)} samples.")

    # ---------- Gemini setup ----------
    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_KEY not found in environment or .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # ---------- output ----------
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/gemini_benchmark_{timestamp}.jsonl"

    # ---------- run ----------
    results = []
    for i, sample in enumerate(samples, 1):
        print(
            f"\n[{i}/{len(samples)}] {sample.get('description', '')} (index={sample.get('index')})"
        )

        prompt = (
            sample["question"]
            + "\n\nSolve the above question. "
            "You MUST NOT ask the user for clarifications. "
            "You MUST use tools/agents to help you. "
            "Deep-research and answer the question always. "
            "Give your answer in the form FINAL ANSWER: <answer>. "
            "ONLY give the final answer letter(s) or number(s) "
            "without any additional text or explanation."
        )
        target_answer = sample.get("gold", "").strip()

        try:
            retry = 0
            t0 = time.time()
            while True:
                response = model.generate_content(prompt, temperature=0.2)
                agent_text = response.text.strip()

                # If FINAL ANSWER not found, ask the model once more (up to max_retries)
                if "FINAL ANSWER" not in agent_text.upper() and retry < max_retries:
                    retry += 1
                    print("  …missing FINAL ANSWER, retrying")
                    prompt = (
                        "Please finish and output exactly one line in the format "
                        "FINAL ANSWER: <answer>."
                    )
                    sleep(2)
                    continue
                break

            elapsed = time.time() - t0
            final_only = extract_final_answer(agent_text)
            is_correct = bool(target_answer and target_answer.lower() in final_only.lower())

            result = {
                "sample_index": sample.get("index"),
                "description": sample.get("description"),
                "subject": sample.get("subject"),
                "input_prompt": prompt,
                "target_answer": target_answer,
                "agent_final_response": agent_text,
                "only_final_answer": final_only,
                "response_time": elapsed,
                "is_correct": is_correct,
            }
            results.append(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"  ✔ done in {elapsed:.1f}s – correct={is_correct}")
            sleep(5)  # be gentle to API
        except Exception as e:
            print(f"  ✖ error: {e}")

    # ---------- summary ----------
    correct = sum(r["is_correct"] for r in results)
    avg_time = (
        sum(r["response_time"] for r in results) / len(results) if results else 0
    )
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total samples: {len(results)}")
    print(f"Correct:       {correct}")
    if results:
        print(f"Accuracy:      {100 * correct / len(results):.2f}%")
    print(f"Avg. response: {avg_time:.2f}s")
    print(f"Saved to:      {results_file}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="Path to .json or .jsonl file")
    ap.add_argument("--num_samples", type=int, default=100, help="How many to sample")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N records")
    ap.add_argument(
        "--model_name",
        default="gemini-2.0-flash",
        help="Gemini model name (e.g., gemini-1.5-pro-latest)",
    )
    args = ap.parse_args()

    benchmark_local(
        data_file=args.data_file,
        num_samples=args.num_samples,
        offset=args.offset,
        model_name=args.model_name,
    )
