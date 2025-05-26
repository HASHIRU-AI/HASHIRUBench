from gradio_client import Client
from datasets import load_dataset
import json
import time
import random
import os
from datetime import datetime
import re
from time import sleep

def get_last_assistant_content(resp):
    """
    Return the last assistant utterance from the response object
    produced by `client.predict`.
    """
    # ❶ If the server wraps things in a (messages, meta) tuple
    if isinstance(resp, tuple):
        resp = resp[0]

    # ❷ At this point `resp` must be the list of message dicts
    if not isinstance(resp, list):
        return ""

    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue

        # a) plain messages
        if turn.get("content"):
            return turn["content"]

        # b) tool / function_response wrapper
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return out

        # c) messages stored as Part objects inside `content`
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]

    return ""

def benchmark_gsm8k(num_samples=1, offset=0):
    """
    Benchmark agent performance on GSM8K dataset
    Args:
        num_samples: Number of samples to test
        offset: Number of samples to skip before starting the benchmark
    """
    # Load GSM8K dataset
    random.seed(42)
    print("Loading GSM8K dataset...")
    dataset = load_dataset("ChilleD/SVAMP")
    
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=[
            "ENABLE_AGENT_CREATION",
            "ENABLE_LOCAL_AGENTS",
            "ENABLE_CLOUD_AGENTS",
            "ENABLE_TOOL_CREATION",
            "ENABLE_TOOL_INVOCATION",
            "ENABLE_RESOURCE_BUDGET",
            "ENABLE_ECONOMY_BUDGET",
        ],
		api_name="/update_model"
    )

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("results", f"svamp_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")

    # Select samples
    all_samples = []
    for split in ['train', 'test']:  # Using validation and test splits
        if split in dataset:
            all_samples.extend(dataset[split])

    all_samples = all_samples[offset:]
    if num_samples:
        all_samples = random.sample(all_samples, num_samples)

    results = []
    for i, sample in enumerate(all_samples):
        start = time.time()
        question = sample['Body']  + sample['Question']
        answer = sample['Answer'] 
        answer_only = sample['Answer'].strip()
        
        prompt = f"You are a student agent taking a math test. " \
                f"Feel free to use tools or agents to help you answer the question. " \
                f"You can decide what tools or agents to use. " \
                f"Your goal is to answer the question correctly. " \
                f"You have been asked the following question: {question} " \
                f"You have to provide the answer in the format: <ANSWER> " \
                f"Do not include any explanation or reasoning in your response. " \
                f"Reply with the answer only, without any additional text. " \
                f"The response format should be: <ANSWER>."

        while True:
            job = client.submit(
                message={"text": prompt.strip(), "files": []},
                api_name="/chat",
            )
            while not job.done():
                time.sleep(0.1)
            response, _history = job.outputs()[-1]
            agent_resp = get_last_assistant_content(_history)
            if agent_resp:
                break
            else:
                print("invalid response retrying")
                time.sleep(10)
                continue

        elapsed = time.time() - start
        time.sleep(30)  # to avoid rate limiting

        result = {
            "question_num": i,
            "question": question,
            "answer": answer,
            "answer_only": answer_only,
            "history": _history,
            "is_correct": answer_only in agent_resp,
            "agent_resp": agent_resp,
            "time_elapsed": elapsed
        }
        
        results.append(result)
        print(f"Processed sample {i+1}/{len(all_samples)}: {result['is_correct']}")
    # Save results to JSONL file
    with open(out_path, "w") as f:
        for res in results:
            f.write(json.dumps(res, indent=2) + "\n")
    print(f"Done. Results written to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GSM8K benchmark.")
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=100,
        help="Number of samples to test (default: 100)."
    )
    parser.add_argument(
        "--offset", "-o",
        type=int,
        default=0,
        help="Zero-based row index to start from (default: 0)."
    )
    args = parser.parse_args()

    benchmark_gsm8k(
        num_samples=args.num_samples,
        offset=args.offset
    )