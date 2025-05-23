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

def benchmark_hle(num_samples=20, categories=None, offset=0):
    """
    Benchmark agent performance on HLE dataset
    
    Args:
        num_samples: Number of samples to test
        categories: List of categories to include (None for all)
        offset: Number of samples to skip before starting the benchmark
    """
    # Load HLE dataset
    print("Loading HLE dataset...")
    dataset = load_dataset("cais/hle")
    
    # Initialize client
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/hle_benchmark_{timestamp}.jsonl"
    
    # Select samples
    all_samples = []
    for split in ['validation', 'test']:  # Using validation and test splits
        if split in dataset:
            all_samples.extend(dataset[split])
    
    # Filter by category if specified
    if categories:
        all_samples = [s for s in all_samples if s.get('category') in categories]
    
    # Filter out prompts mentioning images (text-substring only)
    filtered_samples = [s for s in all_samples if 'image' not in s.get('input', '').lower()]
    removed = len(all_samples) - len(filtered_samples)
    if removed > 0:
        print(f"Filtered out {removed} samples containing 'image'.")
    all_samples = filtered_samples
    
    # Apply offset before sampling
    if offset >= len(all_samples):
        print(f"Offset {offset} exceeds dataset size {len(all_samples)}. Nothing to benchmark.")
        return []

    all_samples = all_samples[offset:]

    # Select random samples
    if len(all_samples) > num_samples:
        random.seed(42)
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
        print(f"Warning: Only found {len(samples)} samples after filtering and offset.")
    
    print(f"Running benchmark on {len(samples)} samples...")
    
    # Run benchmarks
    results = []
    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}")
        category = sample.get('category', 'Unknown')
        prompt = sample.get('question', '') + "\n" + "Solve the above question. You MUST not ask the user for any clarifications. You MUST use tools/agents to help you. Deep research and answer the question always. Give your answer in the form FINAL ANSWER: <answer>.\n"
        print(f"Category: {category}")
        print(f"Question: {prompt[:100]}...")
        
        # Send query to agent
        try:
            start_time = time.time()
            response, history = client.predict(
                message={"text": prompt, "files": []},
                api_name="/chat"
            )
            end_time = time.time()

            target_answer_phrase = sample.get('answer', '').strip()

            agent_final_response_content = get_last_assistant_content(history)

            # Check if the response is empty
            while "FINAL ANSWER" not in agent_final_response_content.upper():
                sleep(5)
                print("…no final verdict yet, asking the agent to continue")
                resp, history = client.predict(
                    # send just “continue” (or “please continue”)
                    {"text": "Please finish the review and give the FINAL ANSWER line.", "files": []},
                    history,            # include the full chat history
                    api_name="/chat"
                )
                agent_final_response_content = get_last_assistant_content(history)
                print(agent_final_response_content)
                sleep(5)

            is_correct = False

            # Only attempt the check if both the target phrase and the agent content are non-empty
            if target_answer_phrase and agent_final_response_content:
                # Perform the simple case-insensitive substring check
                if target_answer_phrase.lower() in agent_final_response_content.lower():
                    is_correct = True
            
            # Record result
            result = {
                "sample_id": sample.get('id', f'sample_{i}'),
                "category": category,
                "input": prompt,
                "target_output": sample.get('answer', ''),
                "agent_full_response": history,
                "agent_final_response": agent_final_response_content,
                "response_time": end_time - start_time,
                "is_correct": is_correct
            }
            
            results.append(result)
            
            # Write to file immediately to preserve progress
            with open(results_file, 'a') as f:
                f.write(json.dumps(result) + '\n')
            
            print(f"Response received in {end_time - start_time:.2f} seconds")
            print(f"Response: {response[:100]}...")
            
            # Add a delay to avoid overwhelming the server
            time.sleep(5)
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    # Print summary statistics
    print("\n===== HLE BENCHMARK SUMMARY =====")
    print(f"Samples processed: {len(results)}")
    
    # Categorize by categories
    by_category = {}
    for result in results:
        category = result.get('category', 'Unknown')
        by_category.setdefault(category, []).append(result)
    
    print("\nSamples by category:")
    for category, items in by_category.items():
        print(f"  {category}: {len(items)} samples")
    
    avg_time = sum(r.get('response_time', 0) for r in results) / len(results) if results else 0
    print(f"\nAverage response time: {avg_time:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark HLE dataset")
    parser.add_argument("--offset", type=int, default=0, help="Offset for dataset samples")
    args = parser.parse_args()

    benchmark_hle(
        num_samples=1,
        categories=None,
        offset=args.offset,
    )
