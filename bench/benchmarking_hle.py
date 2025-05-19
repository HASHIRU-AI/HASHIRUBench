from gradio_client import Client
from datasets import load_dataset
import json
import time
import random
import os
from datetime import datetime
import re

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

def benchmark_hle(num_samples=20, categories=None):
    """
    Benchmark agent performance on HLE dataset
    
    Args:
        num_samples: Number of samples to test
        categories: List of categories to include (None for all)
    """
    # Load HLE dataset
    print("Loading HLE dataset...")
    dataset = load_dataset("cais/hle")
    
    # Initialize client
    client = Client("http://127.0.0.1:7860/")
    
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
    
    # Select random samples
    if len(all_samples) > num_samples:
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
        print(f"Warning: Only found {len(samples)} samples after filtering.")
    
    print(f"Running benchmark on {len(samples)} samples...")
    
    # Run benchmarks
    results = []
    for i, sample in enumerate(samples):
        print(f"\nProcessing sample {i+1}/{len(samples)}")
        category = sample.get('category', 'Unknown')
        prompt = sample.get('question', '')
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
            time.sleep(1)
            
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
    benchmark_hle(
        num_samples=1,
        categories=None
    )
