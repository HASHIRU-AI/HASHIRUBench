import pandas as pd
import json
import time
import os
from datetime import datetime
import re
from datasets import load_dataset
import google.generativeai as genai
from dotenv import load_dotenv

def sanitize_response(input_str):
    """
    Extract yes/no answer from the response
    Handles various formats like: {"answer": "yes"}, answer: "no", etc.
    """
    # Convert to lowercase for case-insensitive matching
    input_lower = input_str.lower()
    
    # Try to match structured formats first
    patterns = [
        r'{"answer":\s*"(yes|no)"}',
        r'"answer":\s*"(yes|no)"',
        r'answer:\s*"(yes|no)"',
        r'{"choice":\s*"(yes|no)"}',
        r'"choice":\s*"(yes|no)"'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_lower)
        if match:
            return match.group(1)
    
    # Fallback: look for explicit yes/no in the response
    if "yes" in input_lower and "no" not in input_lower:
        return "yes"
    elif "no" in input_lower and "yes" not in input_lower:
        return "no"
    
    return None

def load_strategyqa_data(split="train", num_samples=None):
    """
    Load StrategyQA dataset from HuggingFace (ChilleD version)
    """
    # Load the dataset - using ChilleD's version which has better structure
    dataset = load_dataset("ChilleD/StrategyQA", split=split)
    df = pd.DataFrame(dataset)
    
    if num_samples:
        df = df.sample(n=num_samples, random_state=42)
        
    return df

def benchmark_strategyqa(df, out_dir="strategyqa_baseline_results", num_questions=10):
    """
    Benchmark multiagent system on StrategyQA dataset
    """
    if df is None or len(df) == 0:
        print("No data available for benchmarking")
        return
    
    # Sample questions if we have more than requested
    if len(df) > num_questions:
        all_questions = df.sample(n=num_questions, random_state=42)
    else:
        all_questions = df
        num_questions = len(df)
    
    # Prepare output directory
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"strategyqa_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")
    
    # Initialize Gemini client
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_KEY")
        genai.configure(api_key=os.getenv("GEMINI_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        print(f"Error connecting to client: {e}")
        return
    
    correct_resp = 0
    total_processed = 0
    
    for idx, (i, row) in enumerate(all_questions.iterrows()):
        start = time.time()
        question_number = idx + 1
        question = row['question']
        
        # StrategyQA has boolean answers
        correct_answer = "yes" if row.get('answer', False) else "no"
        
        # Get reasoning steps if available
        facts = row.get('facts', []) if 'facts' in row else []
        
        # Create prompt for the multiagent system
        prompt = "You will be asked to answer strategic questions requiring multi-step thinking. " \
                "This question requires careful analysis and step-by-step reasoning. " \
                "Think through the problem logically and provide your final answer. " \
                "Feel free to use or create agents or tools that you need." \
                f"You have been asked the following question: {question} " \
                "Your answer must be either 'yes' or 'no'. " \
                "Reply with your answer in the format: {\"answer\":\"<YES_OR_NO>\"}. " \
                "The response should contain only this JSON format."
        
        max_retries = 3
        retry_count = 0
        agent_resp = None
        
        while retry_count < max_retries:
            try:
                response = model.generate_content(prompt.strip())
                agent_resp = sanitize_response(response.text)
                
                if agent_resp:
                    break
                else:
                    retry_count += 1
                    prompt = f"The previous response did not follow the required format. " \
                            f"Please answer this question: {question} " \
                            f"Your answer must be either 'yes' or 'no' in the format: " \
                            f"{{\"answer\":\"<YES_OR_NO>\"}}. Please try again."
                    print(f"Invalid response, retrying ({retry_count}/{max_retries})")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error during API call: {e}")
                retry_count += 1
                time.sleep(10)
        
        elapsed = time.time() - start
        
        # Prepare result
        result = {
            "question_num": question_number,
            "question": question,
            "correct_answer": correct_answer,
            "agent_resp": agent_resp,
            "is_correct": agent_resp == correct_answer if agent_resp else False,
            "time_elapsed": elapsed,
            "facts": facts if facts else [],
            "retry_count": retry_count
        }
        
        # Update score
        if agent_resp == correct_answer:
            correct_resp += 1
        
        total_processed += 1
        
        # Save result
        with open(out_path, "a") as f:
            f.write(json.dumps(result, indent=2) + "\n")
        
        # Print progress
        accuracy = (correct_resp / total_processed) * 100
        print(f"Question {question_number}/{num_questions} - "
              f"Score: {correct_resp}/{total_processed} ({accuracy:.1f}%) - "
              f"Time: {elapsed:.2f}s")
        
        time.sleep(30)  # Reduced from 50s for faster testing
    
    # Final summary
    final_accuracy = (correct_resp / total_processed) * 100
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Questions: {total_processed}")
    print(f"Correct Answers: {correct_resp}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Results saved to: {out_path}")

def main():
    """
    Main function to run StrategyQA benchmark
    """
    print("Loading StrategyQA dataset...")
    
    # Load the dataset (you can change split to "train" if needed)
    df = load_strategyqa_data(split="test", num_samples=100)  # Load up to 100 samples
    
    if df is not None:
        print(f"Loaded {len(df)} questions from StrategyQA")
        print("Sample question:", df.iloc[0]['question'])
        print("Sample answer:", "yes" if df.iloc[0].get('answer', False) else "no")
        
        # Run benchmark
        benchmark_strategyqa(df=df, num_questions=100)
    else:
        print("Failed to load StrategyQA dataset")

if __name__ == "__main__":
    main()