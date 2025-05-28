from gradio_client import Client
import pandas as pd
import json
import time
import os
from datetime import datetime
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
def sanitize_response(input_str):
    # Regex to match formats like: {choice: a}, choice: "B", etc.
    pattern = r"{\"choice\":\s*\"(\w)\"}"
    
    match = re.search(pattern, input_str)
    if match:
        return match.group(1)

    return None


def benchmark_arc(df, out_dir= "./HASHIRU_results/ai2_arc_results/base", num_questions=10):
    all_questions = df.sample(n=num_questions, random_state=39)
    
    #prepare output
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ai12_arc_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")
    
    # Initialize Gemini client
    try:
        load_dotenv()
        api_key = os.getenv("GEMINI_KEY")
        client = genai.Client(api_key=api_key)
        # return client
    except Exception as e:
        print(f"Error connecting to client: {e}")
        return
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
    correct_resp = 0
    total_processed = 0
    for i, row in all_questions.iterrows():
        start = time.time()
        question_number = i
        question = row.question
        choices = row.choices
        answerKey= row.answerKey
        
        prompt = "Create a student agent that is taking a multiple choice test." \
                f"You have been asked the following question: {question}" \
                f"You have the following choices to answer: {choices}" \
                f"You Have to choose one of the provided choices as the correct answer." \
                "Reply with the selected choice. The response format should be \{\"choice\":\"<OPTION>\"\}. It should not have anything else."
        max_retries = 3
        retry_count = 0
        agent_resp = None
        while retry_count < max_retries:
            try:
                response = response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        safety_settings=safety_settings,
                    ),
                )
                agent_resp = sanitize_response(response.text)
                if agent_resp:
                    break
                else:
                    prompt = "The response does not follow the format \{\"choice\":\"<OPTION>\"\}. Please try to answer the question with the correct requested format"
                    print("invalid response retrying")
                    time.sleep(10)
                    continue
            except Exception as e:
                print(f"Error during API call: {e}")
                retry_count += 1
                time.sleep(10)
        elapsed = time.time() - start
        result = {
            "question_num": question_number,
            "question": question,
            # "history": _history,
            "choices": {'text': choices['text'].tolist(), 'label': choices['label'].tolist()},
            "answerKey": answerKey,
            "agent_resp": agent_resp,
            "time_elapsed": elapsed,
            "is_correct": agent_resp == answerKey if agent_resp else False,
        }
        if answerKey == agent_resp:
            correct_resp +=1
        total_processed += 1   
        with open(out_path, "a") as f:
                f.write(json.dumps(result, indent= 2) + "\n")
        accuracy = (correct_resp / total_processed) * 100
        print(f"Question {total_processed}/{num_questions} - "
              f"Score: {correct_resp}/{total_processed} ({accuracy:.1f}%) - "
              f"Time: {elapsed:.2f}s")
        client = genai.Client(api_key=api_key)
        time.sleep(30)

if __name__ == "__main__":

    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["validation"])
    benchmark_arc(df=df, num_questions=200)
