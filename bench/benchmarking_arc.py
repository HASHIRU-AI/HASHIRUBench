from gradio_client import Client
import pandas as pd
import json
import time
import os
from datetime import datetime
import re

def sanitize_response(input_str):
    # Regex to match formats like: {choice: a}, choice: "B", etc.
    pattern = r"{\"choice\":\s*\"(\w)\"}"
    
    match = re.search(pattern, input_str)
    if match:
        return match.group(1)

    return None


def benchmark_arc(df, out_dir= "ai2_arc_results", num_questions=10):
    all_questions = df.sample(n=num_questions, random_state=42)
    
    #prepare output
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"ai12_arc_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")
    
    # init client
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )
    correct_resp = 0
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
        
        while True:
            job = client.submit(
                message={"text": prompt.strip(), "files": []},
                api_name="/chat",
            )
            while not job.done():
                time.sleep(0.1)
            response, _history = job.outputs()[-1]
            agent_resp = sanitize_response(_history[-1].get("content", ""))
            if agent_resp:
                break
            else:
                prompt = "The response does not follow the format \{\"choice\":\"<OPTION>\"\}. Please try to answer the question with the correct requested format"
                print("invalid response retrying")
                time.sleep(10)
                continue
        elapsed = time.time() - start
        time.sleep(50)
        result = {
            "question_num": question_number,
            "question": question,
            # "history": _history,
            "choices": {'text': choices['text'].tolist(), 'label': choices['label'].tolist()},
            "answerKey": answerKey,
            "agent_resp": agent_resp,
            "time_elapsed": elapsed
        }
        if answerKey == agent_resp:
            correct_resp +=1
            
        with open(out_path, "a") as f:
                f.write(json.dumps(result, indent= 2) + "\n")
        
        print(f"Score: {correct_resp} / {num_questions}")
if __name__ == "__main__":

    splits = {'train': 'ARC-Challenge/train-00000-of-00001.parquet', 'test': 'ARC-Challenge/test-00000-of-00001.parquet', 'validation': 'ARC-Challenge/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/allenai/ai2_arc/" + splits["validation"])
    benchmark_arc(df=df, num_questions=30)
