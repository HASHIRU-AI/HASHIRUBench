import json
import time
import re
from typing import List, Dict
from urllib.parse import urlparse
from gradio_client import Client
import os
import string
from datetime import datetime

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([chr(8216), chr(8217), chr(180), chr(96)]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    from collections import Counter
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def benchmark_triviaqa(gradio_url: str, question_answer_pairs: List[Dict[str, str]], output_dir="triviaqa_results"):
    """Runs a TriviaQA benchmark against a Gradio model."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"triviaqa_benchmark_{timestamp}.jsonl")
    try:
        print(f"Writing results to {out_path}")
    except UnicodeEncodeError:
        print("Could not print file path due to encoding error.")

    results = []
    correct_answers = 0
    total_questions = len(question_answer_pairs)

    try:
        client = Client(gradio_url)
    except Exception as e:
        print(f"Error: Could not connect to Gradio URL. {e}")
        return

    for i, pair in enumerate(question_answer_pairs):
        question = pair["question"]
        correct_answer = pair["answer"]

        try:
            start_time = time.time()
            response, history = client.predict(message={"text": question, "files": []}, api_name="/chat")
            print(f"Question {i + 1}/{total_questions}: {question}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Response: {response}")
            end_time = time.time()
            prediction = response["content"]

            em_score = exact_match_score(prediction, correct_answer)
            f1 = f1_score(prediction, correct_answer)

            if em_score:
                correct_answers += 1

            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "prediction": prediction,
                "exact_match": em_score,
                "f1_score": f1,
                "time_elapsed": end_time - start_time,
            })
            with open(out_path, "a") as f:
                json.dump(results[-1], f, indent=4)
                f.write("\n")


        except Exception as e:
            print(f"Error during prediction: {e}")
            results.append({
                "question": question,
                "correct_answer": correct_answer,
                "prediction": "Error",
                "exact_match": 0,
                "f1_score": 0,
                "time_elapsed": 0,
                "error": str(e),
            })
            with open(out_path, "a") as f:
                json.dump(results[-1], f, indent=4)
                f.write("\n")

    em_accuracy = 100.0 * correct_answers / total_questions if total_questions > 0 else 0.0
    try:
        print(f"Benchmark completed. Exact match accuracy: {em_accuracy:.2f}%")
    except UnicodeEncodeError:
        print("Could not print benchmark results due to encoding error.")

def load_triviaqa_data(file_path: str) -> List[Dict[str, str]]:
    """Loads TriviaQA data from a JSON file and returns a list of question-answer pairs."""
    with open(file_path, "r") as f:
        data = json.load(f)
    qa_pairs = []
    for item in data["Data"]:
        question = item["Question"]
        answer = item["Answer"]["Value"]  # Or another field from Answer, like "NormalizedValue"
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs

if __name__ == "__main__":
    data_file = "bench/data/triviaqa-unfiltered/sample.json"
    question_answer_pairs = load_triviaqa_data(data_file)
    gradio_url = "http://127.0.0.1:7860/"  # Replace with the actual Gradio URL
    benchmark_triviaqa(gradio_url, question_answer_pairs)
