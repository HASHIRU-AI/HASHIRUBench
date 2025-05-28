import os
import json
import time
import argparse
import random
import re
from datetime import datetime
from time import sleep
from datasets import load_dataset, get_dataset_config_names
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from gradio_client import Client

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


def get_last_assistant_content(resp):
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""
    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue
        cont = turn.get("content")
        if isinstance(cont, str):
            return cont
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""


def extract_prediction(answer):
    match = re.search(r"final answer\s*:\s*(.*)", answer, re.IGNORECASE)
    return match.group(1).strip() if match else None


def load_existing_questions(results_file):
    if not os.path.exists(results_file):
        return set()
    with open(results_file) as f:
        return set(json.loads(line)["question"] for line in f if line.strip())


def get_question_text(sample):
    return (
        sample.get("question")
        or sample.get("input")
        or sample.get("prompt")
        or sample.get("text")
        or json.dumps(sample)
    )


def build_prompt(sample):
    question = get_question_text(sample)
    context = sample.get("text", "")
    answer = str(sample.get("answer", "")).strip()

    yes_no_set = {"yes", "no"}
    is_binary = answer.lower() in yes_no_set

    if is_binary:
        if context:
            prompt = (
                "You are given a legal scenario and a question.\n"
                f"Context: \"{context}\"\n"
                f"Question: \"{question}\"\n"
                "Respond only with FINAL ANSWER: Yes or FINAL ANSWER: No."
            )
        else:
            prompt = (
                "You are given a legal question.\n"
                f"Question: \"{question}\"\n"
                "Respond only with FINAL ANSWER: Yes or FINAL ANSWER: No."
            )
    else:
        if context:
            prompt = (
                "You are given a legal scenario and a question.\n"
                f"Context: \"{context}\"\n"
                f"Question: \"{question}\"\n"
                "Respond with your answer as FINAL ANSWER: <your answer here>."
            )
        else:
            prompt = (
                "You are given a legal question.\n"
                f"Question: \"{question}\"\n"
                "Respond with your answer as FINAL ANSWER: <your answer here>."
            )
    return prompt


def run_benchmark(task_name, model_name, num_samples, offset, output_dir, results_file, server_url="http://127.0.0.1:7860"):
    print(f"\nLoading task: {task_name}")
    dataset = load_dataset("nguha/legalbench", task_name)
    data = dataset["test"]

    random.seed(42)
    indices = sorted(random.sample(range(offset, len(data)), min(num_samples, len(data))))
    data = data.select(indices)

    os.makedirs(output_dir, exist_ok=True)
    done_questions = load_existing_questions(results_file)
    print(f"Skipping {len(done_questions)} already evaluated so far.")
    print(f"Appending new results to: {results_file}")

    if model_name == "flash2.0":
        genai.configure(api_key=API_KEY)
        flash_model = genai.GenerativeModel("gemini-2.0-flash")

    correct = 0
    total = 0

    for i, sample in enumerate(data):
        question = get_question_text(sample)
        if question in done_questions:
            continue

        prompt = build_prompt(sample)
        gold = str(sample.get("answer", "")).strip()

        print(f"[{i + 1}/{len(data)}] {question[:60]}...")

        try:
            start = time.time()
            MAX_RETRIES = 3
            RETRY_DELAY = 10 if model_name == "flash2.0" else 5
            answer = ""

            if model_name == "hashiru":
                client = Client(server_url)
                client.predict(
                    modeIndexes=[
                        "ENABLE_AGENT_CREATION", "ENABLE_LOCAL_AGENTS", "ENABLE_CLOUD_AGENTS",
                        "ENABLE_TOOL_CREATION", "ENABLE_TOOL_INVOCATION",
                        "ENABLE_RESOURCE_BUDGET", "ENABLE_ECONOMY_BUDGET"
                    ],
                    api_name="/update_model"
                )
                response, history = client.predict(
                    message={"text": prompt, "files": []},
                    api_name="/chat"
                )
                answer = get_last_assistant_content(history)

                retry = 0
                while "final answer" not in answer.lower() and retry < MAX_RETRIES:
                    print(f"  ...waiting for FINAL ANSWER (retry {retry + 1}/{MAX_RETRIES})")
                    sleep(RETRY_DELAY)
                    response, history = client.predict(
                        message={"text": "Please give FINAL ANSWER only.", "files": []},
                        api_name="/chat"
                    )
                    answer = get_last_assistant_content(history)
                    retry += 1

            elif model_name == "flash2.0":
                response = flash_model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(temperature=0.2),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ]
                )
                answer = response.text.strip()
                retry = 0
                while "final answer" not in answer.lower() and retry < MAX_RETRIES:
                    print(f"  ...waiting for FINAL ANSWER (retry {retry + 1}/{MAX_RETRIES})")
                    sleep(RETRY_DELAY)
                    follow_up = flash_model.generate_content("Please give only the FINAL ANSWER.")
                    answer = follow_up.text.strip()
                    retry += 1

            pred_clean = extract_prediction(answer)
            is_correct = pred_clean == gold

            result = {
                "task": task_name,
                "question": question,
                "context": sample.get("text", ""),
                "gold_answer": gold,
                "model_response": answer,
                "normalized_prediction": pred_clean,
                "correct": is_correct,
                "response_time": time.time() - start
            }

            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"  ✔ Predicted: {pred_clean} | Gold: {gold} | {'Correct' if is_correct else 'Wrong'}")

            total += 1
            if is_correct:
                correct += 1

            sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ✖ Error: {e}")
            continue

    acc = correct / total if total else 0
    print(f"Accuracy on {task_name}: {correct}/{total} = {acc:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["hashiru", "flash2.0"])
    parser.add_argument("--output_dir", type=str, default="legalbench_results/")
    parser.add_argument("--total_samples", type=int, default=250)
    args = parser.parse_args()

    task_list = get_dataset_config_names("nguha/legalbench")
    random.seed(42)
    random.shuffle(task_list)

    results_file = os.path.join(args.output_dir, f"legalbench_all_{args.model_name}.jsonl")
    samples_per_task = max(1, args.total_samples // len(task_list))
    extra = args.total_samples % len(task_list)

    for idx, task in enumerate(task_list):
        num_samples = samples_per_task + (1 if idx < extra else 0)
        run_benchmark(
            task_name=task,
            model_name=args.model_name,
            num_samples=num_samples,
            offset=0,
            output_dir=args.output_dir,
            results_file=results_file
        )
