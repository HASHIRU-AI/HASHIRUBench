import os
import json
import time
import argparse
import random
import re
from datetime import datetime
from time import sleep
from datasets import load_dataset
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
        if turn.get("content"):
            return turn["content"]
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""


def extract_prediction(answer):
    match = re.search(r"final answer\s*:\s*(yes|no)", answer, re.IGNORECASE)
    return match.group(1).capitalize() if match else None


def load_existing_questions(results_file):
    if not os.path.exists(results_file):
        return set()
    with open(results_file) as f:
        return set(json.loads(line)["question"] for line in f if line.strip())


def run_benchmark(task_name, model_name, num_samples, offset, output_dir, server_url="http://127.0.0.1:7860"):
    print(f"Loading task: {task_name}")
    dataset = load_dataset("nguha/legalbench", task_name)
    data = dataset["test"]

    # Fixed sampling with reproducibility
    random.seed(42)
    indices = sorted(random.sample(range(offset, len(data)), num_samples))
    data = data.select(indices)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"{task_name}_{model_name}_{timestamp}.jsonl")

    done_questions = load_existing_questions(results_file)
    print(f"Skipping {len(done_questions)} questions already evaluated.")
    print(f"Saving new results to: {results_file}")

    if model_name == "flash2.0":
        genai.configure(api_key=API_KEY)
        flash_model = genai.GenerativeModel("gemini-2.0-flash")

    correct = 0
    for i, sample in enumerate(data):
        question = sample["question"]
        context = sample.get("text", "")
        gold = sample["answer"]

        if question in done_questions:
            continue

        print(f"\n[{i+1}/{len(data)}] {question[:60]}...")

        if context:
            prompt = (
                "You are given a legal contract clause and a question.\n"
                f"Clause: \"{context}\"\n"
                f"Question: \"{question}\"\n"
                "Does the clause imply the concept asked about?\n"
                "Respond only as FINAL ANSWER: Yes or FINAL ANSWER: No."
            )
        else:
            prompt = (
                "You are given a paraphrased legal contract clause.\n"
                f"Clause: \"{question}\"\n"
                "Does this clause imply the concept asked about?\n"
                "Respond only as FINAL ANSWER: Yes or FINAL ANSWER: No."
            )

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
                    print(f"  ...waiting for FINAL ANSWER (retry {retry+1}/{MAX_RETRIES})")
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
                    print(f"  ...waiting for FINAL ANSWER (retry {retry+1}/{MAX_RETRIES})")
                    sleep(RETRY_DELAY)
                    follow_up = flash_model.generate_content("Please give only the FINAL ANSWER.")
                    answer = follow_up.text.strip()
                    retry += 1

            pred_clean = extract_prediction(answer)
            is_correct = pred_clean == gold

            result = {
                "question": question,
                "context": context,
                "gold_answer": gold,
                "model_response": answer,
                "normalized_prediction": pred_clean,
                "correct": is_correct,
                "response_time": time.time() - start
            }

            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"  ✔ Predicted: {pred_clean} | Correct: {gold} | {'Correct' if is_correct else 'Wrong'}")

            if is_correct:
                correct += 1

            sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  ✖ Error: {e}")
            continue

    acc = correct / (i + 1) if i + 1 else 0
    print(f"\nFinal Accuracy: {correct}/{i+1} = {acc:.2%}")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="contract_qa", help="LegalBench task name")
    parser.add_argument("--model_name", type=str, required=True, choices=["hashiru", "flash2.0"])
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="legalbench_results/")
    args = parser.parse_args()

    run_benchmark(
        task_name=args.task,
        model_name=args.model_name,
        num_samples=args.num,
        offset=args.offset,
        output_dir=args.output_dir
    )
