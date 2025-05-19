from gradio_client import Client
from datasets import load_dataset
import requests
import json
import time
import random
import os
import re
from datetime import datetime

# Fetch the official Wordle guess list from GitHub
WORD_LIST_URL = "https://raw.githubusercontent.com/tabatkins/wordle-list/main/words"

def load_word_list():
    resp = requests.get(WORD_LIST_URL)
    resp.raise_for_status()
    words = [w.strip().lower() for w in resp.text.splitlines()]
    return [w for w in words if len(w) == 5 and w.isalpha()]

WORD_LIST = load_word_list()

def compute_feedback(guess: str, solution: str) -> str:
    """Return a 5-char string of G/Y/B feedback."""
    feedback = ["B"] * 5
    sol = list(solution)
    # Greens first
    for i, g in enumerate(guess):
        if g == sol[i]:
            feedback[i], sol[i] = "G", None
    # Yellows second
    for i, g in enumerate(guess):
        if feedback[i] == "B" and g in sol:
            feedback[i] = "Y"
            sol[sol.index(g)] = None
    return "".join(feedback)

def sanitize_guess(raw: str) -> str:
    """Extract a 5-letter guess from any model output."""
    raw = raw.lower()
    regex = r"{\"guess\":\s*\"(\w*)\"}"
    m = re.search(regex, raw)
    if m:
        return m.group(1)

def benchmark_wordle(num_games: int = 10, max_guesses: int = 6):
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join(
        "results", f"wordle_benchmark_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    results = []

    for gi in range(num_games):
        client = Client("http://127.0.0.1:7860/hashiru/")
        client.predict(
            modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
            api_name="/update_model"
        )
        solution = random.choice(WORD_LIST)
        print(f"Game {gi+1}/{num_games}, solution: {solution}")
        guesses, attempts = [], 0
        start_time = time.time()

        prompt = (
                "We are playing a game of Wordle. The solution is a 5-letter word.\n"
                "You will be given a guess and feedback in the form of G (green), Y (yellow), and B (black).\n"
                "G means the letter is in the correct position.\n"
                "Y means the letter is in the word but in the wrong position.\n"
                "B means the letter is not in the word.\n"
                "Your task is to guess the solution word.\n"
                "I have selected the word, now start guessing!\n"
                "From now on, only respond with the guess, format the guess as \{\"guess\":\"<WORD>\"\}.\n"
            )

        while attempts < max_guesses:
            job = client.submit(
                message={"text": prompt.strip(), "files": []},
                api_name="/chat",
            )
            while not job.done():
                time.sleep(0.1)
            response, _history = job.outputs()[-1]
            print("⇢ model said:", repr(_history))
            guess = sanitize_guess(_history[-1].get("content", ""))
            if not guess or (len(guess) != 5 or guess not in WORD_LIST):
                print(f"Warning: '{guess}' invalid; retrying without using a turn.")
                prompt = "The guess is not 5 letters, not in the word list, or doesn't follow the schema. Please try again.\n"
                time.sleep(5)
                continue
            print(f"Initial guess: {guess}")
            feedback = compute_feedback(guess, solution)
            print(f"Feedback: {feedback}")
            prompt = f"Guess: {guess}, Feedback: {feedback}\n"
            guesses.append((guess, feedback))
            attempts += 1
            print(f"Attempt {attempts}: {guess} -> {feedback}")
            if feedback == "GGGGG":
                break
        results.append(
            {
                "solution": solution,
                "guesses": guesses,
                "solved": guesses[-1][1] == "GGGGG" if guesses else False,
                "turns": len(guesses),
                "time": time.time() - start_time,
            }
        )
        # write entire results to file
        with open(out_path, "w") as f:
            f.write(json.dumps(results, indent=2) + "\n")

    print(f"Benchmark complete, results saved to {out_path}")
    return results

if __name__ == "__main__":
    print(benchmark_wordle(num_games=10, max_guesses=20))
