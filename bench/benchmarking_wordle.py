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
    """Return a 5-char string of G/Y/X feedback."""
    feedback = ["X"] * 5
    sol = list(solution)
    # Greens first
    for i, g in enumerate(guess):
        if g == sol[i]:
            feedback[i], sol[i] = "G", None
    # Yellows second
    for i, g in enumerate(guess):
        if feedback[i] == "X" and g in sol:
            feedback[i] = "Y"
            sol[sol.index(g)] = None
    return "".join(feedback)

def sanitize_guess(raw: str) -> str:
    """Extract a 5-letter guess from any model output."""
    raw = raw.lower()
    regex = r"{[\"\']guess[\"\']:\s*[\"\'](\w*)[\"\']"
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
        # client.predict(
        #     modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
        #     api_name="/update_model"
        # )
        client.reset_session()
        solution = random.choice(WORD_LIST)
        print(f"Game {gi+1}/{num_games}, solution: {solution}")
        guesses, attempts = [], 0
        start_time = time.time()

        prompt = (
                "We are playing a game of Wordle. The solution is a 5-letter word.\n"
                f"Your task is to guess the solution word within {max_guesses} attempts.\n"
                "You will be given feedback in the form of G (Correct position), Y (in the word but wrong position), and X (does not exist in the word).\n"
                "Your task is to guess the solution word.\n"
                "Use agents and tools as nessesary to guess the word.\n"
                "From now on, the final response should be in this format: \{\"guess\":\"<WORD>\"\}.\n"
                "I have selected the word, now start guessing!\n"
            )
        letters_not_in_word = set()

        response = ""
        _history = []
        
        while attempts < max_guesses:
            print(f"Attempt {attempts + 1}/{max_guesses}")
            print("prompt:", repr(prompt))
            response, _history = client.predict(
                {"text": prompt.strip()},
                _history,
                api_name="/chat",
            )
            print("response:", repr(response))
            print("history:", repr(_history[-1]))
            guess = sanitize_guess(_history[-1].get("content", ""))
            if not guess:
                print("Warning: empty guess; retrying without using a turn.")
                prompt = "The guess is empty. This could be due to timeout or output not matching the expected format (\{\"guess\":\"<WORD>\"\}). Please try again.\n"
                time.sleep(60)
                continue
            if len(guess) != 5:
                print(f"guess: {guess} not 5 letters")
                prompt = "The guess is not 5 letters. Please try again.\n"
                time.sleep(5)
                continue
            if guess not in WORD_LIST:
                print(f"guess: {guess} not in word list")
                prompt = "The guess is not in the word list. Please try again.\n"
                time.sleep(5)
                continue
            print(f"Initial guess: {guess}")
            feedback = compute_feedback(guess, solution)
            for i in range(5):
                letter = feedback[i]
                if letter == "X":
                    letters_not_in_word.add(guess[i])
            print(f"Feedback: {feedback}")
            guesses.append((guess, feedback))
            attempts += 1
            prompt = {
                "guess": guess,
                "feedback": feedback,
                "letters_not_in_word": letters_not_in_word,
                "attempts": f"{attempts}/{max_guesses}",
            }
            prompt = str(prompt)
            print(f"Attempt {attempts}: {guess} -> {feedback}")
            if feedback == "GGGGG":
                break
            time.sleep(5)
        results.append(
            {
                "solution": solution,
                "guesses": guesses,
                "solved": guesses[-1][1] == "GGGGG" if guesses else False,
                "turns": len(guesses),
                "time": time.time() - start_time,
                "conversation": _history
            }
        )
        # write entire results to file
        with open(out_path, "w") as f:
            f.write(json.dumps(results, indent=2) + "\n")
        # # set D:\Projects\AI\HASHIRU\src\models\models.json to {}
        # with open("D:\\Projects\\AI\\HASHIRU\\src\\models\\models.json", "w") as f:
        #     f.write("{}")
        # # set D:\Projects\AI\HASHIRU\src\data\memory.json to []
        # with open("D:\\Projects\\AI\\HASHIRU\\src\\data\\memory.json", "w") as f:
        #     f.write("[]")

    print(f"Benchmark complete, results saved to {out_path}")
    return results

if __name__ == "__main__":
    print(benchmark_wordle(num_games=1000, max_guesses=6))
