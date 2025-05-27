#!/usr/bin/env python3
# benchmark_custom.py
"""
Benchmark a local Q&A dataset (JEE style) with a Gradio agent.

Example usage:
    python3 benchmark_custom.py \
        --data_file jee_questions.json \
        --num_samples 100 \
        --offset 0
"""
import argparse, json, os, random, time, re
from datetime import datetime
from time import sleep

from gradio_client import Client


def get_last_assistant_content(resp):
    """Return the last assistant utterance from gradio_client chat history."""
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""
    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue
        if turn.get("content"):
            return turn["content"]
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return out
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""


def load_dataset_file(path: str):
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(l) for l in f]
    with open(path) as f:
        return json.load(f)


def benchmark_local(
    data_file: str,
    num_samples: int = 100,
    offset: int = 0,
    agent_url: str = "http://127.0.0.1:7860/",
):
    # ---------- load ----------
    all_samples = load_dataset_file(data_file)
    if offset >= len(all_samples):
        print(f"Offset {offset} ≥ dataset size {len(all_samples)} – nothing to do.")
        return []
    all_samples = all_samples[offset:]

    # ---------- sample ----------
    if len(all_samples) > num_samples:
        random.seed(42)
        samples = random.sample(all_samples, num_samples)
    else:
        samples = all_samples
        print(f"Dataset smaller than requested; using {len(samples)} samples.")

    # ---------- setup ----------
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/local_benchmark_{timestamp}.jsonl"

    client = Client(agent_url)
    client.predict(
        modeIndexes=[
            "ENABLE_AGENT_CREATION",
            "ENABLE_LOCAL_AGENTS",
            "ENABLE_CLOUD_AGENTS",
            "ENABLE_TOOL_CREATION",
            "ENABLE_TOOL_INVOCATION",
            "ENABLE_RESOURCE_BUDGET",
            "ENABLE_ECONOMY_BUDGET",
        ],
        api_name="/update_model",
    )

    # ---------- run ----------
    results = []
    for i, sample in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] {sample.get('description','')} (index={sample.get('index')})")

        prompt = (
            sample["question"]
            + "\n\nSolve the above question. "
            "You MUST NOT ask the user for clarifications. "
            "You MUST use tools/agents to help you. "
            "Deep-research and answer the question always. "
            "Give your answer in the form FINAL ANSWER: <answer>."
            "ONLY give the final answer letter(s) or number(s) "
            "without any additional text or explanation.\n"
        )
        target_answer = sample.get("gold", "").strip()

        try:
            t0 = time.time()
            response, history = client.predict(
                message={"text": prompt, "files": []},
                api_name="/chat",
            )
            agent_final = get_last_assistant_content(history)

            # if agent hasn’t produced “FINAL ANSWER” yet, keep nudging
            while "FINAL ANSWER" not in agent_final.upper():
                print("  …waiting for FINAL ANSWER")
                sleep(5)
                response, history = client.predict(
                    {"text": "Please finish and output the FINAL ANSWER line.", "files": []},
                    history,
                    api_name="/chat",
                )
                agent_final = get_last_assistant_content(history)

            elapsed = time.time() - t0
            is_correct = bool(
                target_answer
                and target_answer.lower() in agent_final.lower()
            )

            result = {
                "sample_index": sample.get("index"),
                "description": sample.get("description"),
                "subject": sample.get("subject"),
                "input_prompt": prompt,
                "target_answer": target_answer,
                "agent_final_response": agent_final,
                "only_final_answer": agent_final.split("FINAL ANSWER:")[-1].strip(),
                "full_history": history,
                "response_time": elapsed,
                "is_correct": target_answer in agent_final.split("FINAL ANSWER:")[-1].strip(),
            }
            results.append(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f"  ✔ done in {elapsed:.1f}s – correct={is_correct}")

            time.sleep(30)  # be gentle to server
        except Exception as e:
            print(f"  ✖ error: {e}")

    # ---------- summary ----------
    correct = sum(r["is_correct"] for r in results)
    avg_time = sum(r["response_time"] for r in results) / len(results) if results else 0
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total samples: {len(results)}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {100*correct/len(results):.2f}%")
    print(f"Avg. response: {avg_time:.2f}s")
    print(f"Saved to:      {results_file}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="Path to .json or .jsonl file")
    ap.add_argument("--num_samples", type=int, default=100, help="How many to sample")
    ap.add_argument("--offset", type=int, default=0, help="Skip first N records")
    ap.add_argument("--agent_url", default="http://127.0.0.1:7860/", help="Gradio server URL")
    args = ap.parse_args()

    benchmark_local(
        data_file=args.data_file,
        num_samples=args.num_samples,
        offset=args.offset,
        agent_url=args.agent_url,
    )
