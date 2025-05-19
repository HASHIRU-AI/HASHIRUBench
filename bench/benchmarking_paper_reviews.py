from gradio_client import Client
import pandas as pd
import json
import time
import os
from datetime import datetime
from time import sleep

def get_last_assistant_content(resp):
    """
    Return the last assistant utterance from the response object
    produced by `client.predict`.
    """
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

def benchmark_paper_reviews(
    csv_path,
    id_col="ID",
    text_col="concatenated_text",
    num_samples=None,
    offset=0,
    output_dir="results"
):
    """
    Benchmark agent performance on paper reviews.

    Args:
        csv_path: path to the pipe-separated CSV of papers + existing reviews
        id_col:    name of the column containing unique paper IDs
        text_col:  name of the column containing the full paper text
        num_samples: if set, randomly sample this many papers
        output_dir: where to write the JSONL results
    """
    # load CSV
    df = pd.read_csv(csv_path, sep="|")
    if offset or num_samples:
        end = offset + num_samples if num_samples else None
        df = df.iloc[offset:end].reset_index(drop=True)
    # prepare output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"paper_review_benchmark_{timestamp}.jsonl")
    print(f"Writing results to {out_path}")

    # init client
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )


    results = []
    for idx, row in df.iterrows():
        paper_id = row[id_col]
        title = row["Title"]
        prompt = "Create THREE agents with relevant personalities, expertise, and review styles. " \
                "Each agent should provide a review of the paper, and recommend Accept/Reject for ICLR 2023. " \
                "The review should be detailed and include strengths and weaknesses. " \
                "Finish the entire review and DO NOT STOP in the middle. " \
                "GIVE A FINAL DECISION in the form of \"FINAL DECISION: <Accept/Reject>\". " \
                "The paper title is: " + title + "\n\n" + row[text_col]
        print(f"[{idx+1}/{len(df)}] Paper ID: {paper_id}")

        try:
            start = time.time()
            resp, history = client.predict(
                {"text": prompt, "files": []},
                None,
                api_name="/chat"
            )

            content = get_last_assistant_content(history)
            print(content)
            while "FINAL DECISION" not in content.upper():
                sleep(5)
                print("…no final verdict yet, asking the agent to continue")
                resp, history = client.predict(
                    # send just “continue” (or “please continue”)
                    {"text": "Please finish the review and give the FINAL DECISION line.", "files": []},
                    history,            # include the full chat history
                    api_name="/chat"
                )
                content = get_last_assistant_content(history)
                print(content)
                sleep(5)
            elapsed = time.time() - start

            result = {
                "paper_id": paper_id,
                "prompt_snippet": prompt[:200],
                "agent_review": history,
                "ground_truth": row["Decision"],
                "response_time": elapsed
            }

            # write immediately
            with open(out_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(f" → {elapsed:.2f}s, review length {len(history)} chars")
            results.append(result)

            # small delay
            time.sleep(5)
        except Exception as e:
            print(f"  Error on {paper_id}: {e}")

    print(f"\nDone. Results written to {out_path}")
    return results

if __name__ == "__main__":
    # example usage: adjust path & sample count as needed
    benchmark_paper_reviews(
        csv_path="bench/data/ICLR_2023.csv",
        num_samples=1,
        offset=7,
    )
