#!/usr/bin/env python3
import json
import argparse
import time
from tqdm import tqdm

from gradio_client import Client
from sentence_transformers import SentenceTransformer, util

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GRADIO_URL  = "http://127.0.0.1:7860/"      # your Gradio server
GRADIO_API  = "/chat"                       # chat endpoint
EMBED_MODEL = "all-MiniLM-L6-v2"
# ────────────────────────────────────────────────────────────────────────────────

def get_last_assistant_content(resp):
    """
    Return the last assistant utterance from the response object
    produced by `client.predict`.
    """
    # If wrapped in (messages, meta)
    if isinstance(resp, tuple):
        resp = resp[0]
    if not isinstance(resp, list):
        return ""

    for turn in reversed(resp):
        if turn.get("role") != "assistant":
            continue
        # plain content
        if turn.get("content"):
            return turn["content"]
        # function response
        fr = turn.get("function_response", {})
        out = fr.get("result", {}).get("output")
        if out:
            return out
        # parts
        cont = turn.get("content")
        if isinstance(cont, dict):
            parts = cont.get("parts", [])
            if parts and parts[0].get("text"):
                return parts[0]["text"]
    return ""

def call_llm_via_gradio(client: Client, prompt: str):
    """
    Send `prompt` to Gradio and return the extracted FINAL ANSWER text.
    """
    history = []
    # initial chat call
    resp, history = client.predict(
        {"text": prompt, "files": []},
        history,
        api_name=GRADIO_API
    )
    answer = get_last_assistant_content(history)

    # if it didn’t finish with FINAL ANSWER, keep asking to continue
    while "FINAL ANSWER" not in answer.upper():
        time.sleep(2)
        resp, history = client.predict(
            {"text": "Please finish and give me the FINAL ANSWER line.", "files": []},
            history,
            api_name=GRADIO_API
        )
        answer = get_last_assistant_content(history)

    # extract after the colon
    for line in answer.splitlines()[::-1]:
        if "FINAL ANSWER:" in line.upper():
            return line.split(":",1)[1].strip()
    return answer.strip()

def main(input_path, output_path, delay):
    # init embedder and gradio client
    embedder = SentenceTransformer(EMBED_MODEL)
    client   = Client(GRADIO_URL)
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for raw in tqdm(f_in):
            sample = json.loads(raw)

            # 1) Build prompt
            prompt = (
                f"Feel free to use tools or agents to help you answer the question. "
                f"You can decide what tools or agents to use. "
                f"{sample['context'].rstrip()}\n\n"
                f"Q: {sample['question'].strip()}\n\n"
                "Respond with a single line: FINAL ANSWER: <your answer>."
            )

            # 2) Call LLM via Gradio
            try:
                llm_ans = call_llm_via_gradio(client, prompt)
            except Exception as e:
                print(f"[ERROR] sample id={sample.get('id','?')}: {e}")
                continue

            # 3) Embed options + LLM ans
            options = [
                sample.get("answerA",""),
                sample.get("answerB",""),
                sample.get("answerC",""),
            ]
            embeddings = embedder.encode(options + [llm_ans], convert_to_tensor=True)
            opts_emb, ans_emb = embeddings[:3], embeddings[3]

            # 4) Pick most similar
            sims = util.cos_sim(ans_emb, opts_emb)[0]
            pred_idx = int(sims.argmax().item())
            sample["predicted_label"] = str(pred_idx+1)
            sample["is_correct"] = (sample["predicted_label"] == sample.get("label",""))

            # 5) Store LLM answer
            sample["llm_answer"] = llm_ans

            # write out enriched record
            f_out.write(json.dumps(sample) + "\n")

            if delay:
                time.sleep(delay)

    print(f"✅ Done! Wrote results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL with Gradio LLM + MiniLM embeddings"
    )
    parser.add_argument("input",  help="path to input .jsonl")
    parser.add_argument("output", help="path to output .jsonl")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds to sleep between calls")
    args = parser.parse_args()

    main(args.input, args.output, args.delay)
