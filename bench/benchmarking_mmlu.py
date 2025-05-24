import os
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from gradio_client import Client

API_KEY = ""
random.seed(12345)

def get_client():
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )
    return client


def call_api(client, instruction, inputs, tries=0):
    if tries > 3:
        print("Error: too many tries")
        return None
    client = Client("http://127.0.0.1:7860/")
    client.predict(
		modeIndexes=["ENABLE_AGENT_CREATION","ENABLE_LOCAL_AGENTS","ENABLE_CLOUD_AGENTS","ENABLE_TOOL_CREATION","ENABLE_TOOL_INVOCATION","ENABLE_RESOURCE_BUDGET","ENABLE_ECONOMY_BUDGET"],
		api_name="/update_model"
    )
    start = time.time()
    response, history = client.predict(
                message={"text": instruction + inputs, "files": []},
                api_name="/chat"
            )
    if 'error' in response["content"]:
        time.sleep(60)
        response = call_api(client, instruction, inputs, tries + 1)
        return response
        
    print("cost time", time.time() - start)
    return response["content"]


def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res


def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def single_request(client, single_question, cot_examples_dict, exist_result):
    exist = True
    q_id = single_question["question_id"]
    for each in exist_result:
        if q_id == each["question_id"] and single_question["question"] == each["question"]:
            pred = extract_answer(each["model_outputs"])
            return pred, each["model_outputs"], exist
    exist = False
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    for each in cot_examples:
        prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        response = call_api(client, prompt, input_text)
        response = response.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, None, exist
    pred = extract_answer(response)
    return pred, response, exist


def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record


def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res


def evaluate(subjects):
    client = get_client()
    test_df, dev_df = load_mmlu_pro()
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)
    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")
        output_summary_path = os.path.join(args.output_dir, subject + "_summary.json")
        res, category_record = update_result(output_res_path)

        for each in tqdm(test_data):
            label = each["answer"]
            category = subject
            pred, response, exist = single_request(client, each, dev_df, res)
            if response is not None:
                res, category_record = update_result(output_res_path)
                if category not in category_record:
                    category_record[category] = {"corr": 0.0, "wrong": 0.0}
                each["pred"] = pred
                each["model_outputs"] = response
                merge_result(res, each)
                if pred is not None:
                    if pred == label:
                        category_record[category]["corr"] += 1
                    else:
                        category_record[category]["wrong"] += 1
                else:
                    category_record[category]["wrong"] += 1
                save_res(res, output_res_path)
                save_summary(category_record, output_summary_path)
                res, category_record = update_result(output_res_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
        else:
            continue
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))


def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"])
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong)
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="eval_results/")
    parser.add_argument("--model_name", "-m", type=str, default="gpt-4",
                        choices=["gpt-4", "gpt-4o", "o1-preview",
                                 "deepseek-chat", "deepseek-coder",
                                 "gemini-1.5-flash-latest",
                                 "gemini-1.5-pro-latest",
                                 "claude-3-opus-20240229",
                                 "gemini-1.5-flash-8b",
                                 "claude-3-sonnet-20240229",
                                 "gemini-002-pro",
                                 "gemini-002-flash"])
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    assigned_subjects = []
    args = parser.parse_args()

    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)