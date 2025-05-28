import json
from scipy.stats import ttest_rel

# load two JSONL files
def load_jsonl(filepath):
    with open(filepath, "r") as f:
        lines = []
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {i} invalid: {e}")
        return lines
    
def compare_results(file1, file2):
    data1 = load_jsonl(file1)
    data2 = load_jsonl(file2)

    if len(data1) != len(data2):
        # trim to the smaller size
        min_length = min(len(data1), len(data2))
        print(f"Warning: Files have different lengths ({len(data1)} vs {len(data2)}). Trimming to {min_length} entries.")
        data1 = data1[:min_length]
        data2 = data2[:min_length]


    # Extract relevant metrics for comparison
    metrics1 = [int(item['is_correct']) for item in data1]
    metrics2 = [int(item['is_correct']) for item in data2]

    # Perform paired t-test
    t_stat, p_value = ttest_rel(metrics1, metrics2)

    return t_stat, p_value

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two JSONL result files using paired t-test.")
    parser.add_argument("file1", type=str, help="Path to the first JSONL file.")
    parser.add_argument("file2", type=str, help="Path to the second JSONL file.")
    
    args = parser.parse_args()

    t_stat, p_value = compare_results(args.file1, args.file2)
    
    print(f"T-statistic: {t_stat}, P-value: {p_value}")