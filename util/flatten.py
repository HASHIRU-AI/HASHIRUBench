import json
import sys
import re

def flatten_jsonl(input_file, output_file):
    with open(input_file, 'r') as f:
        raw = f.read()

    # Split based on a naive assumption of object ends
    raw_objects = raw.split('\n}\n')
    with open(output_file, 'w') as fout:
        for i, obj_str in enumerate(raw_objects):
            obj_str = obj_str.strip()
            if not obj_str:
                continue
            if not obj_str.endswith('}'):
                obj_str += '}'

            # Fix illegal trailing commas before closing braces/brackets
            obj_str = re.sub(r',\s*([\]}])', r'\1', obj_str)

            try:
                parsed = json.loads(obj_str)
                fout.write(json.dumps(parsed) + '\n')
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {i}: {e}")
# Usage: python flatten_jsonl.py input.json output.jsonl
if __name__ == "__main__":
    flatten_jsonl(sys.argv[1], sys.argv[2])
