import json

def convert_json_to_jsonl(input_file_path, output_file_path):
    """
    Converts a file containing a list of JSON objects to JSONL format.

    Args:
        input_file_path (str): Path to the input file (list of JSON objects).
        output_file_path (str): Path to the output JSONL file.
    """
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            # Load the entire list of JSON objects from the input file
            list_of_objects = json.load(infile)

            # Ensure it's actually a list
            if not isinstance(list_of_objects, list):
                print(f"Error: Input file '{input_file_path}' does not contain a JSON list.")
                return

            # Write each JSON object as a new line in the output file
            for json_object in list_of_objects:
                json.dump(json_object, outfile)  # Writes the object
                outfile.write('\n')             # Adds a newline character

        print(f"Successfully converted '{input_file_path}' to '{output_file_path}' (JSONL).")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file_path}'. Make sure it's a valid JSON list.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON file to JSONL format.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("output_file", help="Path to the output JSONL file.")
    
    args = parser.parse_args()
    
    convert_json_to_jsonl(args.input_file, args.output_file)