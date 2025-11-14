import json

def transform_jsonl_for_training(input_file_path: str, output_file_path: str):
    """
    Converts a JSONL file from its original complex format into a simple
    'prompt':'completion' format suitable for fine-tuning a language model.

    Specifically, it maps 'code_ground_truth' to 'prompt' and the code from
    each entry in 'unit_tests' to a 'completion'. This process can generate
    multiple training examples from a single input line.

    Args:
        input_file_path (str): The path to the source JSONL file.
        output_file_path (str): The path where the transformed JSONL file will be saved.
    """
    try:
        input_lines_count = 0
        output_lines_count = 0

        # Open both files safely using 'with'
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            # Process each line from the input file
            for line in infile:
                input_lines_count += 1

                # Skip any potential empty lines in the file
                if not line.strip():
                    continue

                # Parse the JSON object from the current line
                data = json.loads(line)

                # Safely extract the code and the list of unit tests
                prompt_code = data.get('code_ground_truth')
                unit_tests = data.get('unit_tests')

                # Ensure both required fields are present and valid
                if not prompt_code :
                    print(f"Skipping line {input_lines_count}: Missing 'code_ground_truth' or 'unit_tests' list.")
                    continue
                
                
                # Create the new, simplified dictionary
                new_record = {
                    "prompt": prompt_code,
                    "completion": unit_tests
                }
                
                # Write the new JSON object as a single line in the output file
                outfile.write(json.dumps(new_record) + '\n')
                output_lines_count += 1
            

        print("\nTransformation complete!")
        print(f"Processed {input_lines_count} lines from '{input_file_path}'.")
        print(f"Generated {output_lines_count} training examples in '{output_file_path}'.")

    except FileNotFoundError:
        print(f"Error: The input file '{input_file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON on line {input_lines_count}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Example Usage ---
if __name__ == '__main__':

    input_filename = 'D:/Projects/ai-code-reviewer/support/0002.jsonl'
    output_filename = 'train_data/training_dataset_0004.jsonl'

    transform_jsonl_for_training(input_filename, output_filename)

    print("------------------------------------")
