import os
import argparse
from pathlib import Path

def combine_jsonl_files(input_dir, output_file):
    """
    Combines all .jsonl files from an input directory into a single output .jsonl file.

    Args:
        input_dir (str): The path to the directory containing the .jsonl files.
        output_file (str): The path for the combined output .jsonl file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Searching for .jsonl files in: {input_path.resolve()}")
    
    jsonl_files = sorted(list(input_path.glob('*.jsonl'))) # Sort to ensure consistent order
    
    if not jsonl_files:
        print("No .jsonl files found in the specified directory.")
        return

    print(f"Found {len(jsonl_files)} files to merge.")
    
    lines_written = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for filename in jsonl_files:
                print(f"Processing '{filename.name}'...")
                with open(filename, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # You can add validation here to ensure the line is valid JSON if needed
                        outfile.write(line)
                        lines_written += 1
        
        print(f"\nSuccessfully combined {len(jsonl_files)} files into '{output_path.resolve()}'.")
        print(f"Total lines written: {lines_written}")

    except IOError as e:
        print(f"An I/O error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Combine multiple .jsonl files from a directory into one single file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_dir", 
        type=str,
        help="The path to the directory containing the .jsonl files."
    )
    
    parser.add_argument(
        "output_file", 
        type=str,
        help="The path for the combined output .jsonl file."
    )

    args = parser.parse_args()
    
    combine_jsonl_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
