import pandas as pd
import argparse
import os

def convert_parquet_to_jsonl(input_path, output_path):
    """
    Reads a Parquet file and converts it to a JSONL file.

    Args:
        input_path (str): The path to the input Parquet file.
        output_path (str): The path to the output JSONL file.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at '{input_path}'")
            return

        print(f"Reading Parquet file from '{input_path}'...")
        # Read the Parquet file into a pandas DataFrame
        # The 'pyarrow' engine is typically used for reading Parquet files.
        df = pd.read_parquet(input_path, engine='pyarrow')

        print(f"Converting data and writing to '{output_path}'...")
        # Write the DataFrame to a JSONL file
        # - orient='records': Each row will be a JSON object.
        # - lines=True: Each JSON object will be on a new line.
        df.to_json(output_path, orient='records', lines=True)

        print("Conversion complete!")
        print(f"Successfully converted '{input_path}' to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file to a JSONL file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Add arguments for input and output file paths
    parser.add_argument(
        "input_file",
        help="Path to the input Parquet file."
    )
    parser.add_argument(
        "output_file",
        help="Path for the output JSONL file."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the conversion function with the provided file paths
    convert_parquet_to_jsonl(args.input_file, args.output_file)
