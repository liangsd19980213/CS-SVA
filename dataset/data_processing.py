import os
import pandas as pd

# Define constants
DATA_DIR = './'  # Directory for input files
OUTPUT_DIR = os.path.join(DATA_DIR, 'slim_40')  # Directory for processed outputs


def preprocess_to_single_line(file_path, output_file_name):
    """
    Preprocess the dataset by extracting specific columns and formatting them into a single line.
    Each line contains: [severity]<CODESPLIT>[git_url]<CODESPLIT>[func_name]<CODESPLIT>[diff_func]<CODESPLIT>[func_before].

    Args:
        file_path (str): Path to the input CSV file.
        output_file_name (str): Name of the output file to save results.
    """
    # Step 1: Load the CSV file
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)

    # Step 2: Ensure required columns exist
    required_columns = ['severity', 'func_before']
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' is required in the CSV file but not found.")

    # Step 3: Extract and format data
    print("Extracting and processing data...")
    formatted_entries = []
    for row in data.itertuples(index=False):
        severity = str(row.severity).strip()
        func_before = str(row.func_before).replace("\n", " ").strip()  # Replace newlines and trim
        formatted_entry = f"{severity}<CODESPLIT>{func_before}"
        formatted_entries.append(formatted_entry)

    # Step 4: Save processed data to a single output file
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, output_file_name)

    print(f"Saving processed data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(formatted_entries))
    print("Processing completed.")


def main():
    """
    Main function to preprocess training, validation, and test datasets.
    """
    # Input file paths (update as needed)
    train_csv_path = os.path.join(OUTPUT_DIR, 'train_data.csv')
    val_csv_path = os.path.join(OUTPUT_DIR, 'val_data.csv')
    test_csv_path = os.path.join(OUTPUT_DIR, 'test_data.csv')

    # Preprocess datasets
    preprocess_to_single_line(train_csv_path, output_file_name='train_processed.txt')
    preprocess_to_single_line(val_csv_path, output_file_name='val_processed.txt')
    preprocess_to_single_line(test_csv_path, output_file_name='test_processed.txt')


if __name__ == '__main__':
    main()
