import pandas as pd
import os

# Define file paths
input_file = './megavul_simple_cpp_processed.xlsx'
output_dir = './slim_40/'


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the Excel file
data = pd.read_excel(input_file)

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Calculate split sizes
train_size = int(0.8 * len(data))
val_size = int(0.1 * len(data))

# Split the data
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Save the splits to CSV files
train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
val_data.to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

print("Dataset has been successfully split into training, validation, and test sets.")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
#
# DATA_DIR = './'
# OUTPUT_DIR = os.path.join(DATA_DIR, 'processed_data')
#
#
# def split_and_save_dataset(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
#     """
#     Split the dataset into training, validation, and test sets, ensuring uniform distribution of classes.
#
#     Args:
#         file_path (str): Path to the input CSV file.
#         train_ratio (float): Proportion of the dataset to include in the training set.
#         val_ratio (float): Proportion of the dataset to include in the validation set.
#         test_ratio (float): Proportion of the dataset to include in the test set.
#     """
#     # Step 1: Load data
#     print(f"Loading data from {file_path}...")
#     data = pd.read_excel(file_path)
#
#     # Step 2: Ensure required columns exist
#     required_columns = ['severity', 'func_before']
#     for column in required_columns:
#         if column not in data.columns:
#             raise ValueError(f"Column '{column}' is required in the CSV file but not found.")
#
#     # Step 3: Split data
#     print("Splitting data into train, validation, and test sets...")
#     train_data, temp_data = train_test_split(data, test_size=1 - train_ratio, stratify=data['severity'],
#                                              random_state=42)
#     val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio),
#                                            stratify=temp_data['severity'], random_state=42)
#
#     # Step 4: Save datasets
#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
#
#     train_path = os.path.join(OUTPUT_DIR, 'train_data.csv')
#     val_path = os.path.join(OUTPUT_DIR, 'val_data.csv')
#     test_path = os.path.join(OUTPUT_DIR, 'test_data.csv')
#
#     print(f"Saving train data to {train_path}...")
#     train_data.to_csv(train_path, index=False)
#
#     print(f"Saving validation data to {val_path}...")
#     val_data.to_csv(val_path, index=False)
#
#     print(f"Saving test data to {test_path}...")
#     test_data.to_csv(test_path, index=False)
#
#     print("Data splitting and saving completed.")
#
#
# def main():
#     """
#     Main function to split and preprocess the dataset.
#     """
#     # Input file path
#     input_path = os.path.join(DATA_DIR, 'megavul_simple_cpp_processed.xlsx')  # Change to your dataset file
#
#     # Split and save the dataset
#     split_and_save_dataset(input_path)
#
#
# if __name__ == '__main__':
#     main()
#     # 检查每个子集的类别分布
#     data_train = pd.read_csv('processed_data/train_data.csv')
#     data_val = pd.read_csv('processed_data/val_data.csv')
#     data_test = pd.read_csv('processed_data/test_data.csv')
#
#     print("Training set distribution:")
#     print(data_train['severity'].value_counts(normalize=True))
#     print("Validation set distribution:")
#     print(data_val['severity'].value_counts(normalize=True))
#     print("Test set distribution:")
#     print(data_test['severity'].value_counts(normalize=True))
