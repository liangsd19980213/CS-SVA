# import os
# import time
# import logging
# import pandas as pd
# from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaForSequenceClassification, Trainer, TrainingArguments
# from torch.utils.data import Dataset
#
# # Initialize logging
# logging.basicConfig(
#     filename="training_log.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# logging.getLogger().addHandler(console)
#
# logging.info("Model training started...")
#
# # Define custom dataset class
# class CodeSeverityDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len=256, task_type="classification"):
#         self.dataframe = dataframe
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.task_type = task_type  # "classification" or "generation"
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         input_text = row["func_before"]
#         target_text = str(row["severity"])
#         inputs = self.tokenizer(
#             input_text,
#             max_length=self.max_len,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )
#
#         if self.task_type == "classification":
#             labels = int(target_text)  # Ensure labels are integers for classification
#         elif self.task_type == "generation":
#             labels = self.tokenizer(
#                 target_text,
#                 max_length=10,
#                 padding="max_length",
#                 truncation=True,
#                 return_tensors="pt",
#             )["input_ids"].squeeze(0)
#         else:
#             raise ValueError("Invalid task_type. Choose 'classification' or 'generation'.")
#
#         return {
#             "input_ids": inputs["input_ids"].squeeze(0),  # Remove batch dimension
#             "attention_mask": inputs["attention_mask"].squeeze(0),  # Remove batch dimension
#             "labels": labels,
#         }
#
# # Ensure the 'dataset' directory exists
# dataset_dir = './dataset/'
# os.makedirs(dataset_dir, exist_ok=True)
#
# # Check if dataset files already exist
# train_file = os.path.join(dataset_dir, 'train_data.csv')
# val_file = os.path.join(dataset_dir, 'val_data.csv')
# test_file = os.path.join(dataset_dir, 'test_data.csv')
#
# if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
#     logging.info("Loading existing dataset splits...")
#     train_data = pd.read_csv(train_file)
#     val_data = pd.read_csv(val_file)
#     test_data = pd.read_csv(test_file)
# else:
#     logging.info("Dataset splits not found. Creating new splits...")
#     # Load the dataset
#     data = pd.read_excel('../../dataset/megavul_simple_cpp_processed.xlsx')
#     train_size = int(0.8 * len(data))
#     val_size = int(0.1 * len(data))
#     test_size = len(data) - train_size - val_size
#
#     train_data = data[:train_size]
#     val_data = data[train_size:train_size+val_size]
#     test_data = data[train_size+val_size:]
#
#     # Save dataset splits to CSV
#     train_data.to_csv(train_file, index=False)
#     val_data.to_csv(val_file, index=False)
#     test_data.to_csv(test_file, index=False)
#     logging.info("Dataset splits created and saved.")
#
# # Model selection
# def initialize_model_and_tokenizer(model_name):
#     if model_name.lower() == "codet5":
#         tokenizer = RobertaTokenizer.from_pretrained("../../model/codet5")
#         model = T5ForConditionalGeneration.from_pretrained("../../model/codet5")
#         task_type = "generation"
#     elif model_name.lower() == "codebert":
#         tokenizer = RobertaTokenizer.from_pretrained("../../model/codebert")
#         model = RobertaForSequenceClassification.from_pretrained("../../model/codebert", num_labels=4)
#         task_type = "classification"
#     else:
#         raise ValueError("Invalid model name. Choose either 'codet5' or 'codebert'.")
#     return model, tokenizer, task_type
#
# # Set the model name (choose either 'codet5' or 'codebert')
# model_name = "codebert"  # Change to 'codet5' if needed
# model, tokenizer, task_type = initialize_model_and_tokenizer(model_name)
# logging.info(f"Selected model: {model_name}")
#
# # Create dataset objects
# train_dataset = CodeSeverityDataset(train_data, tokenizer, max_len=256, task_type=task_type)
# val_dataset = CodeSeverityDataset(val_data, tokenizer, max_len=256, task_type=task_type)
#
# # Training arguments
# training_args = TrainingArguments(
#     output_dir="./codebert-results",
#     num_train_epochs=4,
#     per_device_train_batch_size=8,
#     gradient_accumulation_steps=2,
#     learning_rate=5e-5,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     save_total_limit=2,
#     logging_dir="./logs",
#     logging_steps=1000,  # Logging frequency
#     load_best_model_at_end=True,
#     report_to="none",  # Disable reporting tools like wandb
# )
#
# # Initialize Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=tokenizer,
# )
#
# # Record training time
# start_time = time.time()
#
# logging.info("Starting model training...")
# trainer.train()
#
# end_time = time.time()
# elapsed_time = end_time - start_time
# logging.info(f"Model training completed in {elapsed_time:.2f} seconds.")
#
# # Save model
# trainer.save_model(f"./trained_model_{model_name}")
# logging.info(f"Model saved to './trained_model_{model_name}'.")
#
# print(f"Total training time: {elapsed_time:.2f} seconds")





import pandas as pd
from transformers import Trainer, T5ForConditionalGeneration, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
import numpy as np

# Define custom dataset class
class CodeSeverityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256, task_type="classification"):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type  # "classification" or "generation"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_text = row["func_before"]
        target_text = str(row["severity"])
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.task_type == "classification":
            labels = int(target_text)  # Ensure labels are integers for classification
        elif self.task_type == "generation":
            labels = self.tokenizer(
                target_text,
                max_length=10,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
        else:
            raise ValueError("Invalid task_type. Choose 'classification' or 'generation'.")

        return {
            "input_ids": inputs["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": inputs["attention_mask"].squeeze(0),  # Remove batch dimension
            "labels": labels,
        }

# Load the test dataset
test_data = pd.read_csv('./dataset/test_data.csv')

# Select model (Change 'codet5' to 'codebert' for CodeBERT)
model_name = "codebert"  # Change to 'codet5' if using CodeT5

# Load model and tokenizer
if model_name == "codet5":
    model = T5ForConditionalGeneration.from_pretrained("./trained_model_codet5")
    tokenizer = RobertaTokenizer.from_pretrained("../../model/codet5")
    task_type = "generation"
elif model_name == "codebert":
    model = RobertaForSequenceClassification.from_pretrained("./trained_model_codebert")
    tokenizer = RobertaTokenizer.from_pretrained("../../model/codebert")
    task_type = "classification"
else:
    raise ValueError("Invalid model name. Choose either 'codet5' or 'codebert'.")

# Prepare the test dataset
test_dataset = CodeSeverityDataset(test_data, tokenizer, max_len=256, task_type=task_type)

# Use Trainer for inference
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)

# Make predictions
print("Starting testing...")
predictions = trainer.predict(test_dataset)

# Inspect predictions structure
print(f"Predictions structure: {type(predictions.predictions)}")

# Extract logits from predictions
logits = predictions.predictions  # Get the logits directly
if isinstance(logits, tuple):  # Handle the case if logits are a tuple
    logits = logits[0]  # Extract the first element

# Convert logits to predicted class IDs
predicted_ids = np.argmax(logits, axis=-1)  # Use numpy to get predicted class IDs
true_labels = test_data["severity"].tolist()  # Get true labels

# Convert predictions for CodeT5
if model_name == "codet5":
    predicted_labels = [
        int(tokenizer.decode(pred, skip_special_tokens=True)) for pred in predicted_ids
    ]
else:
    predicted_labels = predicted_ids.tolist()

# Compute evaluation metrics
macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
macro_mcc = matthews_corrcoef(true_labels, predicted_labels)

# Output results
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Macro MCC: {macro_mcc:.4f}")






