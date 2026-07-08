import kagglehub

# Download latest version
path = kagglehub.dataset_download("prakharrathi25/banking-dataset-marketing-targets")

print("Path to dataset files:", path)





import pandas as pd
import os

# Construct the full path to the dataset file, using 'train.csv'
data_file_path = os.path.join(path, "train.csv")

# Load the dataset, specifying semicolon as the delimiter
df = pd.read_csv(data_file_path, sep=';')

# Print first 5 rows
print("First 5 Rows:")
print(df.head())

# Print column data types
print("\nColumn Data Types:")
print(df.dtypes)

# Print dataset shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)