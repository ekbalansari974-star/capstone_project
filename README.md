# capstone_project
Here is the Python code:
# Part 1 - Question 1

import pandas as pd

# Load the dataset
df = pd.read_csv("test.csv")

# Print first 5 rows
print("First 5 Rows:")
print(df.head())

# Print column data types
print("\nColumn Data Types:")
print(df.dtypes)

# Print dataset shape
print("\nDataset Shape (Rows, Columns):")
print(df.shape)