import pandas as pd
import numpy as np

def clean_column_name(col_name):
    """Clean column name by removing special characters and replacing spaces with underscores"""
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

# Load both datasets
print("Loading datasets...")
train_df = pd.read_excel('filled_data.xlsx')
test_df = pd.read_excel('../Data/Test-Truncated-Restated.xlsx')

# Get column names from both datasets
train_columns = set(train_df.columns)
test_columns = set(test_df.columns)

# Clean column names for comparison
train_columns_clean = {clean_column_name(col) for col in train_columns}
test_columns_clean = {clean_column_name(col) for col in test_columns}

# Find differences
columns_in_train_not_in_test = train_columns_clean - test_columns_clean
columns_in_test_not_in_train = test_columns_clean - train_columns_clean
common_columns = train_columns_clean.intersection(test_columns_clean)

# Print results
print("\nDataset Comparison Results")
print("=========================")
print(f"\nTraining dataset (filled_data.xlsx):")
print(f"Number of columns: {len(train_columns)}")
print(f"Columns: {sorted(train_columns)}")

print(f"\nTest dataset (Test-Truncated-Restated.xlsx):")
print(f"Number of columns: {len(test_columns)}")
print(f"Columns: {sorted(test_columns)}")

print(f"\nColumns in training but not in test ({len(columns_in_train_not_in_test)}):")
for col in sorted(columns_in_train_not_in_test):
    print(f"- {col}")

print(f"\nColumns in test but not in training ({len(columns_in_test_not_in_train)}):")
for col in sorted(columns_in_test_not_in_train):
    print(f"- {col}")

print(f"\nCommon columns ({len(common_columns)}):")
for col in sorted(common_columns):
    print(f"- {col}")

# Save results to a file
with open('dataset_comparison.txt', 'w') as f:
    f.write("Dataset Comparison Results\n")
    f.write("=========================\n\n")
    
    f.write(f"Training dataset (filled_data.xlsx):\n")
    f.write(f"Number of columns: {len(train_columns)}\n")
    f.write("Columns:\n")
    for col in sorted(train_columns):
        f.write(f"- {col}\n")
    
    f.write(f"\nTest dataset (Test-Truncated-Restated.xlsx):\n")
    f.write(f"Number of columns: {len(test_columns)}\n")
    f.write("Columns:\n")
    for col in sorted(test_columns):
        f.write(f"- {col}\n")
    
    f.write(f"\nColumns in training but not in test ({len(columns_in_train_not_in_test)}):\n")
    for col in sorted(columns_in_train_not_in_test):
        f.write(f"- {col}\n")
    
    f.write(f"\nColumns in test but not in training ({len(columns_in_test_not_in_train)}):\n")
    for col in sorted(columns_in_test_not_in_train):
        f.write(f"- {col}\n")
    
    f.write(f"\nCommon columns ({len(common_columns)}):\n")
    for col in sorted(common_columns):
        f.write(f"- {col}\n")

print("\nResults have been saved to 'dataset_comparison.txt'") 