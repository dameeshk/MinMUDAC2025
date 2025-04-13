import pandas as pd
import numpy as np

# Load the Excel file
file_path = '/Users/dameesh/Desktop/MinMUDAC2025/Data/Test-Truncated-Restated.xlsx'
df = pd.read_excel(file_path)

print(f"Original data shape: {df.shape}")

# Check if 'Match ID 18Char' exists in the dataframe
id_column = 'Match ID 18Char'
if id_column not in df.columns:
    # Try to find a similar column name
    possible_columns = [col for col in df.columns if 'match' in col.lower() and 'id' in col.lower()]
    if possible_columns:
        id_column = possible_columns[0]
        print(f"Using '{id_column}' as the ID column")
    else:
        raise ValueError("Could not find 'Match ID 18Char' column in the dataframe")

# Group by Match ID
grouped = df.groupby(id_column)

# Function to combine rows for each group
def combine_rows(group):
    # Start with the first row
    combined = group.iloc[0].copy()
    
    # For each column, if the first row has NaN, try to find a non-NaN value in the group
    for col in group.columns:
        if pd.isna(combined[col]):
            # Find the first non-NaN value in this column
            non_nan_values = group[col].dropna()
            if len(non_nan_values) > 0:
                combined[col] = non_nan_values.iloc[0]
    
    return combined

# Apply the function to each group
consolidated_data = grouped.apply(combine_rows).reset_index(drop=True)

print(f"Consolidated data shape: {consolidated_data.shape}")

# Save the consolidated data to a new Excel file
output_path = '/Users/dameesh/Desktop/MinMUDAC2025/Data/Test-Truncated-Restated-Consolidated.xlsx'
consolidated_data.to_excel(output_path, index=False)

print(f"Consolidated data saved to {output_path}") 